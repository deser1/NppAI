"""Python-only frozen vocabulary + learned start/exit + hard source cursor.

No prompt-name parser, supplied response prefix, or output rewriting is used.
The hard cursor and def-space trigger are explicit decoding assistance, not a
claim that the original soft pointer learned sequential copying.
"""
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from scripts import gated_copy_model as old
from scripts.resampled_name_training import digest


class SequentialCopyModel(nn.Module):
    def __init__(self, gated):
        super().__init__()
        self.base = gated.base
        self.copy_q, self.copy_k = gated.copy_q, gated.copy_k
        self.exit = nn.Linear(2 * self.base.dim, 1)
        nn.init.zeros_(self.exit.weight)
        nn.init.constant_(self.exit.bias, 4.0)
        self.base.requires_grad_(False)

    def hidden(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.base.max_seq_len:
            raise ValueError('invalid context')
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        x = self.base.tok_embeddings(tokens) + self.base.pos_embeddings(positions)
        causal = torch.tril(torch.ones(length, length, device=tokens.device, dtype=torch.bool))
        with torch.no_grad():
            for layer in self.base.layers:
                x = layer(x, causal[None, :, :])
            return self.base.norm(x).detach()

    def components(self, tokens, source):
        if source.shape != tokens.shape or source.dtype != torch.bool:
            raise ValueError('invalid prompt source mask')
        hidden = self.hidden(tokens)
        length = tokens.shape[1]
        causal = torch.tril(torch.ones(length, length, device=tokens.device, dtype=torch.bool))
        eligible = source[:, None, :] & causal[None, :, :] & (tokens[:, None, :] != 0)
        available = eligible.any(-1, keepdim=True)
        dummy = torch.arange(length, device=tokens.device)[None, None, :] == 0
        scores = self.copy_q(hidden) @ self.copy_k(hidden).transpose(-2, -1) / self.base.dim ** 0.5
        attention = scores.masked_fill(~(eligible | (~available & dummy)), float('-inf')).softmax(-1) * eligible
        return hidden, self.base.output(hidden).detach().softmax(-1), attention

    def exit_probability(self, query_hidden, source_hidden):
        return self.exit(torch.cat((query_hidden, source_hidden), dim=-1)).sigmoid()


def step_decision(model, tokens, mask, offset, output, state):
    """One causal decode step. State cursor uses absolute frame byte offsets."""
    hidden, vocab, attention = model.components(tokens, mask)
    base_byte = int(vocab[0, -1].argmax())
    event = None
    if state['mode'] == 'waiting' and output == b'def ':
        state['mode'] = 'copy'
        if not mask.any():
            state['mode'] = 'ended'
        else:
            state['cursor'] = offset + int(attention[0, -1].argmax())
    if state['mode'] == 'copy':
        cursor = state['cursor'] - offset
        valid = 0 <= cursor < tokens.shape[1] and bool(mask[0, cursor]) and int(tokens[0, cursor]) != 0
        if not valid:
            state['mode'] = 'ended'
            event = {'kind': 'boundary', 'cursor': state['cursor'], 'base_byte': base_byte, 'chosen': base_byte}
        else:
            probability = float(model.exit_probability(hidden[0, -1], hidden[0, cursor]))
            source_byte = int(tokens[0, cursor])
            exit_name = probability >= 0.5
            chosen = base_byte if exit_name else source_byte
            event = {'kind': 'exit' if exit_name else 'copy', 'cursor': state['cursor'],
                     'source_byte': source_byte, 'exit_probability': probability,
                     'base_byte': base_byte, 'chosen': chosen}
            if exit_name:
                state['mode'] = 'ended'
            else:
                state['cursor'] += 1
            return chosen, event
    return base_byte, event


def generate(model, prompt, budget=160, with_events=False):
    frame = ('[USER]: ' + prompt + '\n[AI]:\n').encode()
    if len(frame) > model.base.max_seq_len:
        raise ValueError('prompt exceeds context')
    left, right = old.source_bounds(prompt)
    tokens = list(frame)
    output, events = bytearray(), []
    state = {'mode': 'waiting', 'cursor': None}
    with torch.no_grad():
        for query in range(budget):
            offset = max(0, len(tokens) - model.base.max_seq_len)
            x = torch.tensor([tokens[offset:]], dtype=torch.long)
            mask = torch.tensor([[left <= i < right for i in range(offset, len(tokens))]], dtype=torch.bool)
            token, event = step_decision(model, x, mask, offset, bytes(output), state)
            if event is not None:
                events.append({'response_query': query, **event})
            if token == 0:
                break
            output.append(token)
            tokens.append(token)
    result = output.decode('utf-8', errors='replace')
    return (result, events, list(output)) if with_events else result


def export_bundle(model, path, source_model_id):
    from train_nppai import export_to_bin
    with tempfile.TemporaryDirectory() as directory:
        basepath = Path(directory) / 'base.nppai'
        export_to_bin(model.base, str(basepath))
        base = basepath.read_bytes()
    entries = {}
    for name, tensor in head_parameters(model).items():
        stream = io.BytesIO()
        np.save(stream, tensor.detach().cpu().numpy(), allow_pickle=False)
        entries[name + '.npy'] = stream.getvalue()
    head = old.zip_bytes(entries)
    manifest = {'format': 'nppai-sequential-copy', 'version': 1, 'native_compatible': False,
                'base_sha256': digest(base), 'head_sha256': digest(head), 'source_model_id': source_model_id}
    Path(path).write_bytes(old.zip_bytes({'base.nppai': base, 'head.npz': head,
                                       'manifest.json': json.dumps(manifest, sort_keys=True).encode()}))


def head_parameters(model):
    return {'copy_q': model.copy_q.weight, 'copy_k': model.copy_k.weight,
            'exit_weight': model.exit.weight, 'exit_bias': model.exit.bias}


def load_bundle(path):
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    entries = old.bounded_zip(Path(path).read_bytes(), ('base.nppai', 'head.npz', 'manifest.json'))
    manifest = json.loads(entries['manifest.json'])
    if (set(manifest) != {'format', 'version', 'native_compatible', 'base_sha256', 'head_sha256', 'source_model_id'} or
            manifest['format'] != 'nppai-sequential-copy' or type(manifest['version']) != int or
            manifest['version'] != 1 or manifest['native_compatible'] is not False or
            not isinstance(manifest['source_model_id'], str)):
        raise ValueError('unsupported sequential bundle')
    if digest(entries['base.nppai']) != manifest['base_sha256'] or digest(entries['head.npz']) != manifest['head_sha256']:
        raise ValueError('bundle digest mismatch')
    weights = load_weights(None, raw=entries['base.nppai'])
    arrays = [weights['embedding'], weights['position'], weights['norm'], weights['output'],
              *(a for layer in weights['layers'] for a in layer.values())]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('nonfinite base weights')
    model = SequentialCopyModel(old.GatedCopyModel(torch_model(weights)))
    parameters = head_parameters(model)
    head = old.bounded_zip(entries['head.npz'], tuple(name + '.npy' for name in parameters))
    for name, parameter in parameters.items():
        array = old.read_head_array(head[name + '.npy'], parameter.shape)
        with torch.no_grad():
            parameter.copy_(torch.from_numpy(array))
    return model.eval(), manifest
