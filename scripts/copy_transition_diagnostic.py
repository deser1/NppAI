"""Frozen pointer/name-transition readout. No training, repair or code execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import namespace_retention as pilot

ROOT = Path(__file__).resolve().parents[1]
quality, base = pilot.quality, pilot.base
MODELS = ('initial', *pilot.ARMS)
SOURCE_RUN = 39019625587
ARTIFACT_ID = 12096660315
ARCHIVE_SHA256 = 'f3eee44a4c82751186733de2c6a67932b094a531a18558959a6c7e2b8adb82f2'
SOURCE_REPORT_SHA256 = '9d3b7d342c0dfb74a6dda0ad23ba3d2f5e8a705b8705268612c29a8c16c8d755'
INITIAL_CASES_SHA256 = '42ed1e77485f1b20db4f5f1a227057effc7529ac9dc24c7a072273183683a5e0'
SPEC = {'version': 1, 'source_run': SOURCE_RUN, 'artifact_id': ARTIFACT_ID,
        'archive_sha256': ARCHIVE_SHA256, 'report_sha256': SOURCE_REPORT_SHA256,
        'initial_training_cases_sha256': INITIAL_CASES_SHA256, 'models': list(MODELS),
        'profile': quality.PROFILE, 'arithmetic': 'native FP32 gated readout',
        'full_budget': 160, 'def_scaffold_budget': 32,
        'conditional_targets': 'function-name bytes and opening parenthesis under gold def/name prefix',
        'limits': 'Frozen checkpoints, training-family references only. No optimization or code execution. '
                  'Gold-prefix, def-scaffolded and full autonomous modes are separate. '
                  'Peak pointer positions summarize a soft distribution, not a discrete cursor or causal ablation. '
                  'No task-generalization or promotion claim.'}


def cases(dataset):
    probes = pilot.namespace_probe(dataset)
    result = [{'group': group, 'index': i, 'role': 'new_alias', 'record': row}
              for group, rows in probes.items() for i, row in enumerate(rows)]
    raw = (ROOT / 'docs/evidence/namespace-retention/before-train-original.json').read_bytes()
    if quality.digest(raw) != INITIAL_CASES_SHA256:
        raise ValueError('initial training snapshot differs')
    cached = json.loads(raw)
    original = []
    for batch in base.schedule(base.training_data(dataset), 'resampled'):
        for pool, row in batch:
            if pool == 'original' and len(original) < 16:
                original.append(row)
    for i, (row, prior) in enumerate(zip(original, cached)):
        if prior['index'] != i or prior['id'] != row['id'] or prior['metrics'] != base.score(row, prior['output']):
            raise ValueError('initial training identity differs')
        if prior['metrics']['exact_response']:
            result.append({'group': 'train/original', 'index': i, 'role': 'initial_exact_training', 'record': row})
    if len(result) != 68 or len(original) != 16 or len(cached) != 16:
        raise ValueError('fixed diagnostic case count differs')
    return result


def prepare(archive, output):
    if quality.digest(archive.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError('pinned diagnostic archive differs')
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    states = {}
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('diagnostic archive CRC differs')
        raw = bundle.read('report.json')
        if quality.digest(raw) != SOURCE_REPORT_SHA256:
            raise ValueError('diagnostic source report differs')
        report = json.loads(raw)
        initial = bundle.read('source/both64.nppai')
        if quality.digest(initial) != quality.CHECKPOINTS['both64']:
            raise ValueError('diagnostic initial weights differ')
        paths['initial'] = output / 'initial.nppai'
        paths['initial'].write_bytes(initial)
        for arm in pilot.ARMS:
            raw = bundle.read(f'{arm}/0/experimental.nppaicopy')
            if (raw != bundle.read(f'{arm}/1/experimental.nppaicopy') or
                    quality.digest(raw) != report['arms'][arm]['identity']['checkpoint_sha256']):
                raise ValueError('diagnostic checkpoint or repeated bytes differ')
            paths[arm] = output / (arm + '.nppaicopy')
            paths[arm].write_bytes(raw)
            trace = bundle.read(f'{arm}/0/trace.jsonl')
            if trace != bundle.read(f'{arm}/1/trace.jsonl') or quality.digest(trace) != report['arms'][arm]['identity']['trace_sha256']:
                raise ValueError('diagnostic source trace differs')
            states[arm] = json.loads(trace.splitlines()[-1])['parameters']
        start = json.loads(bundle.read('fixed/0/trace.jsonl').splitlines()[0])
        states['initial'] = start['parameters']
    return paths, report, states


def plan(record):
    from scripts.aligned_name_copy import alignment
    prefix, raw = base.frame(record)
    mapping = alignment(record)
    name = record['function'].encode()
    source = mapping[0][1]
    end = source + len(name)
    while raw[end] in b' \t\r\n\f\v':
        end += 1
    if raw[end] != 40:
        raise ValueError('source name terminator differs')
    return prefix, name + b'(', [s for _, s in mapping] + [end], source


def point(parts, tokens, window_offset, expected=None, source=None):
    import torch
    logits, pointer, gate, attention, mixture = parts
    vocab, copy, mixed = logits[0, -1].softmax(-1), pointer[0, -1], mixture[0, -1]
    g = float(gate[0, -1, 0])
    predicted = int(mixed.clamp_min(torch.finfo(mixed.dtype).tiny).log().argmax())
    peak = int(attention[0, -1].argmax())
    item = {'predicted': predicted, 'vocab_predicted': int(vocab.argmax()), 'pointer_predicted': int(copy.argmax()),
            'gate': g, 'peak_position': peak + window_offset, 'peak_byte': int(tokens[0, peak]),
            'peak_probability': float(attention[0, -1, peak]),
            'predicted_probability': float(mixed[predicted]),
            'predicted_vocab_probability': float(vocab[predicted]),
            'predicted_pointer_probability': float(copy[predicted])}
    if expected is not None:
        item.update(target=expected, target_probability=float(mixed[expected]),
                    vocab_target_probability=float(vocab[expected]), pointer_target_probability=float(copy[expected]))
    if source is not None:
        relative = source - window_offset
        item.update(aligned_source=source,
                    aligned_probability=float(attention[0, -1, relative]) if 0 <= relative < tokens.shape[1] else 0.)
    return item


def step(model, tokens, mask, expected=None, source=None):
    import torch
    width = model.base.max_seq_len
    x = torch.tensor([tokens[-width:]], dtype=torch.long)
    eligible = torch.tensor([mask[-width:]], dtype=torch.bool)
    parts = model.components(x, eligible)
    return point(parts, x, len(tokens) - x.shape[1], expected, source)


def phase(record, offset):
    answer = record['answer'].encode()
    if offset >= len(answer):
        return 'eos'
    if offset < 4:
        return 'def_scaffold'
    if offset < 4 + len(record['function'].encode()):
        return 'name'
    if offset == 4 + len(record['function'].encode()):
        return 'name_termination'
    return 'signature' if offset <= answer.index(b'\n') else 'body'


def full_generate(model, record):
    """Same greedy/context/EOS semantics as gated_copy_model.generate, raw byte drift."""
    from scripts.gated_copy_model import source_bounds
    prefix, target, sources, _ = plan(record)
    tokens = list(prefix)
    left, right = source_bounds(record['prompt'])
    mask = [left <= i < right for i in range(len(tokens))]
    answer = record['answer'].encode() + b'\0'
    output, difference = [], None
    for offset in range(SPEC['full_budget']):
        expected = answer[offset] if difference is None and offset < len(answer) else None
        source = sources[offset - 4] if 4 <= offset < 4 + len(target) else None
        item = step(model, tokens, mask, expected, source)
        token = item['predicted']
        if difference is None and token != expected:
            difference = {'offset': offset, 'phase': phase(record, offset), 'point': item}
        if token == 0:
            return {'output': bytes(output).decode('utf-8', errors='replace'), 'bytes': output,
                    'finish': 'eos', 'first_difference': difference}
        output.append(token)
        tokens.append(token)
        mask.append(False)
    return {'output': bytes(output).decode('utf-8', errors='replace'), 'bytes': output,
            'finish': 'budget', 'first_difference': difference}


def conditional(model, record):
    import torch
    from scripts.gated_copy_model import source_mask
    prefix, target, sources, start = plan(record)
    raw = prefix + b'def ' + target
    x = torch.tensor([list(raw[:-1])], dtype=torch.long)
    with torch.no_grad():
        parts = model.components(x, source_mask(x, record['prompt']))
        rows = []
        for offset, (byte, source) in enumerate(zip(target, sources)):
            query = len(prefix) + 3 + offset
            query_parts = tuple(p[:, query:query+1] for p in parts)
            # Attention's key axis still spans the full sequence.
            item = point(query_parts, x, 0, byte, source)
            truncated = x[:, :query+1]
            shorter = model.components(truncated, source_mask(truncated, record['prompt']))
            delta = float((shorter[-1][0, -1] - parts[-1][0, query]).abs().max())
            if (not torch.allclose(shorter[-1][0, -1], parts[-1][0, query], atol=5e-5, rtol=1e-4) or
                    int(shorter[-1][0, -1].argmax()) != item['predicted']):
                raise ValueError('conditional future-token causality differs')
            item.update(offset=offset, causal_max_delta=delta)
            item['peak_in_name'] = start <= item['peak_position'] < start + len(target) - 1
            item['peak_delta'] = item['peak_position'] - rows[-1]['peak_position'] if rows else None
            rows.append(item)
    return rows


def scaffolded(model, record):
    from scripts.gated_copy_model import source_bounds
    prefix, target, sources, start = plan(record)
    tokens = list(prefix + b'def ')
    left, right = source_bounds(record['prompt'])
    mask = [left <= i < right for i in range(len(tokens))]
    output, trace = [], []
    for offset in range(SPEC['def_scaffold_budget']):
        aligned = bytes(output) == target[:offset] and offset < len(target)
        item = step(model, tokens, mask, target[offset] if aligned else None, sources[offset] if aligned else None)
        item.update(offset=offset, reference_prefix_intact=aligned,
                    peak_in_name=start <= item['peak_position'] < start + len(target) - 1)
        item['peak_delta'] = item['peak_position'] - trace[-1]['peak_position'] if trace else None
        trace.append(item)
        token = item['predicted']
        output.append(token)
        if token in (0, 10, 40):
            break
        tokens.append(token)
        mask.append(False)
    return {'bytes': output, 'trace': trace, 'exact_name': bytes(output) == target}


def initial_at_difference(initial, record, difference):
    from scripts.gated_copy_model import source_bounds
    if difference is None:
        return None
    prefix, target, sources, _ = plan(record)
    offset = difference['offset']
    tokens = list(prefix + record['answer'].encode()[:offset])
    left, right = source_bounds(record['prompt'])
    expected = (record['answer'].encode() + b'\0')[offset]
    source = sources[offset - 4] if 4 <= offset < 4 + len(target) else None
    return step(initial, tokens, [left <= i < right for i in range(len(tokens))], expected, source)


def summarize(items):
    alias = [r for r in items if r['role'] == 'new_alias']
    drift = [r for r in items if r['role'] == 'initial_exact_training']
    from collections import Counter
    return {'aliases': len(alias), 'autonomous_exact': sum(r['full_metrics']['exact_response'] for r in alias),
            'scaffolded_exact_names': sum(r['scaffolded']['exact_name'] for r in alias),
            'conditional_name_bytes': sum(len(r['conditional']) - 1 for r in alias),
            'conditional_correct_name_bytes': sum(p['predicted'] == p['target'] for r in alias for p in r['conditional'][:-1]),
            'conditional_correct_terminators': sum(r['conditional'][-1]['predicted'] == 40 for r in alias),
            'conditional_peak_aligned_name_bytes': sum(p['peak_position'] == p['aligned_source'] for r in alias for p in r['conditional'][:-1]),
            'conditional_vocab_correct_mix_wrong': sum(p['vocab_predicted'] == p['target'] and p['predicted'] != p['target'] for r in alias for p in r['conditional'][:-1]),
            'conditional_pointer_correct_mix_wrong': sum(p['pointer_predicted'] == p['target'] and p['predicted'] != p['target'] for r in alias for p in r['conditional'][:-1]),
            'conditional_peak_adjacent_pairs': sum(b['peak_position'] == a['peak_position'] + 1
                for r in alias for a, b in zip(r['conditional'][:-2], r['conditional'][1:-1])),
            'conditional_peak_pairs': sum(max(0, len(r['conditional']) - 2) for r in alias),
            'scaffolded_peak_adjacent_pairs': sum(b['peak_delta'] == 1 for r in alias
                for a, b in zip(r['scaffolded']['trace'], r['scaffolded']['trace'][1:]) if a['peak_in_name'] and b['peak_in_name']),
            'scaffolded_peak_in_name_pairs': sum(1 for r in alias
                for a, b in zip(r['scaffolded']['trace'], r['scaffolded']['trace'][1:]) if a['peak_in_name'] and b['peak_in_name']),
            'scaffolded_first_errors': dict(Counter(next((str(p['offset']) for p in r['scaffolded']['trace']
                if p['reference_prefix_intact'] and p['predicted'] != p.get('target')), 'none') for r in alias)),
            'original_exact_retained': sum(r['full_metrics']['exact_response'] for r in drift),
            'original_first_error_phases': dict(Counter(r['full']['first_difference']['phase']
                for r in drift if r['full']['first_difference'] is not None)),
            'original_vocab_correct_at_error': sum(r['full']['first_difference']['point']['vocab_predicted'] ==
                r['full']['first_difference']['point']['target'] for r in drift if r['full']['first_difference'] is not None)}


def run(archive, dataset, output):
    import torch
    from scripts.training_repro import configure, provenance, state_digest
    from scripts.gated_copy_model import load_bundle
    from scripts.check_dense_runtime_parity import load_weights
    protocol_raw = (ROOT / 'evaluation/copy-transition-v1.json').read_bytes()
    selected = cases(dataset)
    if json.loads(protocol_raw) != {'spec': SPEC, 'cases_sha256': quality.digest(quality.encoded(selected))}:
        raise ValueError('diagnostic protocol differs')
    paths, cached, states = prepare(archive, output / 'weights')
    configure(torch, quality.PROFILE)
    initial = pilot.previous.initial(load_weights(paths['initial'])).eval()
    models = {'initial': initial, **{arm: load_bundle(paths[arm])[0] for arm in pilot.ARMS}}
    if any(state_digest(model) != states[name] for name, model in models.items()):
        raise ValueError('diagnostic loaded parameter identity differs')
    report = {'protocol_sha256': quality.digest(protocol_raw), 'spec': SPEC,
              'cases_sha256': quality.digest(quality.encoded(selected)), 'runtime': provenance(torch, quality.PROFILE),
              'models': {}, 'promotion': False}
    for name, model in models.items():
        before = state_digest(model)
        items = []
        with torch.no_grad():
            for case in selected:
                record = case['record']
                full = full_generate(model, record)
                expected = (cached['before'] if name == 'initial' else cached['arms'][name]['after'])[case['group']][case['index']]
                if expected['id'] != record['id'] or expected['metrics'] != base.score(record, expected['output']):
                    raise ValueError('cached training-case identity differs')
                item = {k: v for k, v in case.items() if k != 'record'}
                item.update(id=record['id'], full=full, full_metrics=base.score(record, full['output']),
                            cached_output_equal=full['output'] == expected['output'], conditional=conditional(model, record),
                            scaffolded=scaffolded(model, record))
                item['initial_at_difference'] = initial_at_difference(initial, record, full['first_difference']) if case['role'] == 'initial_exact_training' else None
                items.append(item)
        report['models'][name] = {'cases': items, 'summary': summarize(items),
                                  'checkpoint_sha256': quality.digest(paths[name].read_bytes()),
                                  'parameters_before': before, 'parameters_after': state_digest(model),
                                  'parameters_unchanged': state_digest(model) == before}
        (output / 'report.json').write_bytes(quality.encoded(report))
        if not report['models'][name]['parameters_unchanged'] or any(not r['cached_output_equal'] for r in items):
            raise ValueError('frozen diagnostic replay/parameters differ; raw report preserved')
        print(json.dumps({'model': name, 'summary': report['models'][name]['summary']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.archive, args.dataset, args.output)
