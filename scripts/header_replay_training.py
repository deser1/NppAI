"""Shared fixed-budget header replay runner for development experiments."""
import random
import sys
from scripts.header_name_length import STEPS, evaluate
from scripts.header_original_replay import draw_batch
from scripts.train_identifier_headers import initialize_header_weights


def train(original, renamed, development, initial_model, checkpoint):
    import torch
    import torch.nn.functional as F
    from train_nppai import NppAIModel, export_to_bin
    from scripts.train_broader_instruction_experiment import sample
    from scripts.train_instruction_experiment import generate
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    torch.manual_seed(42)
    torch.set_num_threads(2)
    rng = random.Random(42)
    model = NppAIModel(256, 64, 128, 2, 256)
    initialization = initialize_header_weights(model, initial_model)
    model.eval()
    with torch.no_grad():
        before = {split: evaluate(model, records, generate) for split, records in development.items()}
    samples = {key: [sample(r, 256) for r in rows] for key, rows in (('original', original), ('renamed', renamed))}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    draws = {'original': 0, 'renamed': 0}
    model.train()
    for step in range(STEPS):
        selected = draw_batch(samples['original'], samples['renamed'], 'half_original', rng)
        for label, _ in selected:
            draws[label] += 1
        batch = [item for _, item in selected]
        x = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True, padding_value=-100)
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == STEPS - 1:
            print(f'step={step+1} loss={float(loss.detach()):.6f}', flush=True)
    model.eval()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    export_to_bin(model, str(checkpoint))
    reloaded = torch_model(load_weights(checkpoint))
    with torch.no_grad():
        after = {split: evaluate(reloaded, rows, generate) for split, rows in development.items()}
        for split, rows in development.items():
            if [c['output'] for c in after[split]['cases']] != [generate(model, r['prompt'], 256) for r in rows]:
                raise RuntimeError('export/reload changed header outputs')
    return {'initialization': initialization, 'observed_training_draws': draws,
            'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'threads': 2},
            'reload_outputs_identical': True, 'evaluation': {'before': before, 'after': after}}
