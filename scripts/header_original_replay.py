"""Preselected 50% original-example replay versus mixed-name-only header training."""
import argparse
import json
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.header_name_length import (build, read_sources, digest, encoded, evaluate, paired,
    verify_initial, audit as audit_headers, SOURCE_RUN, STEPS)
from scripts.prompt_robustness import augment_training_records
from scripts.train_identifier_headers import initialize_header_weights

POLICIES = ('no_replay', 'half_original')


def pools(dataset):
    renamed, development, base = build(dataset, 'mixed_lengths')
    sources, _ = read_sources(dataset)
    original = augment_training_records(sources['train'], 'four_forms')
    original = [{**r, 'id': 'original-replay::' + r['id'], 'name_length': len(r['function'])} for r in original]
    names = {r['function'] for r in original + renamed}
    if len(original) != 6144 or len(renamed) != 18432 or len(names) != 2048:
        raise ValueError('replay source pool/count differs')
    for records in development.values():
        if names & {r['function'] for r in records}:
            raise ValueError('replay training/development names overlap')
    return original, renamed, development, base


def metadata(original, renamed, policy):
    if policy not in POLICIES:
        raise ValueError('unknown replay policy')
    return {'policy': policy, 'original_pool_records': len(original),
            'renamed_pool_records': len(renamed), 'original_pool_sha256': digest(encoded(original)),
            'renamed_pool_sha256': digest(encoded(renamed)),
            'original_examples_per_batch': 2 if policy == 'half_original' else 0,
            'renamed_examples_per_batch': 2 if policy == 'half_original' else 4,
            'original_example_draws': STEPS * (2 if policy == 'half_original' else 0),
            'renamed_example_draws': STEPS * (2 if policy == 'half_original' else 4),
            'limits': '50% of examples per batch, not 50% of supervised bytes/loss. Same renamed pool and fixed update/batch budgets; replay reduces renamed exposures. Only training examples may be replayed.'}


def draw_batch(original, renamed, policy, rng):
    if policy == 'no_replay':
        return [('renamed', r) for r in rng.sample(renamed, 4)]
    if policy == 'half_original':
        batch = [('original', r) for r in rng.sample(original, 2)] + [('renamed', r) for r in rng.sample(renamed, 2)]
        rng.shuffle(batch)
        return batch
    raise ValueError('unknown replay policy')


def retention(evaluation):
    result = {}
    for split in ('validation/original', 'test/original'):
        pairs = paired(evaluation)[split]['exact_header']
        result[split] = {'baseline_exact': pairs['both'] + pairs['lost'], 'retained_exact': pairs['both'],
                        'lost_exact': pairs['lost'], 'new_exact': pairs['gained']}
    result['all_baseline_exact_retained'] = all(result[s]['lost_exact'] == 0 for s in ('validation/original', 'test/original'))
    result['long_names_exact_present_in_each_development_split'] = all(
        evaluation['after'][split]['summary']['name_length'][length]['exact_header'] > 0
        for split in ('validation/renamed', 'test/renamed') for length in ('12', '15'))
    return result


def audit(report, dataset):
    evidence = audit_headers(report, dataset)
    original, renamed, _, _ = pools(dataset)
    expected = metadata(original, renamed, report['replay']['policy'])
    if report['replay'] != expected or report['observed_training_draws'] != {
        'original': expected['original_example_draws'], 'renamed': expected['renamed_example_draws']}:
        raise ValueError('replay policy/pool/draw counts differ')
    actual = retention(report['evaluation'])
    if report['retention'] != actual:
        raise ValueError('original retention summary differs')
    return {**evidence, 'replay': expected, 'retention': actual,
            'limits': 'Evidence audit and development retention diagnostics; not full-code correctness or model promotion.'}


def run(source, initial, dataset, output, policy):
    original, renamed, development, base = pools(dataset)
    replay = metadata(original, renamed, policy)
    initial_model, provenance = verify_initial(source, initial, dataset)
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
    sample_pools = {key: [sample(r, 256) for r in rows] for key, rows in (('original', original), ('renamed', renamed))}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    draws = {'original': 0, 'renamed': 0}
    model.train()
    for step in range(STEPS):
        selected = draw_batch(sample_pools['original'], sample_pools['renamed'], policy, rng)
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
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / 'NppAI-original-replay-header.nppai'
    export_to_bin(model, str(checkpoint))
    reloaded = torch_model(load_weights(checkpoint))
    with torch.no_grad():
        after = {split: evaluate(reloaded, records, generate) for split, records in development.items()}
        for split, records in development.items():
            if [c['output'] for c in after[split]['cases']] != [generate(model, r['prompt'], 256) for r in records]:
                raise RuntimeError('export/reload changed header outputs')
    report = {'source_run_id': SOURCE_RUN, 'source': provenance, 'arm': 'mixed_lengths',
              'dataset': base, 'replay': replay, 'observed_training_draws': draws,
              'initialization': initialization, 'additional_steps': STEPS, 'total_header_updates': 26000,
              'optimizer': {'name': 'AdamW', 'reset': True, 'learning_rate': 0.001, 'batch_size': 4, 'clip_norm': 1.0, 'seed': 42},
              'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'threads': 2},
              'context': 256, 'new_tokens_per_call': 160, 'reload_outputs_identical': True,
              'model_sha256': digest(checkpoint.read_bytes()), 'evaluation': {'before': before, 'after': after},
              'limits': 'Header-only fixed replay ratio, one seed and exposed development tasks. No fresh-probe access, body training, generated-source execution, native conversion or checkpoint promotion. Same starting20k checkpoint and6000 additional updates. Fixed data/seed are not bit-identical retraining guarantees; within-run reload equality is separately checked.'}
    report['paired_before_after'] = paired(report['evaluation'])
    report['retention'] = retention(report['evaluation'])
    evidence = audit(report, dataset)
    raw = encoded(report)
    (destination / 'replay-report.json').write_bytes(raw)
    evidence['source_report_sha256'] = digest(raw)
    (destination / 'replay-audit.json').write_bytes(encoded(evidence))
    for name, records in (('original-training', original), ('renamed-training', renamed), ('development', development)):
        (destination / (name + '.json')).write_bytes(encoded(records))
    (destination / 'manifest.json').write_bytes(encoded({'base': base, 'replay': replay}))
    for folder in ('header-source', 'initial-header-source', 'initial-body-source'):
        shutil.copytree(Path(source) / folder, destination / folder, dirs_exist_ok=True)
    shutil.copy2(Path(source) / 'staged-report.json', destination / 'source-continuation-report.json')
    print(json.dumps({'policy': policy, 'source': provenance, 'audit': evidence,
        'summary': {stage: {split: group['summary'] for split, group in groups.items()} for stage, groups in report['evaluation'].items()},
        'retention': report['retention'], 'paired_before_after': report['paired_before_after']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'initial', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--policy', choices=POLICIES, required=True)
    args = parser.parse_args()
    run(args.source, args.initial, args.dataset, args.output, args.policy)
