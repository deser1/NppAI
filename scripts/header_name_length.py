"""Matched header-only name-length experiment using instruction-v3 development data."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import string
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_staged_generation import check_source, valid_header
from scripts.prompt_robustness import augment_training_records
from scripts.run_wording_continuation import verify_source, check_continuation, TASKS
from scripts.train_identifier_headers import header_records, initialize_header_weights
from scripts.diagnose_identifier_copy import header_name

SOURCE_RUN = 38618406741
STEPS = 6000
ARMS = ('short_control', 'mixed_lengths')
METRICS = ('valid_header', 'requested_name', 'requested_signature', 'exact_header')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8')


def read_sources(dataset):
    root = Path(dataset)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    splits = {}
    for split, metadata in manifest['splits'].items():
        raw = (root / (split + '.json')).read_bytes()
        if digest(raw) != metadata['sha256']:
            raise ValueError('frozen source split differs')
        splits[split] = header_records(json.loads(raw))
    return splits, manifest


def name_for(split, original, slot, length):
    # Same first candidate across arms; cp_ makes this experiment's namespace
    # disjoint from the existing fn_ task names without reading fresh-probe-v1.
    seed = digest(f'header-length-v1/{split}/{original}/{slot}'.encode())
    rng = random.Random(seed)
    return 'cp_' + ''.join(rng.choices(string.ascii_lowercase + string.digits, k=length - 3))


def rename(record, name, slot):
    old = record['function']
    pattern = r'\b' + re.escape(old) + r'\b'
    prompt, answer = re.sub(pattern, name, record['prompt']), re.sub(pattern, name, record['answer'])
    if prompt == record['prompt'] or not answer.startswith('def ' + name + '('):
        raise ValueError('rename did not preserve declared function')
    return {**record, 'id': record['id'] + '::name-' + str(slot),
            'source_id': record['id'], 'function': name, 'prompt': prompt, 'answer': answer,
            'name_length': len(name)}


def derived(records, split, lengths):
    result = [rename(r, name_for(split, r['function'], slot, length), slot)
              for r in records for slot, length in enumerate(lengths)]
    return augment_training_records(result, 'four_forms')


def build(dataset, arm):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    source, manifest = read_sources(dataset)
    training = derived(source['train'], 'train', (9, 9, 9) if arm == 'short_control' else (9, 12, 15))
    development = {}
    for split in ('validation', 'test'):
        development[split + '/original'] = [{**r, 'name_length': len(r['function']), 'prompt_variant': 'original'} for r in source[split]]
        development[split + '/renamed'] = derived(source[split], split, (9, 12, 15))
    names = {r['function'] for r in training}
    # Names repeat only within the intended source prompt views.
    if len(names) != 512 * 3:
        raise ValueError('generated training names collided')
    for records in development.values():
        if names & {r['function'] for r in records}:
            raise ValueError('training/development names overlap')
    all_records = training + [r for rows in development.values() for r in rows]
    for r in all_records:
        size = len(('[USER]: ' + r['prompt'] + '\n[AI]:\n' + r['answer']).encode()) + 1
        if size > 256:
            raise ValueError('full header sample exceeds context')
    metadata = {'version': 1, 'arm': arm, 'source_manifest': manifest,
                'training_records': len(training), 'training_names': len(names),
                'training_sha256': digest(encoded(training)),
                'development_sha256': digest(encoded(development)),
                'training_lengths': [9] if arm == 'short_control' else [9, 12, 15],
                'development_lengths': [9, 12, 15], 'wording_policy': 'four_forms',
                'limits': 'Only instruction-v3 sources. Development prompts/families were previously exposed; remapped names are new development probes, not a fresh benchmark. Original argument lists are unchanged. 3 name candidates per source name, 4 wording views per source record; correlated views, not independent tasks. cp_ namespace is new in both arms. No fresh-probe-v1 access.'}
    return training, development, metadata


def signature(header):
    if not valid_header(header):
        return None
    node = ast.parse(header + '    pass\n').body[0]
    return node.name, ast.dump(node.args)


def score(record, output):
    return {'valid_header': valid_header(output), 'requested_name': header_name(output) == record['function'],
            'requested_signature': signature(output) == signature(record['answer']),
            'exact_header': output == record['answer']}


def summarize(cases):
    result = {'total': len(cases), **{m: sum(c['metrics'][m] for c in cases) for m in METRICS}}
    for field in ('name_length', 'prompt_variant'):
        result[field] = {str(value): {'total': sum(c[field] == value for c in cases),
            **{m: sum(c['metrics'][m] for c in cases if c[field] == value) for m in METRICS}}
            for value in sorted({c[field] for c in cases})}
    return result


def evaluate(model, records, generate):
    cases = []
    for record in records:
        output = generate(model, record['prompt'], 256)
        cases.append({**record, 'output': output, 'metrics': score(record, output)})
    return {'summary': summarize(cases), 'cases': cases}


def audit(report, dataset):
    training, development, metadata = build(dataset, report['arm'])
    if report['dataset'] != metadata or report['source_run_id'] != SOURCE_RUN or report['additional_steps'] != STEPS:
        raise ValueError('experiment metadata differs')
    if set(report['evaluation']) != {'before', 'after'}:
        raise ValueError('evaluation arms differ')
    for stage in ('before', 'after'):
        if set(report['evaluation'][stage]) != set(development):
            raise ValueError('development splits differ')
        for split, records in development.items():
            group = report['evaluation'][stage][split]
            cases = group['cases']
            if len(cases) != len(records) or len({c['id'] for c in cases}) != len(records):
                raise ValueError('missing or duplicate cases')
            expected = {r['id']: r for r in records}
            for case in cases:
                record = expected.get(case['id'])
                if record is None or any(case.get(k) != v for k, v in record.items()):
                    raise ValueError('development reference/prompt differs')
                if case['metrics'] != score(record, case['output']):
                    raise ValueError('saved header metrics differ')
            if group['summary'] != summarize(cases):
                raise ValueError('saved header summary differs')
    pairs = paired(report['evaluation'])
    if report['paired_before_after'] != pairs:
        raise ValueError('saved paired outcomes differ')
    return {'valid': True, 'training_sha256': metadata['training_sha256'],
            'development_sha256': metadata['development_sha256'],
            'limits': 'Evidence integrity only. Header copying is not body correctness or a fresh independent benchmark.'}


def paired(evaluation):
    result = {}
    for split in evaluation['before']:
        before = {c['id']: c for c in evaluation['before'][split]['cases']}
        after = {c['id']: c for c in evaluation['after'][split]['cases']}
        if before.keys() != after.keys():
            raise ValueError('unpaired development cases')
        result[split] = {metric: {label: sum(bool(before[k]['metrics'][metric]) == b and bool(after[k]['metrics'][metric]) == a for k in before)
            for label, b, a in (('both', True, True), ('lost', True, False), ('gained', False, True), ('neither', False, False))}
            for metric in METRICS}
    return result


def verify_initial(source, initial, dataset):
    origins = verify_source(initial, dataset)
    preserved = verify_source(source, dataset, 'initial-')
    if any(origins[k]['sha256'] != preserved[k]['sha256'] for k in TASKS):
        raise ValueError('preserved starting pair differs')
    folder, model_name, report_name, manifest_key = TASKS['headers']
    model = Path(source) / folder / model_name
    raw = (Path(source) / folder / report_name).read_bytes()
    report = json.loads(raw)
    manifest = json.loads((Path(dataset) / 'manifest.json').read_bytes())
    check_source(report, model.read_bytes(), manifest, manifest_key, 10000)
    check_continuation(report, origins['headers']['sha256'], 'four_forms')
    cached = json.loads((Path(source) / 'staged-report.json').read_bytes())
    if cached.get('source_run_id') != SOURCE_RUN or cached['source_header_report'] != report:
        raise ValueError('fixed continuation source differs')
    return model, {'model_sha256': digest(model.read_bytes()), 'report_sha256': digest(raw),
                   'prior_header_updates': 20000, 'source_policy': 'four_forms'}


def run(source, initial, dataset, output, arm):
    training, development, metadata = build(dataset, arm)
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
    model.train()
    samples = [sample(r, 256) for r in training]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for step in range(STEPS):
        batch = rng.sample(samples, 4)
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
    checkpoint = destination / 'NppAI-name-length-header.nppai'
    export_to_bin(model, str(checkpoint))
    reloaded = torch_model(load_weights(checkpoint))
    with torch.no_grad():
        after = {split: evaluate(reloaded, records, generate) for split, records in development.items()}
        for split, records in development.items():
            if [c['output'] for c in after[split]['cases']] != [generate(model, r['prompt'], 256) for r in records]:
                raise RuntimeError('export/reload changed header outputs')
    report = {'source_run_id': SOURCE_RUN, 'source': provenance, 'arm': arm, 'dataset': metadata,
              'initialization': initialization, 'additional_steps': STEPS, 'total_header_updates': 26000,
              'optimizer': {'name': 'AdamW', 'reset': True, 'learning_rate': 0.001, 'batch_size': 4, 'clip_norm': 1.0, 'seed': 42},
              'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'threads': 2},
              'context': 256, 'new_tokens_per_call': 160, 'reload_outputs_identical': True,
              'model_sha256': digest(checkpoint.read_bytes()), 'evaluation': {'before': before, 'after': after},
              'limits': 'Header-only training/evaluation; no body retraining, complete-code quality claim, generated-code execution, native conversion or production promotion. Both arms change identifier namespace and lexemes. Same initialization, steps, record counts and wording policy; token budgets differ with name length. Single seed, exposed development source families. No fresh-probe access.'}
    report['paired_before_after'] = paired(report['evaluation'])
    evidence = audit(report, dataset)
    raw = encoded(report)
    (destination / 'name-length-report.json').write_bytes(raw)
    evidence['source_report_sha256'] = digest(raw)
    (destination / 'name-length-audit.json').write_bytes(encoded(evidence))
    (destination / 'training.json').write_bytes(encoded(training))
    (destination / 'development.json').write_bytes(encoded(development))
    (destination / 'manifest.json').write_bytes(encoded(metadata))
    for folder in ('header-source', 'initial-header-source', 'initial-body-source'):
        shutil.copytree(Path(source) / folder, destination / folder, dirs_exist_ok=True)
    shutil.copy2(Path(source) / 'staged-report.json', destination / 'source-continuation-report.json')
    print(json.dumps({'arm': arm, 'source': provenance, 'audit': evidence,
                     'summary': {stage: {split: group['summary'] for split, group in groups.items()} for stage, groups in report['evaluation'].items()},
                     'paired_before_after': report['paired_before_after']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'initial', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    run(args.source, args.initial, args.dataset, args.output, args.arm)
