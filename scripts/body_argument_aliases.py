"""Matched full-response continuation; parser-assisted and autonomous evaluations separate."""
import argparse
import json
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.header_argument_aliases import build as header_pools, POLICIES
from scripts.header_original_replay import draw_batch, metadata as replay_metadata
from scripts.header_name_length import digest, encoded, verify_initial, STEPS, SOURCE_RUN
from scripts.evaluate_argument_assistance import rename_code
from scripts.evaluate_task_signature_wording import task_header
from scripts.evaluate_staged_generation import check_source, header_fits
from scripts.run_wording_continuation import verify_source, check_continuation
from scripts.diagnose_header_replay import score, METRICS

ARMS = ('task_signature_assisted', 'single_model')
OPTIMIZER = {'name': 'AdamW', 'reset': True, 'learning_rate': 0.001, 'batch_size': 4, 'clip_norm': 1.0, 'seed': 42}
LIMITS = ('Full-response training, not header-only. Same20k body source and6000 updates per policy; '
          '50% original examples per batch, not bytes/loss. One seed and1100 correlated exposed development views. '
          'Assisted signatures are supplied by a prompt-only parser, not autonomous copying. '
          'Reference exact/AST matches are not execution or semantic correctness. No generated-source execution, '
          'fresh-probe access, native conversion, checkpoint selection or promotion. Fixed seeds do not guarantee bit-identical retraining.')


def frame_size(record):
    return len(('[USER]: ' + record['prompt'] + '\n[AI]:\n' + record['answer']).encode()) + 1


def complete(rows, originals):
    result = []
    for record in rows:
        base_id = record.get('source_id', record['id']).split('::name-')[0]
        original = originals[base_id]
        mapping = {original['function']: record['function'], **record.get('argument_alias_map', {})}
        answer = rename_code(original['answer'], mapping)
        if answer.splitlines(keepends=True)[0] != record['answer']:
            raise ValueError('full reference header differs')
        result.append({**record, 'answer': answer})
    return result


def build(dataset, policy):
    original, renamed, development, header_metadata = header_pools(dataset, policy)
    root = Path(dataset)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    sources = {}
    for split in ('train', 'validation', 'test'):
        raw = (root / (split + '.json')).read_bytes()
        if digest(raw) != manifest['splits'][split]['sha256']:
            raise ValueError('full frozen source differs')
        sources[split] = {r['id']: r for r in json.loads(raw)}
    # Training lookup cannot read validation/test references.
    original, renamed = complete(original, sources['train']), complete(renamed, sources['train'])
    development = {key: complete(rows, sources[key.split('/')[0]]) for key, rows in development.items()}
    sizes = {}
    for key, rows in [('original', original), ('renamed', renamed), *development.items()]:
        lengths = [frame_size(r) for r in rows]
        if max(lengths) > 256:
            raise ValueError('full response exceeds context; never silently truncate')
        sizes[key] = {'records': len(rows), 'max_frame_bytes': max(lengths)}
    return original, renamed, development, {
        'version': 1, 'argument_policy': policy, 'header_data': header_metadata,
        'replay': replay_metadata(original, renamed, 'half_original'),
        'development_sha256': digest(encoded(development)), 'frame_sizes': sizes,
        'full_reference_transform': 'Python NAME tokens only; original frozen split references',
        'limits': LIMITS}


def verify_body(source, initial, dataset):
    _, header_provenance = verify_initial(source, initial, dataset)
    origins = verify_source(initial, dataset)
    path = Path(source) / 'body-source/NppAI-broader-instruction.nppai'
    raw = (Path(source) / 'body-source/broader-report.json').read_bytes()
    report = json.loads(raw)
    manifest = json.loads((Path(dataset) / 'manifest.json').read_bytes())
    check_source(report, path.read_bytes(), manifest, 'dataset_manifest', 10000)
    check_continuation(report, origins['bodies']['sha256'], 'four_forms')
    if json.loads((Path(source) / 'staged-report.json').read_bytes())['source_body_report'] != report:
        raise ValueError('cached body provenance differs')
    return path, {'model_sha256': digest(path.read_bytes()), 'report_sha256': digest(raw),
                  'prior_body_updates': 20000, 'source_policy': 'four_forms',
                  'initial_body_sha256': origins['bodies']['sha256'], 'paired_header': header_provenance}


def summarize(cases):
    def counts(rows):
        return {'total': len(rows), **{arm: {metric: sum(c[arm][metric] for c in rows)
                for metric in METRICS} for arm in ARMS}}
    return {'overall': counts(cases), 'name_length': {str(length): counts([c for c in cases if c['name_length'] == length])
            for length in sorted({c['name_length'] for c in cases})}}


def evaluate(model, development, generate):
    groups = {}
    for split, rows in development.items():
        cases = []
        for record in rows:
            header = task_header(record['prompt'])  # Never supply reference to parser.
            if not header_fits(record['prompt'], header, 256):
                raise ValueError('prompt-derived header does not fit')
            case = {**record, 'assisted_header': header}
            case['task_signature_assisted'] = score(record, generate(model, record['prompt'], 256, assistant_prefix=header))
            case['single_model'] = score(record, generate(model, record['prompt'], 256))
            cases.append(case)
        groups[split] = {'cases': cases, 'summary': summarize(cases)}
        print(json.dumps({'split': split, **groups[split]['summary']['overall']}), flush=True)
    return groups


def paired(evaluation):
    result = {}
    for split, group in evaluation['before'].items():
        before = {c['id']: c for c in group['cases']}
        after = {c['id']: c for c in evaluation['after'][split]['cases']}
        if before.keys() != after.keys():
            raise ValueError('unpaired body cases')
        result[split] = {arm: {metric: {label: sum(bool(before[k][arm][metric]) == b and bool(after[k][arm][metric]) == a for k in before)
            for label, b, a in (('both', True, True), ('lost', True, False), ('gained', False, True), ('neither', False, False))}
            for metric in ('requested_signature', 'reference_ast', 'exact_response')} for arm in ARMS}
    return result


def audit(report, dataset):
    _, _, development, metadata = build(dataset, report['argument_policy'])
    source = report['source']
    if (report['dataset'] != metadata or report['source_run_id'] != SOURCE_RUN or
        report['additional_steps'] != STEPS or report['total_body_updates'] != 26000 or
        report['optimizer'] != OPTIMIZER or report['context'] != 256 or report['new_tokens_per_call'] != 160 or
        report['observed_training_draws'] != {'original': 12000, 'renamed': 12000} or
        report['initialization'] != {'checkpoint_sha256': source['model_sha256'], 'method': 'all dense FP32 weights; fresh AdamW optimizer'} or
        source['prior_body_updates'] != 20000 or source['source_policy'] != 'four_forms' or
        report['reload_outputs_identical'] is not True):
        raise ValueError('body budget/data/initialization/reload differs')
    for stage in ('before', 'after'):
        groups = report['evaluation'][stage]
        if groups.keys() != development.keys():
            raise ValueError('body groups differ')
        for split, rows in development.items():
            group = groups[split]
            expected = {r['id']: r for r in rows}
            if len(group['cases']) != len(rows) or len({c['id'] for c in group['cases']}) != len(rows):
                raise ValueError('missing/duplicate body cases')
            for c in group['cases']:
                record = expected.get(c['id'])
                if record is None or any(c.get(k) != v for k, v in record.items()):
                    raise ValueError('body prompt/reference differs')
                header = task_header(record['prompt'])
                if c['assisted_header'] != header or not c['task_signature_assisted']['output'].startswith(header):
                    raise ValueError('body assisted prefix differs')
                for arm in ARMS:
                    if c[arm] != score(record, c[arm]['output']):
                        raise ValueError('raw body metrics differ')
            if group['summary'] != summarize(group['cases']):
                raise ValueError('body summary differs')
    if report['paired_before_after'] != paired(report['evaluation']):
        raise ValueError('paired body metrics differ')
    return {'valid': True, 'development_sha256': metadata['development_sha256'],
            'development_cases_per_stage': sum(map(len, development.values())), 'limits': LIMITS}


def train(original, renamed, development, initial_model, checkpoint):
    import torch
    import torch.nn.functional as F
    from train_nppai import NppAIModel, export_to_bin
    from scripts.train_identifier_headers import initialize_header_weights
    from scripts.train_broader_instruction_experiment import sample
    from scripts.train_instruction_experiment import generate
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    torch.manual_seed(42)
    torch.set_num_threads(2)
    rng = random.Random(42)
    model = NppAIModel(256, 64, 128, 2, 256)
    # Generic dense all-weight initialization, despite this historical helper name.
    initialization = initialize_header_weights(model, initial_model)
    model.eval()
    with torch.no_grad():
        before = evaluate(model, development, generate)
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
        after = evaluate(reloaded, development, generate)
        for split, rows in development.items():
            for record, cached in zip(rows, after[split]['cases']):
                for arm in ARMS:
                    prefix = task_header(record['prompt']) if arm == 'task_signature_assisted' else ''
                    if cached[arm]['output'] != generate(model, record['prompt'], 256, assistant_prefix=prefix):
                        raise RuntimeError('export/reload changed body outputs')
    return {'initialization': initialization, 'observed_training_draws': draws,
            'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'threads': 2},
            'reload_outputs_identical': True, 'evaluation': {'before': before, 'after': after}}


def run(source, initial, dataset, output, policy):
    original, renamed, development, metadata = build(dataset, policy)
    initial_model, provenance = verify_body(source, initial, dataset)
    destination = Path(output)
    checkpoint = destination / 'NppAI-body-argument-aliases.nppai'
    measured = train(original, renamed, development, initial_model, checkpoint)
    report = {**measured, 'source_run_id': SOURCE_RUN, 'source': provenance, 'argument_policy': policy,
              'dataset': metadata, 'additional_steps': STEPS, 'total_body_updates': 26000,
              'optimizer': OPTIMIZER, 'context': 256, 'new_tokens_per_call': 160,
              'model_sha256': digest(checkpoint.read_bytes()), 'limits': LIMITS}
    report['paired_before_after'] = paired(report['evaluation'])
    evidence = audit(report, dataset)
    raw = encoded(report)
    (destination / 'body-argument-report.json').write_bytes(raw)
    evidence['source_report_sha256'] = digest(raw)
    (destination / 'body-argument-audit.json').write_bytes(encoded(evidence))
    for name, records in (('original-training', original), ('renamed-training', renamed), ('development', development), ('manifest', metadata)):
        (destination / (name + '.json')).write_bytes(encoded(records))
    for folder in ('body-source', 'header-source', 'initial-header-source', 'initial-body-source'):
        shutil.copytree(Path(source) / folder, destination / folder, dirs_exist_ok=True)
    shutil.copy2(Path(source) / 'staged-report.json', destination / 'source-continuation-report.json')
    print(json.dumps({'policy': policy, 'source': provenance, 'audit': evidence,
        'summary': {stage: {split: group['summary'] for split, group in groups.items()} for stage, groups in report['evaluation'].items()},
        'paired_before_after': report['paired_before_after']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'initial', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--policy', choices=POLICIES, required=True)
    args = parser.parse_args()
    run(args.source, args.initial, args.dataset, args.output, args.policy)
