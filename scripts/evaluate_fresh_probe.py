"""Inference-only fresh probe for two preselected continuation pairs; never execute outputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_fresh_probe import build, ROOT
from scripts.evaluate_staged_generation import check_source, header_fits, score_output
from scripts.run_wording_continuation import TASKS, check_continuation, verify_source

SOURCE_RUN = 38618406741
POLICIES = ('four_forms', 'eight_forms')
METRICS = ('nonempty', 'syntax_any_function', 'requested_function', 'requested_signature', 'reference_ast', 'exact_response')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_dataset(folder):
    folder = Path(folder)
    tasks, manifest = build()
    if (folder / 'tasks.json').read_bytes() != tasks or (folder / 'manifest.json').read_bytes() != manifest:
        raise ValueError('fresh frozen dataset bytes differ')
    return json.loads(tasks), json.loads(manifest)


def signature(text):
    try:
        tree = ast.parse(text)
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
            node = tree.body[0]
            return node.name, ast.dump(node.args)
    except (SyntaxError, ValueError):
        pass
    return None


def score(record, output):
    result = score_output(record, output)
    result['requested_signature'] = signature(output) == signature(record['answer'])
    return result


def summarize(cases):
    return {'total': len(cases), 'headers_accepted': sum(c['header_accepted'] for c in cases),
            'headers_exact_reference': sum(c['header_exact_reference'] for c in cases),
            'headers_requested_signature': sum(c['header_requested_signature'] for c in cases),
            **{arm: {metric: sum(c[arm][metric] for c in cases) for metric in METRICS}
               for arm in ('staged', 'single_model')}}


def aggregate(cases):
    groups = {}
    for field in ('wording', 'family', 'name_length'):
        groups[field] = {str(value): summarize([c for c in cases if c[field] == value])
                         for value in sorted({c[field] for c in cases})}
    paired = {}
    for form in sorted({c['wording'] for c in cases} - {'known_format'}):
        originals = {c['task_id']: c for c in cases if c['wording'] == 'known_format'}
        changed = {c['task_id']: c for c in cases if c['wording'] == form}
        if originals.keys() != changed.keys():
            raise ValueError('wording task pairs differ')
        paired[form] = {arm: {metric: {
            label: sum(bool(originals[k][arm][metric]) == before and bool(changed[k][arm][metric]) == after for k in originals)
            for label, before, after in (('both', True, True), ('lost', True, False), ('gained', False, True), ('neither', False, False))}
            for metric in ('requested_signature', 'reference_ast', 'exact_response')}
            for arm in ('staged', 'single_model')}
    return {'overall': summarize(cases), 'groups': groups, 'paired_to_known_format': paired,
            'task_instances': len({c['task_id'] for c in cases})}


def audit(report, dataset):
    records, manifest = load_dataset(dataset)
    if report.get('dataset_manifest') != manifest or report.get('source_run_id') != SOURCE_RUN or report.get('policy') not in POLICIES:
        raise ValueError('report dataset/source/policy differs')
    cases = report['cases']
    if len(cases) != len(records) or len({c['id'] for c in cases}) != len(records):
        raise ValueError('missing or duplicate fresh cases')
    expected = {r['id']: r for r in records}
    for case in cases:
        record = expected.get(case['id'])
        if record is None:
            raise ValueError('unknown fresh case')
        metadata = {key: record[key] for key in ('id', 'task_id', 'family', 'function', 'wording', 'prompt')}
        metadata.update(reference=record['answer'], name_length=len(record['function']))
        if any(case.get(key) != value for key, value in metadata.items()):
            raise ValueError('fresh case metadata differs')
        header = case['generated_header']
        accepted = header_fits(record['prompt'], header, 256)
        if (case['header_accepted'] != accepted or
            case['header_exact_reference'] != (header == record['answer'].splitlines(keepends=True)[0]) or
            case['header_requested_signature'] != (signature(header + '    pass\n') == signature(record['answer']))):
            raise ValueError('header metrics differ')
        if not accepted and case['staged']['output'] != '':
            raise ValueError('rejected header has staged output')
        if accepted and not case['staged']['output'].startswith(header):
            raise ValueError('staged output does not preserve raw header')
        for arm in ('staged', 'single_model'):
            if case[arm] != score(record, case[arm]['output']):
                raise ValueError('raw output metrics differ')
    recalculated = aggregate(cases)
    if report['summary'] != recalculated:
        raise ValueError('aggregate metrics differ')
    return {'valid': True, 'cases': len(cases), 'task_instances': 36,
            'staged_exact_present_in_each_wording': all(g['staged']['exact_response'] > 0 for g in recalculated['groups']['wording'].values()),
            'limits': 'Audit validates saved evidence, not functional correctness, model quality or checkpoint promotion.'}


def verify_pair(source, initial, legacy_dataset, policy):
    source = Path(source)
    origins = verify_source(initial, legacy_dataset)
    copied = verify_source(source, legacy_dataset, 'initial-')
    if any(origins[k]['sha256'] != copied[k]['sha256'] for k in TASKS):
        raise ValueError('preserved initial source differs')
    manifest = json.loads((Path(legacy_dataset) / 'manifest.json').read_bytes())
    cached = json.loads((source / 'staged-report.json').read_bytes())
    if cached.get('source_run_id') != SOURCE_RUN:
        raise ValueError('continuation run differs')
    hashes = {}
    for task, (folder, model_name, report_name, manifest_key) in TASKS.items():
        model = source / folder / model_name
        raw = (source / folder / report_name).read_bytes()
        report = json.loads(raw)
        check_source(report, model.read_bytes(), manifest, manifest_key, 10000)
        check_continuation(report, origins[task]['sha256'], policy)
        if cached['source_header_report' if task == 'headers' else 'source_body_report'] != report:
            raise ValueError('cached continuation report differs')
        hashes[task] = {'checkpoint_sha256': sha(model.read_bytes()), 'report_sha256': sha(raw),
                        'initial_checkpoint_sha256': origins[task]['sha256']}
    return hashes


def run(source, initial, dataset, output, policy):
    records, manifest = load_dataset(dataset)
    hashes = verify_pair(source, initial, ROOT / 'evaluation/instruction-v3', policy)
    import torch
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    from scripts.identifier_curriculum import validate_initial_weights
    from scripts.train_instruction_experiment import generate
    torch.set_num_threads(2)
    models = {}
    for task, (folder, model_name, _, _) in TASKS.items():
        weights = load_weights(Path(source) / folder / model_name)
        validate_initial_weights(weights)
        models[task] = torch_model(weights)
    cases = []
    with torch.no_grad():
        for record in records:
            header = generate(models['headers'], record['prompt'], 256)
            accepted = header_fits(record['prompt'], header, 256)
            staged = generate(models['bodies'], record['prompt'], 256, assistant_prefix=header) if accepted else ''
            single = generate(models['bodies'], record['prompt'], 256)
            case = {key: record[key] for key in ('id', 'task_id', 'family', 'function', 'wording', 'prompt')}
            case.update(reference=record['answer'], name_length=len(record['function']), generated_header=header,
                        header_accepted=accepted, header_exact_reference=header == record['answer'].splitlines(keepends=True)[0],
                        header_requested_signature=signature(header + '    pass\n') == signature(record['answer']),
                        staged=score(record, staged), single_model=score(record, single))
            cases.append(case)
    report = {'source_run_id': SOURCE_RUN, 'policy': policy, 'source_hashes': hashes,
              'dataset_manifest': manifest, 'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'threads': 2},
              'generation': {'context': 256, 'new_token_budget_per_call': 160, 'greedy': True,
                             'header_policy': 'raw learned header; reject invalid/context-overflowing; no fallback or reference correction'},
              'limits': '144 correlated wording views of 36 authored tasks. No training, model-output execution, native replay, tuning or promotion. Signature and AST matching are structural metrics, not semantic correctness. Dataset becomes exposed after this measurement.',
              'cases': cases, 'summary': aggregate(cases)}
    result = audit(report, dataset)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(report, indent=2) + '\n').encode('utf-8')
    (destination / 'fresh-report.json').write_bytes(raw)
    result['source_report_sha256'] = sha(raw)
    (destination / 'fresh-audit.json').write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
    for folder in ('header-source', 'body-source', 'initial-header-source', 'initial-body-source'):
        shutil.copytree(Path(source) / folder, destination / folder, dirs_exist_ok=True)
    shutil.copy2(Path(source) / 'staged-report.json', destination / 'source-continuation-report.json')
    print(json.dumps({'policy': policy, **report['summary'], 'audit': result}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'initial', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--policy', choices=POLICIES, required=True)
    args = parser.parse_args()
    run(args.source, args.initial, args.dataset, args.output, args.policy)
