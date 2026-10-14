"""Frozen replay-header diagnosis and complete-body inference; no training or execution."""
import argparse
import ast
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.header_name_length import build, digest, encoded, verify_initial
from scripts.header_original_replay import audit as audit_replay
from scripts.evaluate_staged_generation import check_source, header_fits, score_output
from scripts.run_wording_continuation import verify_source, check_continuation

REPLAY_RUN = 38635038956
BODY_RUN = 38618406741
REPORT_HASHES = {'half_original': '20d3826c2229529f893611b024d2282b9cf6167dffc96a9b4cbfb908e414d61b',
                 'no_replay': '459be098cef131b1e7586c8cd71a5af21e833e6b1eac2335cc77c16eb30d011d'}
ARMS = ('pretrained', 'no_replay', 'half_original')
METRICS = ('nonempty', 'syntax_any_function', 'requested_function', 'requested_signature', 'reference_ast', 'exact_response')


def arguments(header):
    try:
        node = ast.parse(header + '    pass\n').body[0]
        if isinstance(node, ast.FunctionDef):
            return [node.name, ast.unparse(node.args)]
    except (SyntaxError, ValueError, IndexError):
        pass
    return None


def header_error(expected, actual):
    e, a = arguments(expected), arguments(actual)
    if a is None:
        category = 'malformed'
    elif a[0] != e[0]:
        category = 'name'
    elif a[1] != e[1]:
        category = 'arguments'
    elif actual != expected:
        category = 'format'
    else:
        category = 'none'
    return {'category': category, 'expected_signature': e, 'actual_signature': a}


def lost_cases(report):
    result = []
    for split in ('validation/original', 'test/original'):
        before = {c['id']: c for c in report['evaluation']['before'][split]['cases']}
        for after in report['evaluation']['after'][split]['cases']:
            prior = before[after['id']]
            if prior['metrics']['exact_header'] and not after['metrics']['exact_header']:
                result.append({'split': split, 'id': after['id'], 'prompt': after['prompt'],
                    'reference_header': after['answer'], 'before': prior['output'], 'after': after['output'],
                    'difference': header_error(after['answer'], after['output'])})
    return result


def full_records(dataset):
    _, development, _ = build(dataset, 'mixed_lengths')
    originals = {}
    root = Path(dataset)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    for split in ('validation', 'test'):
        raw = (root / (split + '.json')).read_bytes()
        if digest(raw) != manifest['splits'][split]['sha256']:
            raise ValueError('full reference split hash differs')
        originals.update({r['id']: r for r in json.loads(raw)})
    for records in development.values():
        for record in records:
            source_id = record['id'].split('::name-')[0]
            source = originals[source_id]
            answer = re.sub(r'\b' + re.escape(source['function']) + r'\b', record['function'], source['answer'])
            if answer.splitlines(keepends=True)[0] != record['answer']:
                raise ValueError('full reference header differs')
            record['answer'] = answer
    return development


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
    return {**score_output(record, output), 'requested_signature': signature(output) == signature(record['answer'])}


def verify_inputs(roots, body_pair, initial, dataset):
    pretrained, provenance = verify_initial(body_pair, initial, dataset)
    body_path = Path(body_pair) / 'body-source/NppAI-broader-instruction.nppai'
    body_raw = (Path(body_pair) / 'body-source/broader-report.json').read_bytes()
    body_report = json.loads(body_raw)
    manifest = json.loads((Path(dataset) / 'manifest.json').read_bytes())
    initial_sources = verify_source(initial, dataset)
    check_source(body_report, body_path.read_bytes(), manifest, 'dataset_manifest', 10000)
    check_continuation(body_report, initial_sources['bodies']['sha256'], 'four_forms')
    cached = json.loads((Path(body_pair) / 'staged-report.json').read_bytes())
    if cached['source_body_report'] != body_report:
        raise ValueError('cached body source report differs')
    reports, models = {}, {'pretrained': pretrained}
    for policy, root in roots.items():
        root = Path(root)
        raw = (root / 'replay-report.json').read_bytes()
        if digest(raw) != REPORT_HASHES[policy]:
            raise ValueError('pinned replay report differs')
        report = json.loads(raw)
        audit_replay(report, dataset)
        if (report['source'] != provenance or report['replay']['policy'] != policy or
            report['total_header_updates'] != 26000 or report['reload_outputs_identical'] is not True or
            report['initialization'] != {'checkpoint_sha256': provenance['model_sha256'], 'method': 'all dense FP32 weights; fresh AdamW optimizer'}):
            raise ValueError('replay checkpoint provenance differs')
        model = root / 'NppAI-original-replay-header.nppai'
        if digest(model.read_bytes()) != report['model_sha256']:
            raise ValueError('replay checkpoint hash differs')
        reports[policy], models[policy] = report, model
    if reports['no_replay']['evaluation']['before'] != reports['half_original']['evaluation']['before']:
        raise ValueError('cached baseline outputs differ')
    return reports, models, body_path, {'header_initial': provenance, 'body_checkpoint_sha256': digest(body_path.read_bytes()),
        'body_report_sha256': digest(body_raw), 'replay_report_sha256': REPORT_HASHES,
        'header_checkpoint_sha256': {arm: digest(path.read_bytes()) for arm, path in models.items()}}


def summarize(cases):
    return {'total': len(cases), **{arm: {
        **{metric: sum(c[arm]['score'][metric] for c in cases) for metric in METRICS},
        'headers_accepted': sum(c[arm]['header_accepted'] for c in cases),
        'headers_exact_reference': sum(c[arm]['header_exact_reference'] for c in cases),
        'cached_header_mismatches': sum(not c[arm]['matches_cached_header'] for c in cases)} for arm in ARMS},
        'single_model': {metric: sum(c['single_model'][metric] for c in cases) for metric in METRICS}}


def summaries(groups):
    return {split: {'overall': summarize(cases), 'name_length': {
        str(length): summarize([c for c in cases if c['name_length'] == length])
        for length in sorted({c['name_length'] for c in cases})}} for split, cases in groups.items()}


def paired_pipeline(groups):
    result = {}
    for split, cases in groups.items():
        result[split] = {arm: {metric: {label: sum(bool(c['pretrained']['score'][metric]) == b and bool(c[arm]['score'][metric]) == a for c in cases)
            for label, b, a in (('both', True, True), ('lost', True, False), ('gained', False, True), ('neither', False, False))}
            for metric in ('requested_signature', 'reference_ast', 'exact_response')} for arm in ('no_replay', 'half_original')}
    return result


def cached_headers(reports, split, arm):
    report = reports['half_original' if arm == 'pretrained' else arm]
    stage = 'before' if arm == 'pretrained' else 'after'
    return {c['id']: c['output'] for c in report['evaluation'][stage][split]['cases']}


def audit(report, dataset, reports):
    if report['replay_source_run_id'] != REPLAY_RUN or report['body_source_run_id'] != BODY_RUN:
        raise ValueError('source run differs')
    records = full_records(dataset)
    groups = report['groups']
    if groups.keys() != records.keys():
        raise ValueError('complete-body groups differ')
    for split, rows in records.items():
        cases = groups[split]
        expected = {r['id']: r for r in rows}
        if len(cases) != len(rows) or len({c['id'] for c in cases}) != len(rows):
            raise ValueError('missing or duplicate complete-body cases')
        caches = {arm: cached_headers(reports, split, arm) for arm in ARMS}
        for case in cases:
            record = expected.get(case['id'])
            if record is None or any(case.get(k) != v for k, v in record.items()):
                raise ValueError('complete-body prompt/reference differs')
            if case['single_model'] != score(record, case['single_model']['output']):
                raise ValueError('single-model metrics differ')
            for arm in ARMS:
                item = case[arm]
                header = item['header']
                accepted = header_fits(record['prompt'], header, 256)
                if (item['header_accepted'] != accepted or
                    item['header_exact_reference'] != (header == record['answer'].splitlines(keepends=True)[0]) or
                    item['matches_cached_header'] != (header == caches[arm][record['id']])):
                    raise ValueError('pipeline header metrics differ')
                text = item['score']['output']
                if (not accepted and text != '') or (accepted and not text.startswith(header)):
                    raise ValueError('raw-header composition differs')
                if item['score'] != score(record, text):
                    raise ValueError('pipeline output metrics differ')
    if report['summary'] != summaries(groups) or report['paired_to_pretrained'] != paired_pipeline(groups):
        raise ValueError('pipeline summaries differ')
    if report['archived_header_regressions'] != {policy: lost_cases(r) for policy, r in reports.items()}:
        raise ValueError('archived regression diagnosis differs')
    return {'valid': True, 'cases': sum(len(rows) for rows in records.values()),
            'limits': 'Audited raw structural/reference metrics; not execution, functional correctness or promotion.'}


def run(roots, body_pair, initial, dataset, output):
    reports, model_paths, body_path, provenance = verify_inputs(roots, body_pair, initial, dataset)
    records = full_records(dataset)
    import torch
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    from scripts.identifier_curriculum import validate_initial_weights
    from scripts.train_instruction_experiment import generate
    torch.set_num_threads(2)
    models = {}
    for arm, path in {**model_paths, 'body': body_path}.items():
        weights = load_weights(path)
        validate_initial_weights(weights)
        models[arm] = torch_model(weights)
    groups = {}
    with torch.no_grad():
        for split, rows in records.items():
            caches = {arm: cached_headers(reports, split, arm) for arm in ARMS}
            cases = []
            for record in rows:
                case = dict(record)
                case['single_model'] = score(record, generate(models['body'], record['prompt'], 256))
                for arm in ARMS:
                    header = generate(models[arm], record['prompt'], 256)
                    accepted = header_fits(record['prompt'], header, 256)
                    text = generate(models['body'], record['prompt'], 256, assistant_prefix=header) if accepted else ''
                    case[arm] = {'header': header, 'header_accepted': accepted,
                        'header_exact_reference': header == record['answer'].splitlines(keepends=True)[0],
                        'matches_cached_header': header == caches[arm][record['id']], 'score': score(record, text)}
                cases.append(case)
            groups[split] = cases
            print(json.dumps({'split': split, **summarize(cases)}), flush=True)
    report = {'replay_source_run_id': REPLAY_RUN, 'body_source_run_id': BODY_RUN, 'source_hashes': provenance,
              'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'threads': 2},
              'generation': {'context': 256, 'new_tokens_per_call': 160, 'greedy': True, 'raw_header_no_correction': True},
              'limits': 'Fixed FP32 inference only. Header20k/26k and frozen four-form body20k checkpoints; no body training, fresh-probe access, generated-source execution, native replay, model selection or promotion. Exposed synthetic development tasks with528 correlated renamed views and44 original prompts; same body/control. AST/exact matching is not functional correctness. Cached inference deviations are explicit, never corrected.',
              'archived_header_regressions': {policy: lost_cases(r) for policy, r in reports.items()},
              'groups': groups, 'summary': summaries(groups), 'paired_to_pretrained': paired_pipeline(groups)}
    result = audit(report, dataset, reports)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    raw = encoded(report)
    (destination / 'pipeline-report.json').write_bytes(raw)
    result['source_report_sha256'] = digest(raw)
    (destination / 'pipeline-audit.json').write_bytes(encoded(result))
    for name, source in {**roots, 'body-pair': body_pair}.items():
        shutil.copytree(source, destination / name, dirs_exist_ok=True)
    print(json.dumps({'audit': result, 'archived_header_regressions': report['archived_header_regressions'],
                      'summary': report['summary'], 'paired_to_pretrained': report['paired_to_pretrained']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('no-replay', 'half-original', 'body-pair', 'initial', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run({'no_replay': args.no_replay, 'half_original': args.half_original}, args.body_pair, args.initial, args.dataset, args.output)
