"""Prompt-derived signature assistance versus cached learned headers; no training."""
import argparse
import io
import json
from pathlib import Path
import shutil
import sys
import tokenize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.header_argument_aliases import build, audit as audit_arguments
from scripts.header_name_length import digest, encoded, verify_initial
from scripts.evaluate_staged_generation import check_source, header_fits
from scripts.evaluate_task_signature_wording import task_header
from scripts.diagnose_header_replay import score, METRICS
from scripts.run_wording_continuation import verify_source, check_continuation

HEADER_RUN = 38642891392
BODY_RUN = 38618406741
POLICIES = ('fixed_arguments', 'varied_arguments')
ARMS = (*POLICIES, 'task_signature_assisted')
REPORT_HASHES = {'fixed_arguments': '127bbce6a926af4ee542c2f11c7f7d4852eaa139dbb27f33d85ffdadb3df9913',
                 'varied_arguments': '6511512cf84d3fbf18963f00b04535da0cedf4235474b83637de718397233a20'}


def rename_code(text, mapping):
    # NAME tokens only: do not rename string literals or substrings of names.
    tokens = tokenize.generate_tokens(io.StringIO(text).readline)
    return tokenize.untokenize([t._replace(string=mapping.get(t.string, t.string))
                              if t.type == tokenize.NAME else t for t in tokens])


def full_records(dataset):
    _, _, development, _ = build(dataset, 'fixed_arguments')
    root = Path(dataset)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    originals = {}
    for split in ('validation', 'test'):
        raw = (root / (split + '.json')).read_bytes()
        if digest(raw) != manifest['splits'][split]['sha256']:
            raise ValueError('full frozen reference differs')
        originals.update({r['id']: r for r in json.loads(raw)})
    for rows in development.values():
        for record in rows:
            original = originals[record['id'].split('::name-')[0]]
            mapping = {original['function']: record['function'], **record.get('argument_alias_map', {})}
            answer = rename_code(original['answer'], mapping)
            if answer.splitlines(keepends=True)[0] != record['answer']:
                raise ValueError('full argument reference header differs')
            record['answer'] = answer
    return development


def verify_inputs(roots, body_pair, initial, dataset):
    _, provenance = verify_initial(body_pair, initial, dataset)
    path = Path(body_pair) / 'body-source/NppAI-broader-instruction.nppai'
    raw_body = (Path(body_pair) / 'body-source/broader-report.json').read_bytes()
    body = json.loads(raw_body)
    manifest = json.loads((Path(dataset) / 'manifest.json').read_bytes())
    origins = verify_source(initial, dataset)
    check_source(body, path.read_bytes(), manifest, 'dataset_manifest', 10000)
    check_continuation(body, origins['bodies']['sha256'], 'four_forms')
    cached_body = json.loads((Path(body_pair) / 'staged-report.json').read_bytes())
    if cached_body['source_body_report'] != body:
        raise ValueError('common body report differs')
    reports = {}
    for policy in POLICIES:
        root = Path(roots[policy])
        raw = (root / 'argument-report.json').read_bytes()
        if digest(raw) != REPORT_HASHES[policy]:
            raise ValueError('pinned argument report differs')
        report = json.loads(raw)
        audit_arguments(report, dataset)
        checkpoint = root / 'NppAI-argument-alias-header.nppai'
        if (report['source'] != provenance or report['argument_policy'] != policy or
            report['total_header_updates'] != 26000 or report['reload_outputs_identical'] is not True or
            report['initialization'] != {'checkpoint_sha256': provenance['model_sha256'], 'method': 'all dense FP32 weights; fresh AdamW optimizer'} or
            digest(checkpoint.read_bytes()) != report['model_sha256']):
            raise ValueError('cached learned-header source differs')
        reports[policy] = report
    return reports, path, {'argument_report_sha256': REPORT_HASHES, 'body_sha256': digest(path.read_bytes()),
                           'body_report_sha256': digest(raw_body), 'header_initial': provenance}


def headers_for(reports, split):
    return {policy: {c['id']: c['output'] for c in r['evaluation']['after'][split]['cases']}
            for policy, r in reports.items()}


def summarize(cases):
    return {'total': len(cases), **{arm: {
        **{metric: sum(c[arm]['score'][metric] for c in cases) for metric in METRICS},
        'headers_accepted': sum(c[arm]['header_accepted'] for c in cases),
        'headers_exact_reference': sum(c[arm]['header_exact_reference'] for c in cases)} for arm in ARMS},
        'single_model': {metric: sum(c['single_model'][metric] for c in cases) for metric in METRICS}}


def summaries(groups):
    return {split: {'overall': summarize(cases), 'name_length': {
        str(length): summarize([c for c in cases if c['name_length'] == length])
        for length in sorted({c['name_length'] for c in cases})}} for split, cases in groups.items()}


def paired(groups):
    return {split: {policy: {metric: {label: sum(bool(c[policy]['score'][metric]) == b and bool(c['task_signature_assisted']['score'][metric]) == a for c in cases)
        for label, b, a in (('both', True, True), ('lost', True, False), ('gained', False, True), ('neither', False, False))}
        for metric in ('requested_signature', 'reference_ast', 'exact_response')} for policy in POLICIES}
        for split, cases in groups.items()}


def audit(report, dataset, sources):
    expected = full_records(dataset)
    if report['header_source_run'] != HEADER_RUN or report['body_source_run'] != BODY_RUN or report['groups'].keys() != expected.keys():
        raise ValueError('assisted source/groups differ')
    for split, rows in expected.items():
        cases = report['groups'][split]
        by_id = {r['id']: r for r in rows}
        if len(cases) != len(rows) or len({c['id'] for c in cases}) != len(rows):
            raise ValueError('missing/duplicate assisted cases')
        caches = headers_for(sources, split)
        for case in cases:
            record = by_id.get(case['id'])
            if record is None or any(case.get(k) != v for k, v in record.items()):
                raise ValueError('assisted reference/prompt differs')
            for arm in ARMS:
                item = case[arm]
                expected_header = task_header(record['prompt']) if arm == 'task_signature_assisted' else caches[arm][record['id']]
                accepted = header_fits(record['prompt'], expected_header, 256)
                if (item['header'] != expected_header or item['header_accepted'] != accepted or
                    item['header_exact_reference'] != (expected_header == record['answer'].splitlines(keepends=True)[0])):
                    raise ValueError('assisted/cached header differs')
                output = item['score']['output']
                if (not accepted and output != '') or (accepted and not output.startswith(expected_header)):
                    raise ValueError('prefix composition differs')
                if item['score'] != score(record, output):
                    raise ValueError('assisted raw output metrics differ')
            if case['single_model'] != score(record, case['single_model']['output']):
                raise ValueError('free-body output metrics differ')
    if report['summary'] != summaries(report['groups']) or report['assisted_paired_to_learned'] != paired(report['groups']):
        raise ValueError('assisted summaries differ')
    return {'valid': True, 'cases': sum(len(rows) for rows in expected.values()),
            'limits': 'Assisted signatures come from the prompt parser, not model copying. Raw-reference metrics are not execution or functional correctness.'}


def run(roots, body_pair, initial, dataset, output):
    sources, path, provenance = verify_inputs(roots, body_pair, initial, dataset)
    records = full_records(dataset)
    import torch
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    from scripts.identifier_curriculum import validate_initial_weights
    from scripts.train_instruction_experiment import generate
    torch.set_num_threads(2)
    weights = load_weights(path)
    validate_initial_weights(weights)
    model = torch_model(weights)
    groups = {}
    with torch.no_grad():
        for split, rows in records.items():
            cache = headers_for(sources, split)
            cases = []
            for record in rows:
                # Only the prompt is supplied to the deterministic parser.
                headers = {'task_signature_assisted': task_header(record['prompt']),
                           **{policy: cache[policy][record['id']] for policy in POLICIES}}
                case = dict(record)
                for arm in ARMS:
                    header = headers[arm]
                    accepted = header_fits(record['prompt'], header, 256)
                    text = generate(model, record['prompt'], 256, assistant_prefix=header) if accepted else ''
                    case[arm] = {'header': header, 'header_accepted': accepted,
                        'header_exact_reference': header == record['answer'].splitlines(keepends=True)[0], 'score': score(record, text)}
                case['single_model'] = score(record, generate(model, record['prompt'], 256))
                cases.append(case)
            groups[split] = cases
            print(json.dumps({'split': split, **summarize(cases)}), flush=True)
    report = {'header_source_run': HEADER_RUN, 'body_source_run': BODY_RUN, 'source_hashes': provenance,
              'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'threads': 2},
              'generation': {'context': 256, 'new_tokens_per_body_call': 160, 'greedy': True},
              'limits': 'Inference-only assisted diagnostic on1100 correlated exposed development views. Cached learned headers are composed verbatim; assisted names/arguments are supplied by a restricted prompt-only parser, not learned. Header inference costs differ. No reference repair, training, model selection, generated-source execution, fresh-probe use, native conversion or promotion. Exact/AST matches are structural reference metrics, not semantic correctness.',
              'groups': groups, 'summary': summaries(groups), 'assisted_paired_to_learned': paired(groups)}
    evidence = audit(report, dataset, sources)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    raw = encoded(report)
    (destination / 'assisted-report.json').write_bytes(raw)
    evidence['source_report_sha256'] = digest(raw)
    (destination / 'assisted-audit.json').write_bytes(encoded(evidence))
    for name, source in {**roots, 'body-pair': body_pair}.items():
        shutil.copytree(source, destination / name, dirs_exist_ok=True)
    print(json.dumps({'audit': evidence, 'summary': report['summary'], 'assisted_paired_to_learned': report['assisted_paired_to_learned']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fixed-arguments', 'varied-arguments', 'body-pair', 'initial', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run({'fixed_arguments': args.fixed_arguments, 'varied_arguments': args.varied_arguments}, args.body_pair, args.initial, args.dataset, args.output)
