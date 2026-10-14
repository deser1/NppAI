"""Original replay retained; vary argument aliases in the renamed training half only."""
import argparse
import ast
import builtins
import json
import keyword
from pathlib import Path
import random
import re
import shutil
import string
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.header_original_replay import pools as replay_pools, metadata as replay_metadata, retention
from scripts.header_name_length import (digest, encoded, verify_initial, score, summarize, paired, SOURCE_RUN, STEPS)

POLICIES = ('fixed_arguments', 'varied_arguments')


def argument_names(header):
    node = ast.parse(header + '    pass\n').body[0]
    args = node.args
    if args.posonlyargs or args.defaults or args.vararg or args.kwonlyargs or args.kwarg or any(a.annotation for a in args.args):
        raise ValueError('requires plain positional arguments')
    return [a.arg for a in args.args]


def alias_records(records, namespace, forbidden):
    used = set(forbidden) | {n for r in records for n in argument_names(r["answer"])} | set(dir(builtins)) | set(keyword.kwlist)
    maps, result = {}, []
    for record in records:
        names = argument_names(record['answer'])
        function = record['function']
        if function not in maps:
            rng = random.Random(digest(('argument-alias-v1/' + namespace + '/' + function).encode()))
            mapping = {}
            for index, old in enumerate(names):
                for _ in range(10000):
                    new = ''.join(rng.choices(string.ascii_lowercase, k=(4, 6)[index % 2]))
                    if new not in used:
                        used.add(new)
                        mapping[old] = new
                        break
                else:
                    raise ValueError('argument alias namespace exhausted')
            maps[function] = mapping
        mapping = maps[function]
        if set(mapping) != set(names):
            raise ValueError('source argument names vary within function')
        pattern = r'\b(?:' + '|'.join(re.escape(n) for n in names) + r')\b'
        replace = lambda text: re.sub(pattern, lambda m: mapping[m[0]], text) if names else text
        updated = {**record, 'id': record['id'] + '::argument-alias', 'prompt': replace(record['prompt']),
                   'answer': replace(record['answer']), 'argument_alias_map': mapping}
        if argument_names(updated['answer']) != [mapping[n] for n in names]:
            raise ValueError('argument transformation differs')
        result.append(updated)
    return result, {alias for mapping in maps.values() for alias in mapping.values()}


def build(dataset, policy):
    if policy not in POLICIES:
        raise ValueError('unknown argument policy')
    original, renamed, old_development, base = replay_pools(dataset)
    banned = {name for r in original + renamed for name in argument_names(r['answer'])}
    aliased, training_aliases = alias_records(renamed, 'train', banned)
    development = dict(old_development)
    used = banned | training_aliases
    development_aliases = set()
    for split in ('validation', 'test'):
        rows, names = alias_records(old_development[split + '/renamed'], split, used)
        used |= names
        development_aliases |= names
        development[split + '/argument_aliases'] = rows
    if training_aliases & development_aliases:
        raise ValueError('training/development argument aliases overlap')
    chosen = renamed if policy == 'fixed_arguments' else aliased
    for rows in [original, chosen, *development.values()]:
        for r in rows:
            size = len(('[USER]: ' + r['prompt'] + '\n[AI]:\n' + r['answer']).encode()) + 1
            if size > 256:
                raise ValueError('full alias header reference exceeds context')
    replay = replay_metadata(original, chosen, 'half_original')
    metadata = {'version': 1, 'argument_policy': policy, 'base_name_dataset': base, 'replay': replay,
                'training_argument_aliases': len(training_aliases) if policy == 'varied_arguments' else 0,
                'development_argument_aliases': len(development_aliases),
                'development_sha256': digest(encoded(development)),
                'limits': 'Header-only development. Original train replay remains unchanged. New4/6-letter argument aliases are disjoint between train/development and exclude keywords/builtins. Both policies share development records. Correlated exposed source families, changed byte lengths; no fresh-probe access or body correctness claim.'}
    return original, chosen, development, metadata


def audit(report, dataset):
    original, renamed, development, expected = build(dataset, report['argument_policy'])
    if report['dataset'] != expected or report['source_run_id'] != SOURCE_RUN or report['additional_steps'] != STEPS:
        raise ValueError('argument experiment metadata differs')
    if report['observed_training_draws'] != {'original': 12000, 'renamed': 12000}:
        raise ValueError('replay draw budget differs')
    for stage in ('before', 'after'):
        groups = report['evaluation'][stage]
        if groups.keys() != development.keys():
            raise ValueError('argument development groups differ')
        for split, rows in development.items():
            group = groups[split]
            expected_rows = {r['id']: r for r in rows}
            if len(group['cases']) != len(rows) or len({c['id'] for c in group['cases']}) != len(rows):
                raise ValueError('missing or duplicate argument cases')
            for case in group['cases']:
                record = expected_rows.get(case['id'])
                if record is None or any(case.get(k) != value for k, value in record.items()):
                    raise ValueError('argument prompt/reference differs')
                if case['metrics'] != score(record, case['output']):
                    raise ValueError('argument output metrics differ')
            if group['summary'] != summarize(group['cases']):
                raise ValueError('argument summary differs')
    if report['paired_before_after'] != paired(report['evaluation']) or report['retention'] != retention(report['evaluation']):
        raise ValueError('argument paired/retention summary differs')
    return {'valid': True, 'development_sha256': expected['development_sha256'],
            'renamed_pool_sha256': expected['replay']['renamed_pool_sha256'], 'retention': report['retention'],
            'limits': 'Audited header copying/retention evidence; not full-code correctness or promotion.'}


def run(source, initial, dataset, output, policy):
    original, renamed, development, metadata = build(dataset, policy)
    initial_model, provenance = verify_initial(source, initial, dataset)
    from scripts.header_replay_training import train
    destination = Path(output)
    checkpoint = destination / 'NppAI-argument-alias-header.nppai'
    measured = train(original, renamed, development, initial_model, checkpoint)
    report = {**measured, 'source_run_id': SOURCE_RUN, 'source': provenance, 'argument_policy': policy,
              'dataset': metadata, 'additional_steps': STEPS, 'total_header_updates': 26000,
              'optimizer': {'name': 'AdamW', 'reset': True, 'learning_rate': 0.001, 'batch_size': 4, 'clip_norm': 1.0, 'seed': 42},
              'context': 256, 'new_tokens_per_call': 160, 'model_sha256': digest(checkpoint.read_bytes()),
              'limits': 'Same20k initialization and6000 updates per policy with fixed50% original replay. Header-only; no fresh-probe use, model-output execution, body training, native conversion, ratio tuning or checkpoint promotion. One seed; fixed data/seed do not guarantee identical retraining. Within-run reload identity checked separately.'}
    report['paired_before_after'] = paired(report['evaluation'])
    report['retention'] = retention(report['evaluation'])
    evidence = audit(report, dataset)
    raw = encoded(report)
    (destination / 'argument-report.json').write_bytes(raw)
    evidence['source_report_sha256'] = digest(raw)
    (destination / 'argument-audit.json').write_bytes(encoded(evidence))
    for name, records in (('original-training', original), ('renamed-training', renamed), ('development', development)):
        (destination / (name + '.json')).write_bytes(encoded(records))
    (destination / 'manifest.json').write_bytes(encoded(metadata))
    for folder in ('header-source', 'initial-header-source', 'initial-body-source'):
        shutil.copytree(Path(source) / folder, destination / folder, dirs_exist_ok=True)
    shutil.copy2(Path(source) / 'staged-report.json', destination / 'source-continuation-report.json')
    print(json.dumps({'policy': policy, 'audit': evidence, 'retention': report['retention'],
        'summary': {stage: {split: group['summary'] for split, group in groups.items()} for stage, groups in report['evaluation'].items()}}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'initial', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--policy', choices=POLICIES, required=True)
    args = parser.parse_args()
    run(args.source, args.initial, args.dataset, args.output, args.policy)
