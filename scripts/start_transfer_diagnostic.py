"""Classify frozen source-start transfer/termination failures from audited records."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_start_selection as source
from scripts.alias_copy_failure_diagnostic import diagnose, source_starts
from scripts.audit_protected_sequential import retained

p = source.p
ROOT = source.ROOT


def identifier_part(character):
    return ('a'+character).isidentifier()


def source_location(prompt, cursor, name):
    """Lexical description of a recorded absolute frame byte position."""
    raw = prompt.encode()
    position = cursor-len(b'[USER]: ')
    if not 0 <= position < len(raw):
        raise ValueError('recorded source is outside prompt')
    starts = source_starts(prompt, name)
    try:
        prefix = raw[:position].decode('utf-8')
    except UnicodeDecodeError:
        return {'category': 'utf8_continuation', 'token': None, 'token_offset': None}
    index = len(prefix)
    character = prompt[index]
    if identifier_part(character):
        left = index
        while left and identifier_part(prompt[left-1]): left -= 1
        right = index+1
        while right < len(prompt) and identifier_part(prompt[right]): right += 1
        token = prompt[left:right]
        category = ('requested_start' if cursor in starts else
                    'inside_requested_name' if token == name else
                    'other_identifier_start' if index == left else 'inside_other_identifier')
        return {'category': category, 'token': token,
                'token_offset': len(prompt[left:index].encode())}
    return {'category': 'whitespace' if character.isspace() else 'punctuation',
            'token': character, 'token_offset': 0}


def classify(case, row):
    result = diagnose(case, row)
    raw = bytes(case['output_bytes'])
    expected_name = row['function'].encode()
    end = 4+len(expected_name)
    event = case['events'][0] if case['events'] else None
    location = source_location(row['prompt'], event['cursor'], row['function']) if event else {
        'category': 'no_copy_trigger', 'token': None, 'token_offset': None}
    divergence_event = next((e for e in case['events'] if e['response_query'] == result['first_output_divergence']), None)
    override = bool(divergence_event and divergence_event['kind'] == 'copy' and
                    divergence_event['base_byte'] == result['expected_byte'] and
                    divergence_event['chosen'] != result['expected_byte'])
    generated = re.match(r'^def ([^\W\d]\w*)', case['output'])
    exact_name_bytes = raw.startswith(b'def '+expected_name)
    opening_paren = exact_name_bytes and raw[end:end+1] == b'('
    name_status = ('complete_name_and_paren' if opening_paren else
                   'correct_name_then_extra_identifier' if exact_name_bytes and generated and generated.group(1) != row['function'] else
                   'correct_name_missing_paren' if exact_name_bytes else 'wrong_or_missing_name')
    return {'id': row['id'], 'source_location': location, 'selected_start': result['selected_start'],
            'requested_starts': result['source_starts'],
            'first_output_divergence': result['first_output_divergence'],
            'divergence_region': result['divergence_region'],
            'mechanism_observations': result['mechanism_observations'],
            'name_status': name_status, 'actual_identifier': generated.group(1) if generated else None,
            'first_divergence_event': divergence_event,
            'expected_divergence_byte': result['expected_byte'],
            'copy_overrides_correct_vocabulary': override,
            'exact_response': case['metrics']['exact_response'],
            'requested_signature': case['metrics']['requested_signature']}


def counts(rows):
    return {'total': len(rows), 'copy_overrides_correct_vocabulary': sum(r['copy_overrides_correct_vocabulary'] for r in rows),
            'correct_start_total': sum(r['source_location']['category'] == 'requested_start' for r in rows),
            'correct_start_copy_override': sum(r['source_location']['category'] == 'requested_start' and r['copy_overrides_correct_vocabulary'] for r in rows),
            'exact_response': sum(r['exact_response'] for r in rows),
            'requested_signature': sum(r['requested_signature'] for r in rows),
            'source_locations': dict(sorted(Counter(r['source_location']['category'] for r in rows).items())),
            'selected_tokens': dict(sorted(Counter(r['source_location']['token'] for r in rows
                                                   if r['source_location']['token'] is not None).items())),
            'divergence_regions': dict(sorted(Counter(r['divergence_region'] for r in rows).items())),
            'name_statuses': dict(sorted(Counter(r['name_status'] for r in rows).items())),
            'mechanism_observations': dict(sorted(Counter(f for r in rows for f in r['mechanism_observations']).items()))}


def build():
    report = source.audit(ROOT/'docs/evidence/start-selection')
    initial = retained(ROOT/'docs/evidence/protected-sequential')['arms']['frozen_sequential']['after']
    groups, fingerprints = p.prior.inputs(ROOT/'evaluation/instruction-v3', ROOT/'evaluation/gated-copy-probes-v1/tasks.json')
    stages = {'initial': initial, **{a: r['after'] for a, r in report['arms'].items()}}
    result = {'version': 1, 'mode': 'audited_retained_evidence_only', 'promotion': False,
              'inputs': fingerprints,
              'source_sha256': {str(path.relative_to(ROOT)): p.quality.digest(path.read_bytes()) for path in (
                  ROOT/'docs/evidence/start-selection/retention.json',
                  ROOT/'docs/evidence/protected-sequential/manifest.json')},
              'limits': ['Observational labels of retained cursor/output bytes; no new inference or training.',
                         'No full start logits retained, so candidate masking/re-ranking cannot be simulated.',
                         'Body text mismatch is not proof of semantic failure; historical Docker results remain separate.',
                         'UTF-8 and repeated-name unit fixtures are classifier checks, not measured model coverage.'],
              'stages': {}, 'records': {}, 'transitions': {}}
    for stage, outputs in stages.items():
        records = {group: [classify(case, row) for case, row in zip(outputs[group], rows)]
                   for group, rows in groups.items()}
        result['records'][stage] = records
        summary = {group: counts(rows) for group, rows in records.items()}
        alias = [r for k, rows in records.items() if k.startswith(('validation/', 'test/')) and not k.endswith('/original') for r in rows]
        natural = [r for k, rows in records.items() if k.startswith(('validation/', 'test/')) and k.endswith('/original') for r in rows]
        summary['development_aliases'] = counts(alias)
        summary['development_natural'] = counts(natural)
        result['stages'][stage] = summary
    for stage in ('start_ce', 'start_margin'):
        transitions = Counter()
        for group, rows in result['records'][stage].items():
            if not group.startswith(('validation/', 'test/')) or group.endswith('/original'):
                continue
            for before, after in zip(result['records']['initial'][group], rows):
                if before['id'] != after['id']:
                    raise ValueError('source case order differs')
                transitions[before['source_location']['category']+' -> '+after['source_location']['category']] += 1
        result['transitions'][stage] = dict(sorted(transitions.items()))
    return result


def encoded_members(report):
    metadata = {k: v for k, v in report.items() if k != 'records'}
    members = {'report.json': p.quality.encoded(metadata)}
    for stage, groups in report['records'].items():
        for group, rows in groups.items():
            members[stage+'-'+group.replace('/', '-')+'.json'] = p.quality.encoded(rows)
    return members


def save(directory):
    members = encoded_members(build())
    directory.mkdir(parents=True, exist_ok=True)
    for name, raw in members.items():
        (directory/name).write_bytes(raw)
    manifest = {name: p.quality.digest(raw) for name, raw in sorted(members.items())}
    (directory/'manifest.json').write_bytes(p.quality.encoded(manifest))
    return manifest


def audit(directory):
    manifest = json.loads((directory/'manifest.json').read_bytes())
    expected = encoded_members(build())
    if set(manifest) != set(expected) or {f.name for f in directory.glob('*.json')} != set(expected)|{'manifest.json'}:
        raise ValueError('transfer member set differs')
    for name, raw in expected.items():
        observed = (directory/name).read_bytes()
        if p.quality.digest(observed) != manifest[name] or observed != raw:
            raise ValueError('transfer evidence differs: '+name)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--output', type=Path)
    target.add_argument('--directory', type=Path)
    target.add_argument('--audit', type=Path)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps({'valid': audit(args.audit)}))
    elif args.directory:
        print(json.dumps({'files': len(save(args.directory)), 'valid': audit(args.directory)}))
    else:
        report = build()
        args.output.write_bytes(p.quality.encoded(report))
        print(json.dumps({k: v['development_aliases'] for k, v in report['stages'].items()}, sort_keys=True))
