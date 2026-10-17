"""Read-only diagnosis of audited PR #198 evidence; never execute generated code."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import protected_sequential as pilot
from scripts.audit_protected_sequential import retained


def source_starts(prompt, name):
    """All identifier-boundary matches, as UTF-8 frame byte offsets.

    Labels use the requested name; these positions are never decoding inputs.
    Multiple matches remain ambiguous rather than picking a convenient match.
    """
    starts = []
    for i in range(len(prompt)):
        if not prompt.startswith(name, i):
            continue
        end = i + len(name)
        if i and ('a' + prompt[i-1]).isidentifier():
            continue
        if end < len(prompt) and ('a' + prompt[end]).isidentifier():
            continue
        starts.append(len(('[USER]: ' + prompt[:i]).encode()))
    return starts


def diagnose(case, row):
    pilot.audit_events(case, row)
    actual, expected = bytes(case['output_bytes']), row['answer'].encode()
    mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
    if mismatch is None and len(actual) != len(expected):
        mismatch = min(len(actual), len(expected))
    end = 4 + len(row['function'].encode())
    if mismatch is None:
        category = 'exact_response'
    elif mismatch < 4:
        category = 'declaration_prefix'
    elif mismatch < end:
        category = 'identifier_byte'
    elif mismatch <= expected.index(b'\n'):
        category = 'signature'
    else:
        category = 'body_text'  # Text mismatch alone does not establish semantic failure.
    starts = source_starts(row['prompt'], row['function'])
    events = case['events']
    first = events[0] if events else None
    aligned = first is not None and first['cursor'] in starts
    selected = first['cursor'] if first else None
    flags = []
    if first and starts and not aligned:
        flags.append('wrong_source_start')
    if not starts:
        flags.append('requested_source_span_missing')
    if len(starts) > 1:
        flags.append('ambiguous_source_span')
    if actual.startswith(b'def ') and not first:
        flags.append('missing_copy_events')
    if aligned:
        boundary = selected + len(row['function'].encode())
        if any(e['kind'] in ('exit', 'boundary') and e['cursor'] < boundary for e in events):
            flags.append('premature_exit')
        if any(e['kind'] == 'copy' and e['cursor'] >= boundary for e in events):
            flags.append('late_exit')
    return {'id': row['id'], 'prompt': row['prompt'], 'requested_name': row['function'],
            'source_starts': starts, 'selected_start': selected,
            'first_output_divergence': mismatch, 'divergence_region': category,
            'expected_byte': expected[mismatch] if mismatch is not None and mismatch < len(expected) else None,
            'actual_byte': actual[mismatch] if mismatch is not None and mismatch < len(actual) else None,
            'mechanism_observations': flags, 'output': case['output'],
            'output_bytes': case['output_bytes'], 'events': events}


def functional_record(case):
    """Static parse/lookup plus verbatim historical Docker diagnostics, no exec."""
    try:
        tree = ast.parse(case['source'])
        names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        parse = {'valid': True, 'top_level_functions': names}
    except SyntaxError as exc:
        parse = {'valid': False, 'message': exc.msg, 'line': exc.lineno, 'offset': exc.offset}
    return {**case, 'static_parse': parse,
            'execution': 'historical Docker result only; not rerun',
            'lookup_limit': 'Top-level declarations are static evidence, not a runtime callable lookup.'}


def build(directory):
    report = retained(directory)
    dataset = pilot.ROOT/'evaluation/instruction-v3'
    tasks = pilot.ROOT/'evaluation/gated-copy-probes-v1/tasks.json'
    pilot.audit(report, dataset, tasks)
    groups, _ = pilot.prior.inputs(dataset, tasks)
    arm = report['arms']['frozen_sequential']
    records, summaries = {}, {}
    for group, cases in arm['after'].items():
        rows = [diagnose(case, row) for case, row in zip(cases, groups[group])]
        records[group] = rows
        summaries[group] = {'total': len(rows),
            'divergence_regions': dict(sorted(Counter(r['divergence_region'] for r in rows).items())),
            'mechanism_observations': dict(sorted(Counter(f for r in rows for f in r['mechanism_observations']).items()))}
    return {'version': 1, 'mode': 'retained_evidence_only', 'promotion': False,
            'source_manifest_sha256': pilot.quality.digest((directory/'manifest.json').read_bytes()),
            'checkpoint_sha256': arm['identity']['checkpoint_sha256'],
            'limits': ['No new inference, training, Docker execution or oracle counterfactuals.',
                       'Mechanism observations overlap and are not experimentally established causes.',
                       'Only retained events are available; start probabilities and full vocabulary readouts were not retained.',
                       'Cursor discontinuities fail the evidence audit rather than becoming model-error counts.'],
            'summary': summaries, 'records': records,
            'functional': [functional_record(c) for c in arm['functional']['models']['copy']['cases']]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=pilot.ROOT/'docs/evidence/protected-sequential')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.directory)
    args.output.write_bytes(pilot.quality.encoded(result))
    print(json.dumps(result['summary'], sort_keys=True))
