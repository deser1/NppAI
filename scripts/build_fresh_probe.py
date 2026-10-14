"""Freeze evaluation-only tasks created after the selected model checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import string

ROOT = Path(__file__).resolve().parents[1]
FORMS = {
    "known_format": "Define {signature} in Python to return {description}.",
    "request": "Create {signature}; result: {description}. Use Python.",
    "contract": "{signature} -> {description}. Python code only.",
    "brief": "Code request for {signature}: {description}.",
}
# Project-authored references and trusted example checks, not model outputs.
SPECS = {
    "count_equal": ([('values', 'x'), ('items', 'target'), ('seq', 'needle')], '{0}.count({1})', 'the number of items equal to {1} in list {0}', [(([1, 2, 1], 1), 2), (([], 1), 0)]),
    "membership": ([('values', 'x'), ('items', 'target'), ('seq', 'needle')], '{1} in {0}', 'whether {1} is an element of list {0}', [(([1, 2], 2), True), (([], 2), False)]),
    "filter_nonzero": ([('values',), ('items',), ('numbers',)], '[v for v in {0} if v != 0]', 'nonzero numbers from {0}, keeping their order', [(([0, -2, 0, 3],), [-2, 3]), (([],), [])]),
    "sum_squares": ([('values',), ('items',), ('numbers',)], 'sum(v * v for v in {0})', 'the sum of squared numbers in {0}', [(([-2, 3],), 13), (([],), 0)]),
    "first_or_none": ([('values',), ('items',), ('seq',)], '{0}[0] if {0} else None', 'the first element of {0}, or None when empty', [(([5, 6],), 5), (([],), None)]),
    "lookup_default": ([('values', 'x', 'default'), ('data', 'key', 'fallback'), ('mapping', 'name', 'backup')], '{0}.get({1}, {2})', 'the value for key {1} in mapping {0}, or {2} if missing', [(({'a': 0}, 'a', 7), 0), (({}, 'b', 7), 7)]),
    "invert_mapping": ([('values',), ('data',), ('mapping',)], '{{v: k for k, v in {0}.items()}}', '{0} with keys and unique hashable values swapped', [(({'a': 1, 'b': 2},), {1: 'a', 2: 'b'}), (({},), {})]),
    "starts_prefix": ([('text', 'x'), ('value', 'prefix'), ('s', 'start')], '{0}.startswith({1})', 'whether {0} starts with {1}, case sensitive', [(('Abc', 'A'), True), (('Abc', 'a'), False), (('', ''), True)]),
    "ends_suffix": ([('text', 'x'), ('value', 'suffix'), ('s', 'end')], '{0}.endswith({1})', 'whether {0} ends with {1}, case sensitive', [(('Abc', 'bc'), True), (('Abc', 'BC'), False), (('', ''), True)]),
    "replace_literal": ([('text', 'old', 'new'), ('value', 'before', 'after'), ('s', 'needle', 'replacement')], '{0}.replace({1}, {2})', '{0} with every literal substring {1} replaced by {2}', [(('aba', 'a', 'x'), 'xbx'), (('abc', 'z', 'x'), 'abc')]),
    "split_separator": ([('text', 'sep'), ('value', 'delimiter'), ('s', 'separator')], '{0}.split({1})', 'pieces of {0} split on nonempty {1}, keeping empty pieces', [(('a,,b', ','), ['a', '', 'b']), (('', ','), [''])]),
    "zip_pairs": ([('values', 'other'), ('left', 'right'), ('first', 'second')], 'list(zip({0}, {1}))', 'pairs from {0} and {1}, stopping at the shorter input', [(([1, 2], ['a']), [(1, 'a')]), (([], ['a']), [])]),
}


def legacy_context(root=ROOT):
    names, families, prompts, snapshots = set(), set(), set(), {}
    for version in (1, 2, 3):
        folder = Path(root) / 'evaluation' / ('instruction-v' + str(version))
        manifest = json.loads((folder / 'manifest.json').read_bytes())
        for split, metadata in manifest['splits'].items():
            raw = (folder / (split + '.json')).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != metadata['sha256']:
                raise ValueError('legacy frozen bytes differ')
            snapshots[f'instruction-v{version}/{split}.json'] = digest
            for record in json.loads(raw):
                names.add(record['function'])
                families.add(record['family'])
                prompts.add(record['prompt'])
    return names, families, prompts, snapshots


def build(root=ROOT):
    banned, prior_families, old_prompts, snapshots = legacy_context(root)
    if set(SPECS) & prior_families:
        raise ValueError('fresh task families overlap prior labeled families')
    rng = random.Random(20261014)
    used, records = set(banned), []
    for family, (aliases, expression, description, _) in SPECS.items():
        for index, args in enumerate(aliases):
            while True:
                name = 'fn_' + ''.join(rng.choices(string.ascii_lowercase + string.digits, k=(6, 9, 12)[index]))
                if name not in used:
                    used.add(name)
                    break
            signature = name + '(' + ', '.join(args) + ')'
            answer = 'def ' + signature + ':\n    return ' + expression.format(*args) + '\n'
            task_id = family + '-' + str(index)
            for form, template in FORMS.items():
                prompt = template.format(signature=signature, description=description.format(*args))
                if prompt in old_prompts:
                    raise ValueError('fresh prompt overlaps a prior prompt')
                size = len(('[USER]: ' + prompt + '\n[AI]:\n' + answer).encode('utf-8')) + 1
                if size > 256 or len(answer.encode('utf-8')) > 160:
                    raise ValueError('fresh task exceeds the fixed runtime budget')
                records.append({'id': task_id + '::' + form, 'task_id': task_id,
                    'family': family, 'function': name, 'wording': form,
                    'prompt': prompt, 'answer': answer, 'full_sequence_bytes': size})
    raw = (json.dumps(records, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    manifest = {'version': 1, 'evaluation_only': True, 'seed': 20261014,
                'model_cutoff_training_run': 38618406741,
                'source': 'new project-authored synthetic reference tasks',
                'families': sorted(SPECS), 'task_instances': 36, 'wording_views': len(records),
                'wording_forms': list(FORMS), 'name_lengths': [9, 12, 15],
                'context': 256, 'generation_budget': 160,
                'max_full_sequence_bytes': max(r['full_sequence_bytes'] for r in records),
                'tasks_sha256': hashlib.sha256(raw).hexdigest(), 'legacy_snapshot_sha256': snapshots,
                'limits': '12 authored task families, 36 name/argument instances, 144 correlated wording views; not 144 independent programming problems. New for the pinned prior checkpoints, public and exposed after evaluation. Family labels do not prove every primitive is unseen. No training split or automatic promotion; exact/AST matching alone is not functional correctness.'}
    return raw, (json.dumps(manifest, indent=2) + '\n').encode('utf-8')


def save(output, check=False):
    directory = Path(output)
    tasks, manifest = build()
    files = {'tasks.json': tasks, 'manifest.json': manifest,
             '.gitattributes': b'*.json text eol=lf\n.gitattributes text eol=lf\n'}
    if check:
        for name, expected in files.items():
            if (directory / name).read_bytes() != expected:
                raise ValueError('fresh frozen file differs: ' + name)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        for name, payload in files.items():
            (directory / name).write_bytes(payload)
    print(manifest.decode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'evaluation/fresh-probe-v1'))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    save(args.output, args.check)
