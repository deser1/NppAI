"""Post-hoc exposed-data name/argument/wording diagnosis; no new training.

The model/decoder are unchanged. References are controls, never inference input.
This grid explains a failed development pilot and is not an independent test.
"""
import argparse
import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_backends import DenseLocalBackend, GenerationRequest
from scripts.function_body_training import DATA, SOURCE, audit as audit_pilot, load_bank, save, score, sha, summary
from scripts.evaluate_released_model import weights_sha


def probes():
    _, bank, development = load_bank()
    rows = []
    for family in sorted({r['family'] for r in bank}):
        for language in ('en', 'pl'):
            suffix = '-0-0-0' if language == 'en' else '-0-0-2'
            known = next(r for r in bank if r['family'] == family and r['id'].endswith(suffix))
            novel = next(r for r in development if r['family'] == family and r['language'] == language and '-0-' in r['id'])
            for name_bit, arg_bit, word_bit in itertools.product((0, 1), repeat=3):
                name = (known, novel)[name_bit]['function']
                args = (known, novel)[arg_bit]['arguments']
                description = (known, novel)[word_bit]['request']['prompt'].split('. ', 1)[1]
                prompt = ('Implement ' if language == 'en' else 'Napisz funkcję ')+name+'('+', '.join(args)+'). '+description
                reference = known['answer']
                mapping = dict(zip(known['arguments'], args))
                if name != known['function'] or args != known['arguments']:
                    tree = ast.parse(reference)
                    tree.body[0].name = name
                    for node in tree.body[0].args.args: node.arg = mapping.get(node.arg, node.arg)
                    for node in ast.walk(tree.body[0]):
                        if isinstance(node, ast.Name): node.id = mapping.get(node.id, node.id)
                    # This transforms authored control source only, never a model response.
                    reference = ast.unparse(tree)+'\n'
                cases = copy.deepcopy(known['cases'])
                for case in cases: case['function'] = name
                cell = f'name{name_bit}-args{arg_bit}-wording{word_bit}'
                rows.append(dict(id=f'{family}-{language}-{cell}', cell=cell, family=family, language=language,
                                 function=name, arguments=args, request=dict(prompt=prompt, max_new_tokens=256),
                                 answer=reference, cases=cases))
    return rows


def probe_sha(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def audit(report, directory):
    base = json.loads((directory/'report.json').read_bytes())
    audit_pilot(base, directory)
    rows = probes()
    if (report['pilot_report_sha256'] != sha(directory/'report.json')
            or report['probe_sha256'] != probe_sha(rows) or report['complete'] is not True
            or report['script_sha256'] != sha(ROOT/'scripts/diagnose_body_transfer.py')
            or report['promotion'] is not False or set(report['arms']) != set(base['arms'])):
        raise ValueError('Transfer diagnostic provenance changed')
    image = report['validator_image']
    if image:
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image) or report['controls'] != dict(positive=[0]*96, negative=[1]*96):
            raise ValueError('Diagnostic Docker controls failed/missing')
    elif report['controls'] is not None: raise ValueError('Unverified controls')
    for arm, data in report['arms'].items():
        if data['model_sha256'] != base['arms'][arm]['model_sha256'] or data['weights_unchanged'] is not True:
            raise ValueError('Diagnostic changed model identity')
        actual = data['rows']
        if [r['id'] for r in actual] != [r['id'] for r in rows]: raise ValueError('Diagnostic omitted/reordered probes')
        for result, task in zip(actual, rows):
            if (result['cell'] != task['cell'] or result['request'] != task['request']
                    or result['output_sha256'] != hashlib.sha256(result['output'].encode()).hexdigest()
                    or result['metrics'] != score(result['output'], task)):
                raise ValueError('Diagnostic raw score differs')
            status = result['functional_status']
            if image and (type(status) is not int or status not in (0, 1)): raise ValueError('Invalid functional status')
            if not image and status is not None: raise ValueError('Unverified execution')
            if status == 0 and result['error']: raise ValueError('Failed generation cannot pass')
        expected = {cell: summary([r for r in actual if r['cell'] == cell]) for cell in sorted({r['cell'] for r in rows})}
        if data['cells'] != expected: raise ValueError('Diagnostic summary differs')
    return True


def run(source, directory, image=None):
    base = json.loads((directory/'report.json').read_bytes())
    audit_pilot(base, directory)
    if sha(source) != SOURCE: raise ValueError('Wrong diagnostic initialization')
    rows = probes()
    if len(rows) != 96: raise ValueError('Diagnostic grid changed')
    report = dict(pilot_report_sha256=sha(directory/'report.json'), probe_sha256=probe_sha(rows),
                  script_sha256=sha(ROOT/'scripts/diagnose_body_transfer.py'), validator_image=image,
                  controls=None, arms={}, complete=False, promotion=False,
                  scope='Post-hoc exposed orthogonal grid: names/arguments/wording, one alias pair per family; no independent claim')
    if image:
        import sandbox_validate
        positive, negative = [], []
        for row in rows:
            for code, bucket, expected in ((row['answer'], positive, 0),
                    ('def '+row['function']+'(*args, **kwargs):\n    return None\n', negative, 1)):
                status, diagnostic = sandbox_validate.validate(code, 'tests', row['cases'], image)
                bucket.append(status)
                if status != expected: raise RuntimeError('Diagnostic control failed: '+row['id']+' '+diagnostic)
        report['controls'] = dict(positive=positive, negative=negative)
    for arm in base['arms']:
        path = source if arm == 'source' else directory/arm/'candidate.nppai'
        if sha(path) != base['arms'][arm]['model_sha256']: raise ValueError('Candidate changed before diagnosis')
        backend = DenseLocalBackend(str(path), prompt_bytes=256, python_functions=True)
        before = weights_sha(backend)
        data = dict(model_sha256=sha(path), weights_unchanged=False, rows=[], cells={})
        report['arms'][arm] = data
        for row in rows:
            started = time.monotonic()
            text, error = '', None
            try:
                response = backend.generate(GenerationRequest(**row['request']))
                if response.model_id != 'sha256:'+data['model_sha256']: raise RuntimeError('Diagnostic response identity changed')
                text = response.output
            except (ValueError, RuntimeError) as exc:
                error = type(exc).__name__+': '+str(exc)
            elapsed = time.monotonic()-started
            status, diagnostic = None, 'Not executed'
            if image:
                if error or not text.strip(): status, diagnostic = 1, error or 'Empty candidate'
                else:
                    status, diagnostic = sandbox_validate.validate(text, 'tests', row['cases'], image)
                    if status not in (0, 1): raise RuntimeError('Diagnostic Docker unavailable: '+diagnostic)
            data['rows'].append(dict(id=row['id'], cell=row['cell'], request=row['request'], output=text,
                output_sha256=hashlib.sha256(text.encode()).hexdigest(), metrics=score(text, row),
                error=error, elapsed_seconds=elapsed, functional_status=status, diagnostics=diagnostic))
        data['cells'] = {cell: summary([r for r in data['rows'] if r['cell'] == cell]) for cell in sorted({r['cell'] for r in rows})}
        data['weights_unchanged'] = weights_sha(backend) == before and sha(path) == data['model_sha256']
        save(directory/'transfer-diagnostic.json', report)
        print(arm, json.dumps(data['cells']), flush=True)
    report['complete'] = True
    audit(report, directory)
    save(directory/'transfer-diagnostic.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.audit:
        audit(json.loads((args.output/'transfer-diagnostic.json').read_bytes()), args.output)
        print('Transfer diagnostic evidence audit passed')
    else: run(args.source, args.output, args.image)
