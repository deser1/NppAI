"""Frozen released-model evaluation. Candidate source executes only in Docker.

References/expected values never enter generation. No training or model selection.
The fresh suite becomes exposed regression evidence after its first readout.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_backends import DenseLocalBackend, GenerationRequest, format_prompt

SUITE = ROOT / 'evaluation/released-generalization-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_suite(directory=SUITE):
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    if sha(directory / 'tasks.json') != manifest['tasks_sha256']:
        raise ValueError('Frozen task bytes changed')
    tasks = json.loads((directory / 'tasks.json').read_bytes())
    if len(tasks) != manifest['tasks'] or len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Task set is incomplete or duplicated')
    for task in tasks:
        payload = GenerationRequest(**task['request'])
        format_prompt(payload, 256)  # Reject context overflow before any readout.
        if payload.assistant_prefix or payload.context:
            raise ValueError('Evaluation must not supply code or reference context')
        if not task['cases'] or any(c['function'] != task['function'] for c in task['cases']):
            raise ValueError('Requested callable and cases disagree')
        if not metrics(task['reference'], task)['signature']:
            raise ValueError('Authored control signature is invalid')
    return manifest, tasks


def metrics(source, task):
    """Parse only; never import or execute the model's code on the host."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return dict(nonempty=bool(source.strip()), syntax=False, signature=False)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    matching = [f for f in functions if f.name == task['function']]
    signature = False
    if len(matching) == 1:
        a = matching[0].args
        signature = ([x.arg for x in a.posonlyargs + a.args] == task['arguments']
                     and not a.posonlyargs and not a.kwonlyargs and not a.vararg
                     and not a.kwarg and not a.defaults)
    return dict(nonempty=bool(source.strip()), syntax=True, signature=signature)


def weights_sha(backend):
    import numpy as np
    def chunks(value):
        if isinstance(value, dict):
            for key in sorted(value):
                yield key.encode()
                yield from chunks(value[key])
        elif isinstance(value, list):
            for item in value:
                yield from chunks(item)
        elif isinstance(value, np.ndarray):
            yield str((value.shape, value.dtype)).encode()
            yield value.tobytes()
        else:
            yield repr(value).encode()
    h = hashlib.sha256()
    for chunk in chunks(backend.weights):
        h.update(chunk)
    return h.hexdigest()


def summarize(rows):
    result = {}
    for group in sorted({r['group'] for r in rows}):
        selected = [r for r in rows if r['group'] == group]
        result[group] = dict(total=len(selected), syntax=sum(r['metrics']['syntax'] for r in selected),
                            signature=sum(r['metrics']['signature'] for r in selected),
                            functional_passed=sum(r['functional_status'] == 0 for r in selected),
                            functional_measured=sum(r['functional_status'] in (0, 1) for r in selected),
                            generation_errors=sum(r['error'] is not None for r in selected))
    return result


def audit(report, manifest, tasks):
    if (report['checkpoint_sha256'] != manifest['checkpoint_sha256']
            or report['manifest_sha256'] != sha(SUITE / 'manifest.json')
            or report['tasks_sha256'] != manifest['tasks_sha256']
            or report['promotion'] is not False or report['complete'] is not True
            or set(report['arms']) != set(manifest['arms'])):
        raise ValueError('Report provenance/incomplete arm set')
    image = report['validator_image']
    if image:
        if (not re.fullmatch(r'sha256:[0-9a-f]{64}', image)
                or report['controls'] != {'positive': [0] * len(tasks), 'negative': [1] * len(tasks)}):
            raise ValueError('Docker controls missing or failed')
    elif report['controls'] is not None:
        raise ValueError('Controls cannot pass without Docker')
    for arm, data in report['arms'].items():
        rows = data['rows']
        if data['settings'] != manifest['arms'][arm] or data['weights_unchanged'] is not True:
            raise ValueError('Readout settings or weights changed')
        if [r['id'] for r in rows] != [t['id'] for t in tasks]:
            raise ValueError('Missing, reordered or duplicate readout')
        for r, t in zip(rows, tasks):
            if (r['request'] != t['request'] or r['group'] != t['group']
                    or r['output_sha256'] != hashlib.sha256(r['output'].encode()).hexdigest()
                    or r['metrics'] != metrics(r['output'], t)):
                raise ValueError('Readout/score differs from raw source')
            if r['error'] is None and r['response_model_id'] != 'sha256:' + manifest['checkpoint_sha256']:
                raise ValueError('Response checkpoint differs')
            if r['functional_status'] not in (None, 0, 1):
                raise ValueError('Infrastructure failure is not a candidate failure')
            if not image and r['functional_status'] is not None:
                raise ValueError('Functional result requires Docker')
            if image and r['functional_status'] is None:
                raise ValueError('Missing functional result')
            if r['functional_status'] == 0 and (r['error'] or not r['metrics']['signature']):
                raise ValueError('Invalid candidate cannot count as a success')
        if data['summary'] != summarize(rows):
            raise ValueError('Summary differs from raw rows')
    return True


def run(model, output, image=None):
    import numpy as np
    manifest, tasks = load_suite()
    if sha(model) != manifest['checkpoint_sha256']:
        raise ValueError('Only the frozen released checkpoint is permitted')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(checkpoint_sha256=sha(model), manifest_sha256=sha(SUITE / 'manifest.json'),
                  tasks_sha256=manifest['tasks_sha256'], source_commit=os.getenv('GITHUB_SHA'),
                  source_sha256={p: sha(ROOT / p) for p in ('scripts/evaluate_released_model.py',
                      'model_backends.py', 'python_function_decoding.py', 'sandbox_validate.py')},
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, system=platform.system()),
                  validator_image=image, controls=None, arms={}, complete=False, promotion=False,
                  scope=manifest['scope'], attempts=1)
    if image:
        import sandbox_validate
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
            raise ValueError('Immutable local Docker image ID required')
        positive, negative = [], []
        for t in tasks:
            for source, bucket, expected in ((t['reference'], positive, 0),
                    ('def ' + t['function'] + '(*args, **kwargs):\n    return None\n', negative, 1)):
                status, diagnostics = sandbox_validate.validate(source, 'tests', t['cases'], image)
                bucket.append(status)
                if status != expected:
                    raise RuntimeError('Docker control failed for ' + t['id'] + ': ' + diagnostics)
        report['controls'] = dict(positive=positive, negative=negative)
    save(output / 'report.json', report)
    for arm, settings in manifest['arms'].items():
        backend = DenseLocalBackend(str(model), **settings)
        before = weights_sha(backend)
        rows = []
        data = dict(settings=settings, rows=rows, weights_unchanged=False)
        report['arms'][arm] = data
        for task in tasks:
            start = time.monotonic()
            error = response_id = None
            source = ''
            response_meta = None
            try:
                # Only the frozen user request crosses the generation boundary.
                response = backend.generate(GenerationRequest(**task['request']))
                response_id = response.model_id
                if response_id != 'sha256:' + manifest['checkpoint_sha256']:
                    raise RuntimeError('Unexpected response checkpoint')
                source = response.output
                response_meta = dict(generated_tokens=response.generated_tokens, finish_reason=response.finish_reason)
            except (ValueError, RuntimeError) as exc:
                error = type(exc).__name__ + ': ' + str(exc)
            elapsed = time.monotonic() - start
            score = metrics(source, task)
            status, diagnostics = None, 'Not executed: Docker unavailable for this readout'
            if image:
                if error or not source.strip():
                    status, diagnostics = 1, error or 'Empty generated candidate'
                else:
                    status, diagnostics = sandbox_validate.validate(source, 'tests', task['cases'], image)
                    if status not in (0, 1):
                        raise RuntimeError('Docker infrastructure unavailable: ' + diagnostics)
            rows.append(dict(id=task['id'], group=task['group'], language=task['language'], family=task['family'],
                             request=task['request'], output=source, output_sha256=hashlib.sha256(source.encode()).hexdigest(),
                             response_model_id=response_id, response=response_meta, error=error, elapsed_seconds=elapsed,
                             metrics=score, functional_status=status, diagnostics=diagnostics))
            data['summary'] = summarize(rows)
            save(output / 'report.json', report)
            print(arm, task['id'], score, 'functional=', status, flush=True)
        data['weights_unchanged'] = before == weights_sha(backend) and sha(model) == manifest['checkpoint_sha256']
    report['complete'] = True
    audit(report, manifest, tasks)
    save(output / 'report.json', report)
    print(json.dumps({arm: d['summary'] for arm, d in report['arms'].items()}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--image')
    p.add_argument('--audit', type=Path)
    a = p.parse_args()
    if a.audit:
        manifest, tasks = load_suite()
        audit(json.loads(a.audit.read_bytes()), manifest, tasks)
        print('Frozen report audit passed')
    else:
        run(a.model, a.output, a.image)
