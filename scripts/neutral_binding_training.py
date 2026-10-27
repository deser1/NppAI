"""Fixed neutral-name and prompt-position training experiment; evaluation never updates weights.

Development data is authored and exposed, not an independent generalization test.
References and cases never enter inference. Candidate execution is Docker-only.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_backends import DenseLocalBackend, GenerationRequest, format_prompt
from scripts.factorial_repair import dataset as factorial_data
from scripts.evaluate_released_model import weights_sha

DATA = ROOT / 'evaluation/neutral-binding-v1.json'
SOURCE = 'f8ef88dccbbd3280c21e5acd51c41c6744e6ba597aa63686f3f43fd96beac229'
SOURCE_FILES = ('scripts/neutral_binding_training.py', 'scripts/neutral_binding_data.py', 'scripts/factorial_repair.py',
                'model_backends.py', 'python_function_decoding.py',
                'scripts/check_dense_runtime_parity.py', 'scripts/training_repro.py', 'train_nppai.py', 'sandbox_validate.py',
                'evaluation/function-body-v1/train.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def score(source, row):
    result = dict(nonempty=bool(source.strip()), syntax=False, signature=False, value_return=False,
                  exact=source == row['answer'])
    try:
        tree = ast.parse(source)
        result['syntax'] = bool(source.strip())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == row['function']]
        if len(funcs) == 1:
            f = funcs[0]
            a = f.args
            result['signature'] = ([n.arg for n in a.args] == row['arguments']
                and not a.posonlyargs and not a.kwonlyargs and not a.kwarg and not a.vararg and not a.defaults)
            # A return inside a nested helper is not a return from the requested function.
            pending = list(f.body)
            while pending:
                node = pending.pop()
                if isinstance(node, ast.Return) and node.value is not None:
                    result['value_return'] = True
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    pending.extend(ast.iter_child_nodes(node))
    except (SyntaxError, ValueError, RecursionError):
        pass
    return result


def load_bank():
    from scripts.neutral_binding_data import banks, digest
    manifest = json.loads(DATA.read_bytes())
    training, development = banks()
    if (manifest['source_sha256'] != SOURCE
            or manifest['training_sha256'] != digest(training)
            or manifest['development_sha256'] != digest(development)):
        raise ValueError('Frozen neutral bank changed')
    for row in training + development:
        payload = GenerationRequest(**row['request'])
        format_prompt(payload, 256)
        if payload.context or payload.assistant_prefix or not all(score(row['answer'], row).values()):
            raise ValueError('Invalid authored bank row')
    return manifest, training, development


def original_rows():
    raw = json.loads((ROOT/'evaluation/staged-protocol-v1/train.json').read_bytes())
    result = []
    for row in raw:
        f = ast.parse(row['answer']).body[0]
        result.append(dict(id=row['id'], family=row['family'], function=f.name,
                           arguments=[a.arg for a in f.args.args], answer=row['answer'],
                           request=dict(prompt=row['prompt'], max_new_tokens=128)))
    return result


def factorial_rows():
    train, dev = factorial_data()
    rows = dev + [r for r in train if r['id'].startswith('plain-')]
    for row in rows:
        f = ast.parse(row['answer']).body[0]
        row['arguments'] = [a.arg for a in f.args.args]
    return rows


def samples(row, context, width):
    import torch
    payload = GenerationRequest(**row['request'])
    prefix = (format_prompt(payload, width) + payload.assistant_prefix).encode()
    answer = row['answer'].encode()
    assistance = payload.assistant_prefix.encode()
    if not answer.startswith(assistance):
        raise ValueError('Target conflicts with authored prefix')
    tokens = list(prefix) + list(answer[len(assistance):]) + [0]
    if len(tokens) > context:
        raise ValueError('No supervision truncation allowed')
    x, y = torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])
    y[:len(prefix)-1] = -100
    return x, y


def schedule(manifest, old_count, factorial_count, train_count):
    rng = random.Random(manifest['protocol']['seed'])
    offsets = random.Random(manifest['protocol']['position_seed'])
    return [dict(original=rng.sample(range(old_count), 2), factorial=rng.sample(range(factorial_count), 2),
                 bodies=rng.sample(range(train_count), 4), shifts=[offsets.choice((0,8,16)) for _ in range(4)])
            for _ in range(manifest['protocol']['updates'])]


def selected_samples(arm, draw, old, factorial, bodies):
    if arm not in ('fixed_position', 'position_jitter'):
        raise ValueError('Unknown training arm')
    offsets = draw['shifts'] if arm == 'position_jitter' else [0]*4
    return ([old[i] for i in draw['original']] + [factorial[i] for i in draw['factorial']]
            + [bodies[(i, shift)] for i, shift in zip(draw['bodies'], offsets)])


def train(source, output):
    import torch
    import torch.nn.functional as F
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    from scripts.training_repro import state_digest
    from train_nppai import export_to_bin
    manifest, bank, _ = load_bank()
    if sha(source) != SOURCE:
        raise ValueError('Wrong published initialization')
    p = manifest['protocol']
    if p['updates'] != 1024 or p['arms'] != ['fixed_position', 'position_jitter']:
        raise ValueError('This pilot has a fixed budget and two arms')
    torch.set_num_threads(2)
    torch.manual_seed(p['seed'])
    torch.use_deterministic_algorithms(True)
    initial = torch_model(load_weights(source))
    for layer in initial.modules():
        if isinstance(layer, torch.nn.RMSNorm): layer.eps = 1e-5
    before = state_digest(initial)
    old = [samples(r, initial.max_seq_len, 0) for r in original_rows()]
    factorial_train, _ = factorial_data()
    fact = [samples(r, initial.max_seq_len, 256) for r in factorial_train]
    bodies = {}
    for i, row in enumerate(bank):
        for shift in (0, 8, 16):
            shifted = copy.deepcopy(row)
            shifted['request']['prompt'] = ' '*shift + row['request']['prompt']
            bodies[i, shift] = samples(shifted, initial.max_seq_len, 256)
    draws = schedule(manifest, len(old), len(fact), len(bank))
    output.mkdir(parents=True, exist_ok=True)
    save(output/'schedule.json', draws)
    training = dict(source_sha256=SOURCE, source_commit=os.getenv('GITHUB_SHA'),
                    source_sha256_files={f: sha(ROOT/f) for f in SOURCE_FILES},
                    bank_manifest_sha256=sha(DATA), protocol=p,
                    original_replay_sha256=sha(ROOT/'evaluation/staged-protocol-v1/train.json'),
                    factorial_training_sha256=hashlib.sha256(json.dumps(factorial_train, sort_keys=True).encode()).hexdigest(),
                    schedule_sha256=sha(output/'schedule.json'), runtime=dict(torch=torch.__version__), arms={})
    for arm in p['arms']:
        model = copy.deepcopy(initial).train()
        o = p['optimizer']
        optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], betas=tuple(o['betas']),
                                      eps=o['eps'], weight_decay=o['weight_decay'])
        trace = []
        start = time.monotonic()
        for index, draw in enumerate(draws):
            batch = selected_samples(arm, draw, old, fact, bodies)
            x = torch.nn.utils.rnn.pad_sequence([s[0] for s in batch], batch_first=True)
            y = torch.nn.utils.rnn.pad_sequence([s[1] for s in batch], batch_first=True, padding_value=-100)
            losses = F.cross_entropy(model(x).transpose(1, 2), y, ignore_index=-100, reduction='none')
            loss = (losses.sum(1) / (y != -100).sum(1)).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), p['clip_grad_norm'])
            optimizer.step()
            trace.append(dict(step=index+1, loss=float(loss.detach())))
            if index % 128 == 0:
                print(arm, index+1, '/', len(draws), 'loss', trace[-1]['loss'], flush=True)
        model.eval()
        folder = output/arm
        save(folder/'trace.json', trace)
        export_to_bin(model, str(folder/'candidate.nppai'))
        training['arms'][arm] = dict(model_sha256=sha(folder/'candidate.nppai'),
            trace_sha256=sha(folder/'trace.json'), state_sha256=state_digest(model),
            elapsed_seconds=time.monotonic()-start)
        save(output/'training.json', training)
    if state_digest(initial) != before or sha(source) != SOURCE:
        raise ValueError('Initialization mutated')
    training['initialization_unchanged'] = True
    save(output/'training.json', training)
    return training


def groups():
    _, train_rows, dev = load_bank()
    probes = [r for r in train_rows if r['id'].endswith('-0-0-0') or r['id'].endswith('-0-0-2')]
    return dict(development=dev, training_probes=probes, original=original_rows(), factorial=factorial_rows())


def summary(rows):
    return dict(total=len(rows), **{m: sum(r['metrics'][m] for r in rows)
                for m in ('nonempty', 'syntax', 'signature', 'value_return', 'exact')},
                functional_measured=sum(r['functional_status'] in (0, 1) for r in rows),
                functional_passed=sum(r['functional_status'] == 0 for r in rows),
                generation_errors=sum(r['error'] is not None for r in rows))


def decision(report, manifest):
    if not report['validator_image']:
        return dict(status='not_measured', eligible_for_fresh_evaluation=False, promotion=False)
    arms = report['arms']
    candidate, baseline, control = (arms[a] for a in ('position_jitter', 'source', 'fixed_position'))
    primary = {r['id'] for r in groups()['development'] if r['primary']}
    def passed(arm):
        return sum(r['id'] in primary and r['functional_status'] == 0 for r in arm['groups']['development'])
    gain = passed(candidate)
    minimum = manifest['gate']['min_development_functional']
    reasons = []
    if gain < minimum: reasons.append('development functional minimum not reached')
    if gain <= max(passed(baseline), passed(control)):
        reasons.append('no strict functional improvement over source and matched fixed-position control')
    for group, key in (('original', 'exact'), ('factorial', 'functional_status')):
        old, new = baseline['groups'][group], candidate['groups'][group]
        if [r['id'] for r in old] != [r['id'] for r in new]:
            raise ValueError('Retention rows omitted or reordered')
        for a, b in zip(old, new):
            was_hit = a['metrics']['exact'] if key == 'exact' else a[key] == 0
            kept = b['metrics']['exact'] if key == 'exact' else b[key] == 0
            if was_hit and not kept: reasons.append('lost prior '+group+' hit: '+a['id'])
    return dict(status='rejected' if reasons else 'development_gate_passed', reasons=reasons,
                eligible_for_fresh_evaluation=not reasons, promotion=False)


def audit(report, directory):
    manifest, bank, dev = load_bank()
    expected = groups()
    training = json.loads((directory/'training.json').read_bytes())
    if (sha(directory/'training.json') != report['training_sha256'] or report['promotion'] is not False
            or report['complete'] is not True or training.get('initialization_unchanged') is not True
            or training['bank_manifest_sha256'] != sha(DATA)
            or training['source_sha256'] != SOURCE or training['protocol'] != manifest['protocol']
            or training['schedule_sha256'] != sha(directory/'schedule.json')
            or training['original_replay_sha256'] != sha(ROOT/'evaluation/staged-protocol-v1/train.json')
            or training['factorial_training_sha256'] != hashlib.sha256(json.dumps(factorial_data()[0], sort_keys=True).encode()).hexdigest()
            or set(training['arms']) != set(manifest['protocol']['arms'])
            or training['source_sha256_files'] != {f: sha(ROOT/f) for f in SOURCE_FILES}
            or set(report['arms']) != {'source', 'fixed_position', 'position_jitter'}):
        raise ValueError('Incomplete or changed experiment provenance')
    expected_draws = schedule(manifest, len(expected['original']), len(factorial_data()[0]), len(bank))
    if json.loads((directory/'schedule.json').read_bytes()) != expected_draws:
        raise ValueError('Optimization draw schedule changed')
    for name, item in training['arms'].items():
        if sha(directory/name/'candidate.nppai') != item['model_sha256'] or sha(directory/name/'trace.json') != item['trace_sha256']:
            raise ValueError('Trained model/trace identity changed')
        trace = json.loads((directory/name/'trace.json').read_bytes())
        if ([r['step'] for r in trace] != list(range(1, 1025))
                or any(not __import__('math').isfinite(r['loss']) for r in trace)):
            raise ValueError('Incomplete fixed-budget training')
    image = report['validator_image']
    if image:
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image) or report['controls'] != dict(positive=[0]*(len(bank)+len(dev)), negative=[1]*len(dev)):
            raise ValueError('Real Docker controls missing')
    elif report['controls'] is not None:
        raise ValueError('Unverified Docker controls')
    for arm, data in report['arms'].items():
        model_sha = SOURCE if arm == 'source' else training['arms'][arm]['model_sha256']
        if data['model_sha256'] != model_sha or not data['weights_unchanged'] or set(data['groups']) != set(expected):
            raise ValueError('Readout model/group identity changed')
        for group, tasks in expected.items():
            rows = data['groups'][group]
            if [r['id'] for r in rows] != [r['id'] for r in tasks]: raise ValueError('Incomplete/omitted readout')
            for row, task in zip(rows, tasks):
                if (row['request'] != task['request'] or row['output_sha256'] != hashlib.sha256(row['output'].encode()).hexdigest()
                        or row['metrics'] != score(row['output'], task)):
                    raise ValueError('Metrics differ from raw output')
                functional = bool(image and group in ('development', 'factorial'))
                if functional and (type(row['functional_status']) is not int or row['functional_status'] not in (0, 1)):
                    raise ValueError('Functional failure/infrastructure status confused')
                if not functional and row['functional_status'] is not None:
                    raise ValueError('Unmeasured task cannot pass')
                if row['functional_status'] == 0 and row['error']:
                    raise ValueError('Failed generation cannot pass')
            if data['summary'][group] != summary(rows): raise ValueError('Raw summary mismatch')
            if group == 'development':
                cells = {c: summary([r for r, t in zip(rows, tasks) if t['cell'] == c])
                         for c in sorted({t['cell'] for t in tasks})}
                if data['cells'] != cells: raise ValueError('Cell summaries differ')
    if report['arms']['source']['summary']['original']['exact'] != 96:
        raise ValueError('Source no longer reproduces 96 original hits')
    if report['decision'] != decision(report, manifest): raise ValueError('Decision hides failures')
    return True


def measure(source, directory, image=None):
    import numpy as np
    manifest, bank, dev = load_bank()
    training = json.loads((directory/'training.json').read_bytes())
    if sha(source) != SOURCE or training['bank_manifest_sha256'] != sha(DATA):
        raise ValueError('Wrong frozen source or bank')
    if image and not re.fullmatch(r'sha256:[0-9a-f]{64}', image): raise ValueError('Immutable Docker image required')
    paths = dict(source=source, **{a: directory/a/'candidate.nppai' for a in manifest['protocol']['arms']})
    for arm, path in paths.items():
        expected_sha = SOURCE if arm == 'source' else training['arms'][arm]['model_sha256']
        if sha(path) != expected_sha: raise ValueError('Candidate changed before readout')
    report = dict(training_sha256=sha(directory/'training.json'), source_commit=os.getenv('GITHUB_SHA'),
                  runtime=dict(numpy=np.__version__), validator_image=image, controls=None, arms={}, complete=False, promotion=False)
    if image:
        import sandbox_validate
        positive, negative = [], []
        for row in bank+dev:
            status, diagnostic = sandbox_validate.validate(row['answer'], 'tests', row['cases'], image)
            positive.append(status)
            if status != 0: raise RuntimeError('Authored reference/Docker failure: '+row['id']+' '+diagnostic)
        for row in dev:
            wrong = 'def '+row['function']+'(*args, **kwargs):\n    return None\n'
            status, diagnostic = sandbox_validate.validate(wrong, 'tests', row['cases'], image)
            negative.append(status)
            if status != 1: raise RuntimeError('Negative control failure: '+diagnostic)
        report['controls'] = dict(positive=positive, negative=negative)
    expected = groups()
    for arm, path in paths.items():
        backend = DenseLocalBackend(str(path), prompt_bytes=256, python_functions=True)
        before = weights_sha(backend)
        data = dict(model_sha256=sha(path), groups={}, summary={}, weights_unchanged=False)
        report['arms'][arm] = data
        for group, tasks in expected.items():
            rows = []
            for row in tasks:
                start = time.monotonic()
                error = None
                text = ''
                try:
                    response = backend.generate(GenerationRequest(**row['request']))
                    if response.model_id != 'sha256:'+data['model_sha256']: raise RuntimeError('Readout model mismatch')
                    text = response.output
                except (ValueError, RuntimeError) as exc:
                    error = type(exc).__name__+': '+str(exc)
                elapsed = time.monotonic()-start
                status, diagnostic = None, 'Not executed'
                if image and group in ('development', 'factorial'):
                    if error or not text.strip(): status, diagnostic = 1, error or 'Empty candidate'
                    else:
                        status, diagnostic = sandbox_validate.validate(text, 'tests', row['cases'], image)
                        if status not in (0, 1): raise RuntimeError('Docker infrastructure unavailable: '+diagnostic)
                rows.append(dict(id=row['id'], request=row['request'], output=text,
                    output_sha256=hashlib.sha256(text.encode()).hexdigest(), metrics=score(text, row),
                    error=error, elapsed_seconds=elapsed, functional_status=status, diagnostics=diagnostic))
            data['groups'][group] = rows
            data['summary'][group] = summary(rows)
            if group == 'development':
                data['cells'] = {c: summary([r for r, t in zip(rows, tasks) if t['cell'] == c])
                                 for c in sorted({t['cell'] for t in tasks})}
            save(directory/'report.json', report)
            print(arm, group, data['summary'][group], flush=True)
        data['weights_unchanged'] = weights_sha(backend) == before and sha(path) == data['model_sha256']
    report['complete'] = True
    report['decision'] = decision(report, manifest)
    audit(report, directory)
    save(directory/'report.json', report)
    print(json.dumps(report['decision'], indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('train', 'measure', 'audit'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image')
    a = parser.parse_args()
    if a.phase == 'train': train(a.source, a.output)
    elif a.phase == 'measure': measure(a.source, a.output, a.image)
    else:
        audit(json.loads((a.output/'report.json').read_bytes()), a.output)
        print('Body training evidence audit passed')
