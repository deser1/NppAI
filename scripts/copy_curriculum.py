"""Fixed training-only identifier curriculum; experimental Python bundles only."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import portable_quality as quality

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('control', 'curriculum')
STEPS = 1024
WARMUP = 256
HEAD_SEED = 4242
SPEC = {
    'version': 1, 'initial_checkpoint_sha256': quality.CHECKPOINTS['both64'],
    'steps': STEPS, 'warmup_steps': WARMUP, 'head_seed': HEAD_SEED,
    'profile': quality.PROFILE, 'arithmetic': 'native FP32 gated readout in both arms',
    'optimizer': {'name': 'AdamW', 'lr': 0.001, 'betas': [0.9, 0.999], 'eps': 1e-8,
                  'weight_decay': 0.01, 'clip_norm': 1.0, 'reset': True},
    'base_frozen_first_steps_both_arms': WARMUP,
    'control': 'full response NLL for all updates',
    'curriculum': 'def name( plus EOS for first 256 updates, full response thereafter',
    'curriculum_auxiliary': {'source_position_nll': 1.0, 'gate_bce': 0.1,
                             'gate_positive': 'function-name response bytes only'},
    'decoding': {'context': 256, 'new_bytes': 160, 'greedy': True, 'prefix': '', 'repair': 0},
    'limits': 'One fixed bundled intervention and seed; not a coefficient/architecture search. '
              'Only train split enters optimization. Exposed development and functional regression '
              'tasks are not fresh holdout. Same-host repeats do not certify cross-CPU gated training. '
              'No native compatibility, default or weight promotion.'}


def schedule(dataset, arm):
    from scripts import resampled_name_training as base
    if arm not in ARMS:
        raise ValueError('unknown curriculum arm')
    for step, batch in enumerate(base.schedule(base.training_data(dataset), 'resampled', STEPS)):
        records = [dict(record) for _, record in batch]
        if arm == 'curriculum' and step < WARMUP:
            for record in records:
                record['answer'] = 'def ' + record['function'] + '('
        yield records


def schedule_evidence(dataset, arm):
    import hashlib
    digest = hashlib.sha256()
    targets = 0
    for records in schedule(dataset, arm):
        digest.update(quality.encoded(records))
        targets += sum(len(r['answer'].encode()) + 1 for r in records)
    return {'sha256': digest.hexdigest(), 'response_targets': targets,
            'updates': STEPS, 'examples': STEPS * 4}


def supervision(records, x, y):
    """Labels depend only on training records; query predicts byte query+1."""
    import torch
    from scripts.aligned_name_copy import alignment
    target_gate = torch.zeros_like(y, dtype=torch.float32)
    points = []
    for batch, record in enumerate(records):
        for query, source in alignment(record):
            if not 0 <= source < query < x.shape[1] or int(x[batch, source]) != int(y[batch, query]):
                raise ValueError('invalid causal copy supervision')
            target_gate[batch, query] = 1
            points.append((batch, query, source))
    return target_gate, points


def losses(model, records, x, y, auxiliary):
    import torch
    from torch.nn import functional as F
    from scripts.gated_copy_model import source_mask
    eligible = torch.stack([source_mask(x[i:i+1], r['prompt'])[0] for i, r in enumerate(records)])
    _, _, gate, attention, mixture = model.components(x, eligible)
    nll = F.nll_loss(mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log().reshape(-1, 256),
                     y.reshape(-1), ignore_index=-100)
    targets, points = supervision(records, x, y)
    pointer = -torch.stack([attention[b, q, s] for b, q, s in points]).clamp_min(1e-30).log().mean()
    active = y != -100
    gate_loss = F.binary_cross_entropy(gate.squeeze(-1)[active], targets[active])
    loss = nll + pointer + 0.1 * gate_loss if auxiliary else nll
    return loss, {'response_nll': float(nll.detach()), 'pointer_nll': float(pointer.detach()),
                  'gate_bce': float(gate_loss.detach())}


def initial(weights):
    import torch
    from scripts.check_dense_runtime_parity import torch_model
    from scripts.gated_copy_model import GatedCopyModel
    model = torch_model(weights)
    torch.manual_seed(HEAD_SEED)
    return GatedCopyModel(model)


def train(weights, dataset, directory, arm):
    import torch
    from scripts import resampled_name_training as base
    from scripts.training_repro import configure, state_digest
    from scripts.gated_copy_model import export_bundle, load_bundle
    configure(torch, quality.PROFILE)
    model = initial(weights)
    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    directory.mkdir(parents=True, exist_ok=True)
    trace = directory / 'trace.jsonl'
    trace.write_bytes(quality.encoded({'step': 0, 'parameters': state_digest(model)}))
    for step, records in enumerate(schedule(dataset, arm)):
        if step == WARMUP:
            for parameter in model.base.parameters():
                parameter.requires_grad_(True)
        samples = [base.sample(r) for r in records]
        x = torch.nn.utils.rnn.pad_sequence([s[0] for s in samples], batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence([s[1] for s in samples], batch_first=True, padding_value=-100)
        loss, parts = losses(model, records, x, y, arm == 'curriculum')
        if not math.isfinite(float(loss.detach())):
            raise ValueError('nonfinite curriculum loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('nonfinite curriculum parameter')
        with trace.open('ab') as stream:
            stream.write(quality.encoded({'step': step + 1, 'parameters': state_digest(model),
                                         'loss_hex': float(loss.detach()).hex(), 'losses': parts,
                                         'batch_sha256': quality.digest(quality.encoded(records))}))
        if step % 128 == 0 or step == STEPS - 1:
            print(json.dumps({'arm': arm, 'step': step + 1, **parts}), flush=True)
    model.eval()
    checkpoint = directory / 'experimental.nppaicopy'
    export_bundle(model, checkpoint, 'sha256:' + quality.CHECKPOINTS['both64'])
    loaded, _ = load_bundle(checkpoint)
    if state_digest(model) != state_digest(loaded):
        raise ValueError('curriculum roundtrip differs')
    return loaded, {'checkpoint_sha256': quality.digest(checkpoint.read_bytes()),
                    'trace_sha256': quality.digest(trace.read_bytes()), 'roundtrip_equal': True}


def gate_diagnostic(model, records):
    """Teacher-forced TRAINING-only values; separate from autonomous complete code."""
    import torch
    from scripts import resampled_name_training as base
    from scripts.gated_copy_model import source_mask, PromptAdapter
    from scripts.diagnose_name_bytes import infer, plan
    cases = []
    with torch.no_grad():
        for record in records:
            x, y = base.sample(record)
            x, y = x[None, :], y[None, :]
            _, _, gate, attention, _ = model.components(x, source_mask(x, record['prompt']))
            target, points = supervision([record], x, y)
            active = y != -100
            on, off = (target == 1) & active, (target == 0) & active
            cases.append({'id': record['id'], 'function': record['function'], 'prompt': record['prompt'],
                          'name_gate': gate.squeeze(-1)[on].tolist(),
                          'other_gate': gate.squeeze(-1)[off].tolist(),
                          'pointer_target_probability': [float(attention[b, q, s]) for b, q, s in points],
                          'name_bytes': infer(PromptAdapter(model, record['prompt']), *plan(record))})
    return cases


def functional(model, tasks, image):
    # The existing helper's literal "copy" arm selects learned gated generation.
    from scripts.train_gated_copy import functional_probes
    return functional_probes({'copy': model}, tasks, image)


def audit(report, dataset, tasks):
    from scripts import resampled_name_training as base
    from scripts import diagnose_name_bytes as diagnostic
    protocol_raw = (ROOT / 'evaluation/copy-curriculum-v1.json').read_bytes()
    protocol = json.loads(protocol_raw)
    groups, fingerprints = quality.inputs(dataset, tasks)
    if (report['protocol_sha256'] != quality.digest(protocol_raw) or report['spec'] != SPEC or
            protocol['spec'] != SPEC or report['evaluation_inputs'] != fingerprints or
            fingerprints != protocol['evaluation_inputs'] or report['promotion'] is not False or
            set(report['arms']) != set(ARMS)):
        raise ValueError('curriculum report provenance differs')
    if report['before_summary'] != quality.audit_groups(report['before'], groups):
        raise ValueError('initial curriculum summary differs')
    import re
    expected_tasks = json.loads(tasks.read_bytes())
    for arm, result in report['arms'].items():
        if (schedule_evidence(dataset, arm) != protocol['schedules'][arm] or
                result['same_host_repeat_equal'] is not True or
                result['parameters_unchanged_by_evaluation'] is not True or
                result['identity']['roundtrip_equal'] is not True or
                any(not re.fullmatch(r'[0-9a-f]{64}', result['identity'][name])
                    for name in ('checkpoint_sha256', 'trace_sha256'))):
            raise ValueError('curriculum training identity differs')
        if result['summary'] != quality.audit_groups(result['after'], groups):
            raise ValueError('curriculum summary differs')
        if result['retention'] != base.retention(report['before'], result['after']):
            raise ValueError('curriculum retention differs')
        training = groups['train/original'] + groups['train/renamed']
        cases = result['training_only_gate_diagnostic']
        if len(cases) != len(training):
            raise ValueError('training-only diagnostic count differs')
        for case, record in zip(cases, training):
            _, target = diagnostic.plan(record)
            response_count = len(record['answer'].encode()) + 1
            if (any(case[key] != record[key] for key in ('id', 'function', 'prompt')) or
                    len(case['name_gate']) != len(target) or
                    len(case['pointer_target_probability']) != len(target) or
                    len(case['other_gate']) != response_count - len(target)):
                raise ValueError('training-only diagnostic identity differs')
            probabilities = case['name_gate'] + case['other_gate'] + case['pointer_target_probability']
            if any(not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
                raise ValueError('invalid copy/gate probability')
            item = case['name_bytes']
            if item['metrics'] != diagnostic.metrics(target, item['predictions'], item['target_probabilities'], item['scaffolded_bytes']):
                raise ValueError('conditional training-byte score differs')
            check = item['prefix_check']
            if (check['consistent'] is not True or check['count'] != len(target) or
                    not math.isfinite(check['max_logit_delta']) or not 0 <= check['max_logit_delta'] <= 0.01):
                raise ValueError('training causal prefix differs')
        f = result['functional']
        if (f['manifest_sha256'] != fingerprints['tasks_sha256'] or set(f['models']) != {'copy'} or
                not re.fullmatch(r'sha256:[0-9a-f]{64}', f['image'])):
            raise ValueError('functional curriculum provenance differs')
        measured = f['models']['copy']
        if measured['tasks'] != len(expected_tasks) or len(measured['cases']) != len(expected_tasks):
            raise ValueError('functional curriculum count differs')
        for case, task in zip(measured['cases'], expected_tasks):
            if (any(case[key] != value for key, value in task.items()) or
                    case['suite_sha256'] != fingerprints['suites'][task['task_id']] or
                    case['category'] not in ('empty', 'syntax_failure', 'functional_failure', 'passed')):
                raise ValueError('functional curriculum case differs')
        if measured['passed'] != sum(c['category'] == 'passed' for c in measured['cases']):
            raise ValueError('functional curriculum pass count differs')
    return True


def retained(directory):
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    raw = (directory / 'report.json').read_bytes()
    if quality.digest(raw) != manifest['metadata_sha256']:
        raise ValueError('retained curriculum metadata differs')
    report = json.loads(raw)
    for item in manifest['chunks']:
        raw = (directory / item['file']).read_bytes()
        if quality.digest(raw) != item['sha256']:
            raise ValueError('retained curriculum chunk differs')
        value = json.loads(raw)
        if item['kind'] == 'before':
            report['before'][item['group']] = value
        elif item['kind'] == 'after':
            report['arms'][item['arm']]['after'][item['group']] = value
        elif item['kind'] == 'diagnostic':
            report['arms'][item['arm']]['training_only_gate_diagnostic'].extend(value)
        else:
            raise ValueError('unknown curriculum evidence chunk')
    if quality.digest(quality.encoded(report)) != manifest['report_sha256']:
        raise ValueError('reconstructed curriculum report differs')
    return report


def audit_trace(raw, dataset, arm, expected_sha256):
    import re
    if quality.digest(raw) != expected_sha256:
        raise ValueError('curriculum trace digest differs')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != STEPS + 1 or [row['step'] for row in rows] != list(range(STEPS + 1)):
        raise ValueError('curriculum trace steps differ')
    for row in rows:
        if not re.fullmatch(r'[0-9a-f]{64}', row['parameters']):
            raise ValueError('curriculum trace parameter hash differs')
    for records, row in zip(schedule(dataset, arm), rows[1:]):
        if row['batch_sha256'] != quality.digest(quality.encoded(records)):
            raise ValueError('curriculum trace batch differs')
        components = row['losses']
        if set(components) != {'response_nll', 'pointer_nll', 'gate_bce'} or any(
                not math.isfinite(v) or v < 0 for v in components.values()):
            raise ValueError('curriculum trace loss components differ')
        loss = float.fromhex(row['loss_hex'])
        expected = components['response_nll']
        if arm == 'curriculum':
            expected += components['pointer_nll'] + 0.1 * components['gate_bce']
        if not math.isfinite(loss) or not math.isclose(loss, expected, rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError('curriculum trace total loss differs')
    return rows


def retained_trace(directory, arm):
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    data = []
    for item in manifest['traces'][arm]:
        raw = (directory / item['file']).read_bytes()
        if quality.digest(raw) != item['sha256']:
            raise ValueError('retained curriculum trace chunk differs')
        data.extend(json.loads(raw))
    return b''.join(quality.encoded(row) for row in data)


def run(archive, dataset, tasks, output, image):
    import torch
    from scripts import resampled_name_training as base
    from scripts.check_dense_runtime_parity import load_weights
    from scripts.train_gated_copy import evaluate
    from scripts.training_repro import configure, provenance, state_digest
    protocol_raw = (ROOT / 'evaluation/copy-curriculum-v1.json').read_bytes()
    protocol = json.loads(protocol_raw)
    if protocol['spec'] != SPEC:
        raise ValueError('curriculum protocol differs')
    for arm in ARMS:
        if schedule_evidence(dataset, arm) != protocol['schedules'][arm]:
            raise ValueError('curriculum schedule differs')
    groups, fingerprints = quality.inputs(dataset, tasks)
    if fingerprints != protocol['evaluation_inputs']:
        raise ValueError('curriculum evaluation differs')
    weights = load_weights(quality.checkpoints(archive, output / 'source')['both64'])
    configure(torch, quality.PROFILE)
    model = initial(weights).eval()
    before = evaluate(model, groups, True, False)
    report = {'protocol_sha256': quality.digest(protocol_raw), 'spec': SPEC,
              'evaluation_inputs': fingerprints, 'before': before,
              'before_summary': quality.audit_groups(before, groups),
              'runtime': provenance(torch, quality.PROFILE), 'arms': {}, 'promotion': False}
    (output / 'report.json').write_bytes(quality.encoded(report))
    for arm in ARMS:
        first, identity = train(weights, dataset, output / arm / '0', arm)
        _, repeat = train(weights, dataset, output / arm / '1', arm)
        for name in ('trace.jsonl', 'experimental.nppaicopy'):
            if (output / arm / '0' / name).read_bytes() != (output / arm / '1' / name).read_bytes():
                raise ValueError('same-host training repeat differs: ' + arm)
        if identity != repeat:
            raise ValueError('same-host repeat identity differs')
        audit_trace((output / arm / '0/trace.jsonl').read_bytes(), dataset, arm, identity['trace_sha256'])
        parameter_state = state_digest(first)
        after = evaluate(first, groups, True, False)
        # The diagnostic sees only the same 16 original and 16 renamed TRAIN views.
        diagnostic = gate_diagnostic(first, groups['train/original'] + groups['train/renamed'])
        result = {'identity': identity, 'same_host_repeat_equal': True,
                  'after': after, 'summary': quality.audit_groups(after, groups),
                  'retention': base.retention(before, after),
                  'training_only_gate_diagnostic': diagnostic, 'functional': functional(first, tasks, image),
                  'parameters_unchanged_by_evaluation': state_digest(first) == parameter_state}
        if not result['parameters_unchanged_by_evaluation']:
            raise ValueError('evaluation mutated model')
        report['arms'][arm] = result
        (output / 'report.json').write_bytes(quality.encoded(report))
        print(json.dumps({'arm': arm, 'summary': result['summary'],
                          'functional_passed': result['functional']['models']['copy']['passed']}), flush=True)
    audit(report, dataset, tasks)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'dataset', 'tasks', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    run(args.archive, args.dataset, args.tasks, args.output, args.image)
