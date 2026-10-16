"""Fixed 1024-update portable native/both64 repeatability experiment; no promotion."""
import argparse
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.arithmetic_variants import arithmetic_mode, complete_step, digest
from scripts.first_step_ops import encoded, gradients
from scripts.training_repro import configure, provenance, state_digest
STEPS = 1024
SOURCE_MODEL_ID = 'sha256:161d6303c7e77f005cbabba5a1790b9d6a6dfc1812c121bfbb3f1cb6820e5b1a'

PROFILE = 'portable-cpu-v1'
VARIANTS = ('native', 'both64')
GATE_FILES = {
    'comparison.json': '7ae898d7a85cb7b7fe56c485e6043038765fb932620349cd2bb31916e51da443',
    'validation-comparison.json': 'af6666d9fabb2cab99b71d5febaeb6773c61457cfd38693cbbda1168f186636a',
}


def first_step_gate(root):
    evidence = []
    for filename, expected in GATE_FILES.items():
        raw = (root / filename).read_bytes()
        if digest(raw) != expected:
            raise ValueError('fixed first-step evidence differs')
        result = json.loads(raw)[PROFILE]
        if len(result['reports']) != 4 or len(result['pairs']) != 6:
            raise ValueError('first-step coverage incomplete')
        cross_vendor = sum(pair['cross_vendor'] for pair in result['pairs'])
        if not cross_vendor:
            raise ValueError('first-step Intel/AMD coverage missing')
        for report in result['reports']:
            if report['variant'] != 'both64' or report['source_model_id'] != SOURCE_MODEL_ID:
                raise ValueError('first-step recipe/source differs')
            if not report['trace_preserves_untraced_step'] or not report['trace_repeat']['identical'] or not report['saved_repeat']:
                raise ValueError('unreliable first-step evidence')
        if any(not all(p['step'].values()) or not p['first_operator_difference']['identical'] or not p['first_saved_difference']['identical'] for p in result['pairs']):
            raise ValueError('first-step candidate gate failed')
        evidence.append({'file': filename, 'sha256': expected, 'cross_vendor_pairs': cross_vendor,
                         'initial_parameters': result['reports'][0]['reference']['state']['initial_parameters'],
                         'reference': result['reports'][0]['reference']})
    return evidence


def update(model, optimizer, x, y, variant):
    import torch
    import torch.nn.functional as F
    mode = arithmetic_mode(variant)
    with mode:
        mode.phase = 'forward'; logits = model(x)
        mode.phase = 'loss'; loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        mode.phase = 'backward'; loss.backward()
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
        raise ValueError('nonfinite forward/loss')
    if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
        raise ValueError('nonfinite gradient')
    raw = gradients(model)
    with mode:
        mode.phase = 'clip'; torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    clipped = gradients(model)
    with mode:
        mode.phase = 'optimizer'; optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('nonfinite parameters')
    if variant == 'both64' and not all(mode.overrides.get(k, 0) > 0 for k in (
        'aten._softmax.default', 'aten._softmax_backward_data.default', 'aten.sqrt.default')):
        raise ValueError('missing arithmetic replacement')
    return {'loss_hex': float(loss.detach()).hex(), 'logits': digest(logits.detach().contiguous().numpy().tobytes()),
            'raw_gradients': raw, 'clipped_gradients': clipped, 'updated_parameters': state_digest(model)}, mode.overrides


def train(weights, batches, output, variant):
    import torch
    from train_nppai import export_to_bin
    from scripts.check_dense_runtime_parity import torch_model, load_weights
    if variant not in VARIANTS:
        raise ValueError('unknown full-training variant')
    configure(torch, PROFILE); output.mkdir(parents=True, exist_ok=True)
    model = torch_model(weights).train(); initial = state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    trace = output / 'trace.jsonl'; trace.write_bytes(encoded({'step': 0, 'parameters': initial}))
    first = None; calls = {}; steps = 0
    for steps, (x, y) in enumerate(batches, 1):
        if steps == 1:
            with torch.random.fork_rng(devices=[]):
                reference = complete_step(weights, x, y, PROFILE, variant)
        state, overrides = update(model, optimizer, x, y, variant)
        if steps == 1:
            first = {'state': {'initial_parameters': initial, **state}, 'overrides': overrides}
            if first != reference:
                raise RuntimeError('training update differs from fixed first-step reference')
        for op, count in overrides.items(): calls[op] = calls.get(op, 0) + count
        row = {'step': steps, 'parameters': state['updated_parameters'], 'loss_hex': state['loss_hex'],
               'logits': state['logits'], 'raw_gradients': digest(encoded(state['raw_gradients'])),
               'clipped_gradients': digest(encoded(state['clipped_gradients'])),
               'inputs': digest(encoded({'x': list(x.shape), 'y': list(y.shape)}) + x.numpy().tobytes() + y.numpy().tobytes())}
        with trace.open('ab') as stream: stream.write(encoded(row))
        if steps % 128 == 0: print(variant, 'step', steps, 'loss', state['loss_hex'], flush=True)
    if not steps: raise ValueError('empty training schedule')
    checkpoint = output / 'experimental.nppai'; export_to_bin(model.eval(), str(checkpoint))
    loaded = torch_model(load_weights(checkpoint))
    if any(not torch.equal(a, b) for a, b in zip(model.parameters(), loaded.parameters())):
        raise RuntimeError('export roundtrip differs')
    return {'steps': steps, 'first_step': first, 'initial_parameters': initial,
            'final_parameters': state_digest(model), 'trace_sha256': digest(trace.read_bytes()),
            'checkpoint_sha256': digest(checkpoint.read_bytes()), 'overrides': calls,
            'exported_parameters_equal': True}


def run(source, dataset, output):
    import torch
    from scripts import resampled_name_training as base
    from scripts.run_local_model_smoke import pinned_backend
    from scripts.aligned_name_copy import CONTROL_MODEL_SHA256
    from scripts.check_dense_runtime_parity import torch_model
    if base.STEPS != STEPS or base.SOURCE_MODEL_ID != SOURCE_MODEL_ID:
        raise ValueError('fixed training protocol differs')
    configure(torch, PROFILE); output.mkdir(parents=True, exist_ok=True)
    gate = first_step_gate(Path(__file__).resolve().parents[1] / 'docs/evidence/both64-trace')
    backend = pinned_backend(source)
    if backend.model_id != base.SOURCE_MODEL_ID: raise ValueError('pinned source differs')
    initial = state_digest(torch_model(backend.weights))
    if any(r['initial_parameters'] != initial for r in gate): raise ValueError('gate initial weights differ')
    rows = base.training_data(dataset); batches = []
    for selected in base.schedule(rows, 'resampled'):
        samples = [base.sample(r) for _, r in selected]
        batches.append((torch.nn.utils.rnn.pad_sequence([s[0] for s in samples], batch_first=True),
                        torch.nn.utils.rnn.pad_sequence([s[1] for s in samples], batch_first=True, padding_value=-100)))
    preflight = complete_step(backend.weights, *batches[0], PROFILE, 'both64')
    preflight_passed = all(preflight == g['reference'] for g in gate)
    (output / 'preflight.json').write_bytes(encoded({'profile': PROFILE, 'environment': provenance(torch, PROFILE),
        'first_step': preflight, 'matches_frozen_candidate': preflight_passed, 'promotion': False}))
    if not preflight_passed: raise RuntimeError('current-host candidate first-step gate failed; no full training')
    arms = {}; report = {'version': 1, 'profile': PROFILE, 'source_model_id': backend.model_id,
        'gate': gate, 'preflight_passed': True, 'steps': base.STEPS, 'schedule': base.schedule_evidence(rows, 'resampled'),
        'training_sha256': digest(base.encoded(rows)), 'optimizer': base.OPTIMIZER,
        'environment': provenance(torch, PROFILE), 'arms': arms, 'promotion': False,
        'limits': 'One seed, one fixed 1024-update schedule and measured CPU/runtime environments only. No generation, quality, functional evaluation, native compatibility, default recipe change or weight promotion.'}
    for variant in VARIANTS:
        repeats = []
        arms[variant] = {'repeats': repeats, 'same_host_identical': False}
        for repeat in range(2):
            value = train(backend.weights, batches, output / variant / str(repeat), variant)
            value['historical_checkpoint_equal'] = value['checkpoint_sha256'] == CONTROL_MODEL_SHA256
            repeats.append(value)
            (output / 'report.json').write_bytes(encoded(report))
        arms[variant]['same_host_identical'] = repeats[0] == repeats[1]
        (output / 'report.json').write_bytes(encoded(report))
        if not arms[variant]['same_host_identical']: raise RuntimeError('full-training same-host repeat differs')
    print(json.dumps({v: {'same_host_identical': a['same_host_identical'], 'checkpoint_sha256': a['repeats'][0]['checkpoint_sha256']} for v, a in arms.items()}))


def trace_difference(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b: return {'identical': False, 'first_different_step': i, 'left': a, 'right': b}
    return {'identical': len(left) == len(right), 'counts': [len(left), len(right)]}


def compare(root, *, directories=None, emit=True, allow_single=False):
    dirs = list(directories) if directories is not None else [root / f'portable-full-{i}' for i in range(1, 5)]
    if len(dirs) < (1 if allow_single else 2): raise ValueError('at least two completed training runs required')
    reports = [json.loads((p / 'report.json').read_bytes()) for p in dirs]
    traces = {}
    for i, (p, report) in enumerate(zip(dirs, reports)):
        if report['version'] != 1 or report['profile'] != PROFILE or report['steps'] != STEPS or report['promotion'] is not False or set(report['arms']) != set(VARIANTS):
            raise ValueError('full-training schema differs')
        for variant, arm in report['arms'].items():
            if not arm['same_host_identical'] or len(arm['repeats']) != 2 or arm['repeats'][0] != arm['repeats'][1]:
                raise ValueError('unreliable full-training repeat')
            checkpoints = [(p / variant / str(rep) / 'experimental.nppai').read_bytes() for rep in range(2)]
            if checkpoints[0] != checkpoints[1] or any(digest(raw) != arm['repeats'][0]['checkpoint_sha256'] for raw in checkpoints):
                raise ValueError('checkpoint repeat bytes differ')
            raw = [(p / variant / str(rep) / 'trace.jsonl').read_bytes() for rep in range(2)]
            if raw[0] != raw[1] or any(digest(r) != arm['repeats'][0]['trace_sha256'] for r in raw):
                raise ValueError('trace repeat bytes differ')
            trace = [json.loads(line) for line in raw[0].splitlines()]
            if len(trace) != STEPS + 1 or [r['step'] for r in trace] != list(range(STEPS + 1)):
                raise ValueError('incomplete training trace')
            if trace[0]['parameters'] != arm['repeats'][0]['initial_parameters'] or trace[-1]['parameters'] != arm['repeats'][0]['final_parameters']:
                raise ValueError('trace endpoint differs')
            traces[i, variant] = trace
    for report in reports[1:]:
        if any(report[k] != reports[0][k] for k in ('profile', 'source_model_id', 'gate', 'steps', 'schedule', 'training_sha256', 'optimizer')):
            raise ValueError('source or training exposure differs')
    for variant in VARIANTS:
        if len({r['arms'][variant]['repeats'][0]['initial_parameters'] for r in reports}) != 1: raise ValueError('initial weights differ')
        for index in range(len(reports)):
            if [r['inputs'] for r in traces[index, variant][1:]] != [r['inputs'] for r in traces[0, 'native'][1:]]:
                raise ValueError('per-step training inputs differ')
    pairs = []
    for a, b in itertools.combinations(range(len(reports)), 2):
        av, bv = [reports[i]['environment']['cpu'].get('vendor_id') for i in (a, b)]
        arms = {}
        for variant in VARIANTS:
            x, y = [reports[i]['arms'][variant]['repeats'][0] for i in (a, b)]
            arms[variant] = {'trace': trace_difference(traces[a, variant], traces[b, variant]),
                'checkpoint_identical': x['checkpoint_sha256'] == y['checkpoint_sha256'],
                'final_parameters_identical': x['final_parameters'] == y['final_parameters']}
        pairs.append({'replicas': [a + 1, b + 1], 'cross_vendor': bool(av and bv and av != bv), 'arms': arms})
    cross_vendor = sum(p['cross_vendor'] for p in pairs)
    result = {'reports': reports, 'pairs': pairs, 'coverage': {'cpu_models': [r['environment']['cpu'].get('model name') for r in reports],
        'vendors': [r['environment']['cpu'].get('vendor_id') for r in reports], 'cross_vendor_pairs': cross_vendor},
        'summary': {v: {'all_training_traces_and_checkpoints_identical': bool(pairs) and all(p['arms'][v]['trace']['identical'] and p['arms'][v]['checkpoint_identical'] and p['arms'][v]['final_parameters_identical'] for p in pairs)} for v in VARIANTS},
        'promotion': False, 'limits': reports[0]['limits']}
    (root / 'comparison.json').write_bytes(encoded(result))
    if emit: print(json.dumps({k: v for k, v in result.items() if k not in ('reports', 'pairs')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--compare', type=Path)
    for name in ('source', 'dataset', 'output'): parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.compare: compare(args.compare)
    else:
        if not all((args.source, args.dataset, args.output)): parser.error('source, dataset and output required')
        run(args.source, args.dataset, args.output)
