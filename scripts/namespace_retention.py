"""Fixed 2x2 training-only namespace/retention pilot; no production promotion."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import copy_curriculum as previous
from scripts import resampled_name_training as base

quality = previous.quality
ROOT = Path(__file__).resolve().parents[1]
STEPS, WARMUP = 512, 128
ARMS = {'fixed': (False, False), 'diverse': (True, False),
        'retained': (False, True), 'diverse_retained': (True, True)}
PREFIXES = ('cp_', 'token_', 'word_', 'id_')
SPEC = {'version': 1, 'arms': {k: list(v) for k, v in ARMS.items()},
        'factors': ['diverse_namespace', 'original_training_teacher_kl'],
        'initial_checkpoint_sha256': quality.CHECKPOINTS['both64'],
        'head_seed': previous.HEAD_SEED, 'steps': STEPS, 'warmup': WARMUP,
        'prefixes': list(PREFIXES), 'distillation_coefficient': 0.5,
        'optimizer': previous.SPEC['optimizer'], 'profile': quality.PROFILE,
        'arithmetic': 'native FP32 gated model; no cross-CPU certificate',
        'warmup_targets': 'originals remain full code; renamed examples use def name( plus EOS',
        'post_warmup_targets': 'all full code',
        'auxiliary': previous.SPEC['curriculum_auxiliary'],
        'decoding': previous.SPEC['decoding'],
        'limits': 'New matched 512-update pilot, not reproduction of the prior 1024-update control. '
                  'Original full targets remain active even in warmup in all arms. '
                  'Teacher is the same frozen initialization, original TRAIN positions only. '
                  'Equal updates/examples, not equal tokens or compute. One seed/coefficient; no search. '
                  'Exposed development/functional regressions are not fresh holdout. No promotion.'}


def schedule(dataset, arm):
    if arm not in ARMS:
        raise ValueError('unknown namespace arm')
    diverse, _ = ARMS[arm]
    for step, batch in enumerate(base.schedule(base.training_data(dataset), 'resampled', STEPS)):
        records, original = [], []
        alias_slot = 0
        for pool, record in batch:
            row = dict(record)
            original.append(pool == 'original')
            if pool == 'renamed':
                if diverse:
                    row = base.rename(row, PREFIXES[(step * 2 + alias_slot) % len(PREFIXES)] + row['function'][3:])
                alias_slot += 1
                if step < WARMUP:
                    row['answer'] = 'def ' + row['function'] + '('
            base.frame(row)
            records.append(row)
        yield records, original


def schedule_evidence(dataset, arm):
    digest, prompts = hashlib.sha256(), hashlib.sha256()
    targets = originals = aliases = 0
    for records, original in schedule(dataset, arm):
        digest.update(quality.encoded({'records': records, 'original': original}))
        prompts.update(quality.encoded([r['prompt'] for r in records]))
        targets += sum(len(r['answer'].encode()) + 1 for r in records)
        originals += sum(original)
        aliases += len(original) - sum(original)
    return {'sha256': digest.hexdigest(), 'prompts_sha256': prompts.hexdigest(),
            'response_targets': targets, 'originals': originals, 'aliases': aliases, 'updates': STEPS}


def namespace_probe(dataset):
    """New suffixes on TRAIN task families; these rows never enter optimization."""
    rows = random.Random(4242).sample(base.training_data(dataset), 16)
    groups = {}
    for prefix in ('nc_', 'id_', 'q_', 'rename_'):
        groups['train-family/' + prefix] = [base.rename(r, prefix + base.digest(
            ('namespace-probe-v1/' + str(i) + '/' + r['id']).encode())[:8]) for i, r in enumerate(rows)]
    used = {r['function'] for arm in ARMS for records, _ in schedule(dataset, arm) for r in records}
    if used & {r['function'] for rows in groups.values() for r in rows}:
        raise ValueError('probe suffix appeared in optimization')
    return groups


def inputs(dataset, tasks):
    groups, fingerprints = quality.inputs(dataset, tasks)
    probes = namespace_probe(dataset)
    return {**groups, **probes}, {**fingerprints, 'namespace_probe_sha256': quality.digest(quality.encoded(probes))}


def distillation(student, teacher, x, y, records, original):
    import torch
    from torch.nn import functional as F
    from scripts.gated_copy_model import source_mask
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('teacher must be frozen and in eval mode')
    indices = [i for i, flag in enumerate(original) if flag]
    if len(indices) != 2:
        raise ValueError('exactly two original training examples required')
    tokens, labels = x[indices], y[indices]
    mask = torch.stack([source_mask(tokens[i:i+1], records[j]['prompt'])[0] for i, j in enumerate(indices)])
    with torch.no_grad():
        target = teacher(tokens, mask)
    prediction = student(tokens, mask)
    return F.kl_div(prediction, target, log_target=True, reduction='none').sum(-1)[labels != -100].mean()


def evaluate(model, groups):
    from scripts.gated_copy_model import generate
    return {group: [{'id': row['id'], 'index': i, 'output': output, 'metrics': base.score(row, output),
                      'name_prefix': output.startswith('def ' + row['function'] + '(')}
                     for i, row in enumerate(rows) for output in [generate(model, row['prompt'])]]
            for group, rows in groups.items()}


def prefix_summary(results):
    return {group: {'total': len(cases), 'name_prefix': sum(c['name_prefix'] for c in cases)}
            for group, cases in results.items()}


def train(weights, dataset, output, arm):
    import torch
    from scripts.training_repro import configure, state_digest
    from scripts.gated_copy_model import export_bundle, load_bundle
    configure(torch, quality.PROFILE)
    model = previous.initial(weights)
    teacher = previous.initial(weights).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_state = state_digest(teacher)
    for p in model.base.parameters():
        p.requires_grad_(False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    output.mkdir(parents=True, exist_ok=True)
    trace = output / 'trace.jsonl'
    trace.write_bytes(quality.encoded({'step': 0, 'parameters': state_digest(model), 'teacher': teacher_state}))
    for step, (records, original) in enumerate(schedule(dataset, arm)):
        if step == WARMUP:
            for p in model.base.parameters():
                p.requires_grad_(True)
        samples = [base.sample(r) for r in records]
        x = torch.nn.utils.rnn.pad_sequence([s[0] for s in samples], batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence([s[1] for s in samples], batch_first=True, padding_value=-100)
        loss, parts = previous.losses(model, records, x, y, True)
        kl = distillation(model, teacher, x, y, records, original) if ARMS[arm][1] else torch.tensor(0.)
        loss = loss + 0.5 * kl
        parts['teacher_kl'] = float(kl.detach())
        if not math.isfinite(float(loss.detach())):
            raise ValueError('nonfinite pilot loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('nonfinite pilot parameter')
        with trace.open('ab') as stream:
            stream.write(quality.encoded({'step': step + 1, 'parameters': state_digest(model),
                'batch_sha256': quality.digest(quality.encoded({'records': records, 'original': original})),
                'loss_hex': float(loss.detach()).hex(), 'losses': parts}))
        if step % 128 == 0 or step == STEPS - 1:
            print(json.dumps({'arm': arm, 'step': step + 1, **parts}), flush=True)
    if state_digest(teacher) != teacher_state or any(p.grad is not None for p in teacher.parameters()):
        raise ValueError('teacher changed during training')
    model.eval()
    path = output / 'experimental.nppaicopy'
    export_bundle(model, path, 'sha256:' + quality.CHECKPOINTS['both64'])
    loaded, _ = load_bundle(path)
    if state_digest(model) != state_digest(loaded):
        raise ValueError('pilot roundtrip differs')
    return loaded, {'checkpoint_sha256': quality.digest(path.read_bytes()),
                    'trace_sha256': quality.digest(trace.read_bytes()), 'teacher_unchanged': True, 'roundtrip_equal': True}


def run(archive, dataset, tasks, output, image):
    from scripts.training_repro import configure, provenance, state_digest
    from scripts.check_dense_runtime_parity import load_weights
    import torch
    protocol_raw = (ROOT / 'evaluation/namespace-retention-v1.json').read_bytes()
    protocol = json.loads(protocol_raw)
    groups, fingerprints = inputs(dataset, tasks)
    if protocol != {'spec': SPEC, 'inputs': fingerprints,
                    'schedules': {a: schedule_evidence(dataset, a) for a in ARMS}}:
        raise ValueError('frozen namespace pilot differs')
    weights = load_weights(quality.checkpoints(archive, output / 'source')['both64'])
    configure(torch, quality.PROFILE)
    before = evaluate(previous.initial(weights).eval(), groups)
    report = {'protocol_sha256': quality.digest(protocol_raw), 'spec': SPEC, 'inputs': fingerprints,
              'before': before, 'before_summary': quality.audit_groups(before, groups),
              'before_prefix_summary': prefix_summary(before), 'runtime': provenance(torch, quality.PROFILE),
              'arms': {}, 'promotion': False}
    (output / 'report.json').write_bytes(quality.encoded(report))
    for arm in ARMS:
        model, identity = train(weights, dataset, output / arm / '0', arm)
        _, repeat = train(weights, dataset, output / arm / '1', arm)
        for name in ('trace.jsonl', 'experimental.nppaicopy'):
            if (output / arm / '0' / name).read_bytes() != (output / arm / '1' / name).read_bytes():
                raise ValueError('pilot same-host repeat differs')
        if identity != repeat:
            raise ValueError('pilot repeated identity differs')
        state = state_digest(model)
        after = evaluate(model, groups)
        result = {'identity': identity, 'same_host_repeat_equal': True, 'after': after,
                  'summary': quality.audit_groups(after, groups), 'prefix_summary': prefix_summary(after),
                  'retention': base.retention(before, after), 'functional': previous.functional(model, tasks, image),
                  'parameters_unchanged_by_evaluation': state == state_digest(model)}
        if not result['parameters_unchanged_by_evaluation']:
            raise ValueError('pilot evaluation mutated parameters')
        report['arms'][arm] = result
        (output / 'report.json').write_bytes(quality.encoded(report))
        print(json.dumps({'arm': arm, 'summary': result['summary'], 'prefix_summary': result['prefix_summary'],
                          'functional_passed': result['functional']['models']['copy']['passed']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'dataset', 'tasks', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    run(args.archive, args.dataset, args.tasks, args.output, args.image)
