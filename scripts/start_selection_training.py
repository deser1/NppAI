"""Paired train-only start selection continuation from the frozen PR #198 pilot."""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import protected_sequential as pilot
from scripts import alias_oracle_diagnostic as frozen
from scripts.audit_protected_sequential import retained

ROOT = pilot.ROOT
ARMS = ('start_ce', 'start_margin')
STEPS = 256
SPEC = {'version': 1, 'steps': STEPS, 'arms': list(ARMS),
        'initial_archive_sha256': '35d8a994d589fbe7776013e20a8d38a3c7e5281195e2f1fb6c4387f1f1d4e074',
        'trainable': ['copy_q.weight', 'copy_k.weight'],
        'schedule': 'first 256 existing diverse namespace TRAIN batches, two original and two resampled aliases per update',
        'control_loss': 'mean source-start cross entropy over prompt-only byte positions',
        'treatment_loss': 'control + mean relu(1 + highest non-gold source score - gold score)',
        'margin': 1.0, 'margin_coefficient': 1.0,
        'optimizer': {'name': 'AdamW', 'lr': .001, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01, 'clip': 1.0},
        'profile': pilot.quality.PROFILE,
        'decoding': 'unchanged sequential decoder; empty response prefix; no oracle, parser or rewriting at inference',
        'gates': 'exact train original hits retained, autonomous alias/full-code improvement over matched control, separate functional gates; no promotion in this pilot',
        'limits': 'One fixed seed/recipe, equal updates/examples not compute. Existing exposed development and six Docker probes only. Both arms are new start-only continuations, not replication of the earlier full objective.'}


def schedule(dataset):
    yield from itertools.islice(pilot.schedule(dataset), STEPS)


def protocol(dataset):
    records = [{'records': batch, 'original': flags} for batch, flags in schedule(dataset)]
    return {'spec': SPEC, 'schedule_sha256': pilot.quality.digest(pilot.quality.encoded(records)),
            'batches': len(records), 'originals': sum(sum(r['original']) for r in records),
            'aliases': sum(len(r['original'])-sum(r['original']) for r in records)}


def pointer_objective(scores, eligible, gold, arm):
    """Loss over actual source logits; ineligible and gold negatives are excluded."""
    import torch
    from torch.nn import functional as F
    if arm not in ARMS or scores.ndim != 2 or eligible.shape != scores.shape or eligible.dtype != torch.bool:
        raise ValueError('invalid start loss inputs')
    if gold.shape != (scores.shape[0],) or gold.dtype != torch.long:
        raise ValueError('invalid start labels')
    if not bool(((gold >= 0) & (gold < scores.shape[1])).all()):
        raise ValueError('start label outside sequence')
    rows = torch.arange(scores.shape[0], device=scores.device)
    if not bool(eligible[rows, gold].all()) or not bool((eligible.sum(-1) >= 2).all()):
        raise ValueError('start requires gold plus a competing prompt byte')
    masked = scores.masked_fill(~eligible, float('-inf'))
    ce = F.cross_entropy(masked, gold)
    competitors = eligible.clone()
    competitors[rows, gold] = False
    negative = scores.masked_fill(~competitors, float('-inf')).argmax(-1).detach()
    margin = F.relu(SPEC['margin'] + scores[rows, negative] - scores[rows, gold]).mean()
    loss = ce + (SPEC['margin_coefficient'] if arm == 'start_margin' else 0.0) * margin
    return loss, {'start_ce': float(ce.detach()), 'hard_negative_margin': float(margin.detach()),
                  'start_correct': int((masked.argmax(-1) == gold).sum()), 'examples': scores.shape[0]}


def losses(model, records, x, arm):
    import torch
    from scripts.aligned_name_copy import alignment
    from scripts.gated_copy_model import source_mask
    hidden = model.hidden(x)
    mappings = [alignment(r)[0] for r in records]
    rows = torch.arange(len(records))
    query = torch.tensor([q for q, _ in mappings])
    source = torch.tensor([s for _, s in mappings])
    mask = torch.stack([source_mask(x[i:i+1], r['prompt'])[0] for i, r in enumerate(records)])
    mask &= torch.arange(x.shape[1])[None, :] < query[:, None]
    mask &= x != 0
    scores = (model.copy_q(hidden[rows, query])[:, None, :] @ model.copy_k(hidden).transpose(-2, -1)).squeeze(1) / model.base.dim**.5
    return pointer_objective(scores, mask, source, arm)


def train(source, dataset, output, arm):
    import torch
    from scripts.sequential_copy_model import load_bundle, export_bundle
    from scripts.training_repro import configure, state_digest
    configure(torch, pilot.quality.PROFILE)
    model, manifest = load_bundle(source)
    model.requires_grad_(False)
    model.copy_q.requires_grad_(True)
    model.copy_k.requires_grad_(True)
    model.train()
    protected = {'base': state_digest(model.base), 'exit': state_digest(model.exit)}
    opt = SPEC['optimizer']
    optimizer = torch.optim.AdamW([*model.copy_q.parameters(), *model.copy_k.parameters()],
        lr=opt['lr'], betas=tuple(opt['betas']), eps=opt['eps'], weight_decay=opt['weight_decay'])
    output.mkdir(parents=True, exist_ok=True)
    trace = [{'step': 0, 'parameters': state_digest(model), **protected}]
    for step, (records, flags) in enumerate(schedule(dataset), 1):
        samples = [pilot.base.sample(r)[0] for r in records]
        x = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True)
        loss, parts = losses(model, records, x, arm)
        if not math.isfinite(float(loss.detach())):
            raise ValueError('nonfinite start loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*model.copy_q.parameters(), *model.copy_k.parameters()], opt['clip'], error_if_nonfinite=True)
        optimizer.step()
        if any(p.grad is not None or p.requires_grad for n, p in model.named_parameters() if n not in SPEC['trainable']):
            raise ValueError('protected parameters received gradients')
        if protected != {'base': state_digest(model.base), 'exit': state_digest(model.exit)}:
            raise ValueError('base or exit changed')
        trace.append({'step': step, 'parameters': state_digest(model), **protected,
            'batch_sha256': pilot.quality.digest(pilot.quality.encoded({'records': records, 'original': flags})),
            'loss_hex': float(loss.detach()).hex(), 'losses': parts})
        if step % 64 == 0:
            print(json.dumps({'arm': arm, 'step': step, **parts}), flush=True)
    trace_raw = b''.join(pilot.quality.encoded(r) for r in trace)
    (output/'trace.jsonl').write_bytes(trace_raw)
    path = output/'experimental.nppaicopy'
    export_bundle(model.eval(), path, manifest['source_model_id'])
    loaded, _ = load_bundle(path)
    if state_digest(model) != state_digest(loaded):
        raise ValueError('start model roundtrip differs')
    return loaded, {'checkpoint_sha256': pilot.quality.digest(path.read_bytes()),
                    'trace_sha256': pilot.quality.digest(trace_raw), 'parameters': state_digest(model),
                    'protected': protected, 'roundtrip_equal': True}


def run(archive, output, image):
    import torch
    from scripts.training_repro import configure, provenance, state_digest
    configure(torch, pilot.quality.PROFILE)
    dataset, tasks = ROOT/'evaluation/instruction-v3', ROOT/'evaluation/gated-copy-probes-v1/tasks.json'
    expected_protocol = json.loads((ROOT/'evaluation/start-selection-v1.json').read_bytes())
    if protocol(dataset) != expected_protocol:
        raise ValueError('start selection protocol differs')
    evidence = ROOT/'docs/evidence/protected-sequential'
    old = retained(evidence)
    pilot.audit(old, dataset, tasks)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((evidence/'manifest.json').read_bytes())
    source_model = frozen.load_checkpoint(archive, output, manifest, old)
    groups, fingerprints = pilot.prior.inputs(dataset, tasks)
    baseline = pilot.evaluate(source_model, groups, 'frozen_sequential')
    if baseline != old['arms']['frozen_sequential']['after']:
        raise ValueError('initial autonomous historical replay differs')
    report = {'spec': SPEC, 'protocol': expected_protocol, 'inputs': fingerprints,
              'initial_archive_sha256': manifest['archive_sha256'],
              'initial_checkpoint_sha256': old['arms']['frozen_sequential']['identity']['checkpoint_sha256'], 'runtime': provenance(torch, pilot.quality.PROFILE),
              'source_sha256': pilot.quality.digest((ROOT/'scripts/start_selection_training.py').read_bytes()),
              'baseline_replay_equal': True, 'promotion': False, 'arms': {}}
    for arm in ARMS:
        model, identity = train(output/'frozen.nppaicopy', dataset, output/arm/'0', arm)
        repeated, other_identity = train(output/'frozen.nppaicopy', dataset, output/arm/'1', arm)
        if identity != other_identity or (output/arm/'0/experimental.nppaicopy').read_bytes() != (output/arm/'1/experimental.nppaicopy').read_bytes() or (output/arm/'0/trace.jsonl').read_bytes() != (output/arm/'1/trace.jsonl').read_bytes():
            raise ValueError('start repeated training differs')
        state = state_digest(model)
        after = pilot.evaluate(model, groups, 'frozen_sequential')
        if after != pilot.evaluate(repeated, groups, 'frozen_sequential'):
            raise ValueError('start repeated evaluation differs')
        functional = pilot.functional(model, tasks, image, 'frozen_sequential')
        if state_digest(model) != state:
            raise ValueError('start evaluation changed weights')
        report['arms'][arm] = {'identity': identity, 'same_host_repeat_equal': True, 'after': after,
            'after_summary': pilot.prior.audit_views(after, groups),
            'retention': pilot.base.retention(baseline, after), 'functional': functional,
            'parameters_unchanged_by_evaluation': True}
        print(json.dumps({'arm': arm, 'summary': report['arms'][arm]['after_summary'],
                          'retention': report['arms'][arm]['retention']['train/original'],
                          'functional_passed': functional['models']['copy']['passed']}), flush=True)
        (output/'report.json').write_bytes(pilot.quality.encoded(report))
    report['complete'] = True
    (output/'report.json').write_bytes(pilot.quality.encoded(report))
    hashes = {str(p.relative_to(output)): pilot.quality.digest(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'manifest.json').write_bytes(pilot.quality.encoded(hashes))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    run(args.archive, args.output, args.image)
