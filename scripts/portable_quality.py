"""Frozen exported-checkpoint readout; no optimization or generated-code execution here."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'portable-cpu-v1'
SOURCE_RUN = 38959870764
ARTIFACT_ID = 12072377313
ARCHIVE_SHA256 = 'f7d5f4816e1a2d9fd4119b7512c1ba88f6fc6603ddf5237142549e33ab261745'
CHECKPOINTS = {
    'native': '75b01f90c184507f63ee8aba4a1a95d30c1c61f780e3fd595fccd71d56c4a743',
    'both64': '1bd024a162d9cbc79929520db6fb6a28d256be98a87a3bcc4b8149670c9e1da7',
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False) + '\n').encode()


def checkpoints(archive, output):
    """Accept only the predeclared original archive and byte-identical repeats."""
    if digest(archive.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError('pinned training archive differs')
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('training archive CRC failure')
        for arm, expected in CHECKPOINTS.items():
            raw = bundle.read(f'{arm}/0/experimental.nppai')
            if digest(raw) != expected or raw != bundle.read(f'{arm}/1/experimental.nppai'):
                raise ValueError('pinned checkpoint or repeat differs: ' + arm)
            paths[arm] = output / (arm + '.nppai')
            paths[arm].write_bytes(raw)
    return paths


def inputs(dataset, tasks):
    from scripts import aligned_name_copy as aligned
    import sandbox_validate
    groups = aligned.development(dataset)
    raw = tasks.read_bytes()
    suites = {}
    for task in json.loads(raw):
        _, suite = sandbox_validate.load_cases(Path(task['cases_file']))
        suites[task['task_id']] = suite
    return groups, {'development_sha256': digest(encoded(groups)),
                    'tasks_sha256': digest(raw), 'suites': suites}


def audit_groups(results, groups):
    from scripts import aligned_name_copy as aligned
    from scripts import resampled_name_training as base
    if results.keys() != groups.keys():
        raise ValueError('evaluation group set differs')
    for group, rows in groups.items():
        cases = results[group]
        if len(cases) != len(rows):
            raise ValueError('evaluation case count differs')
        for index, (record, case) in enumerate(zip(rows, cases)):
            if (case['id'] != record['id'] or case['index'] != index or
                    case['metrics'] != base.score(record, case['output'])):
                raise ValueError('raw evaluation score differs')
    return aligned.summaries(results)


def run(archive, dataset, tasks, output, image):
    import torch
    from scripts import aligned_name_copy as aligned
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    from scripts.train_gated_copy import functional_probes
    from scripts.training_repro import configure, provenance, state_digest
    protocol_raw = (ROOT / 'evaluation/portable-quality-v1.json').read_bytes()
    protocol = json.loads(protocol_raw)
    groups, fingerprints = inputs(dataset, tasks)
    if fingerprints != protocol['inputs']:
        raise ValueError('frozen quality inputs differ')
    configure(torch, PROFILE)
    paths = checkpoints(archive, output / 'checkpoints')
    report = {'protocol_sha256': digest(protocol_raw), 'inputs': fingerprints,
              'source_run': SOURCE_RUN, 'artifact_id': ARTIFACT_ID,
              'archive_sha256': ARCHIVE_SHA256, 'checkpoint_sha256': CHECKPOINTS,
              'inference': protocol['inference'], 'runtime': provenance(torch, PROFILE),
              'models': {}, 'promotion': False}
    # Native FP32 forward for BOTH exported checkpoints. No arithmetic dispatch,
    # assistant prefix, source adapter, optimizer, repair or reference assistance.
    models = {arm: torch_model(load_weights(path)) for arm, path in paths.items()}
    before = {arm: state_digest(model) for arm, model in models.items()}
    for arm, model in models.items():
        with torch.no_grad():
            cases = aligned.evaluate(model, groups)
        report['models'][arm] = {'groups': cases, 'summary': audit_groups(cases, groups)}
        (output / 'report.json').write_bytes(encoded(report))
        print(json.dumps({'arm': arm, 'summary': report['models'][arm]['summary']}), flush=True)
    functional = functional_probes(models, tasks, image)
    if (functional['manifest_sha256'] != fingerprints['tasks_sha256'] or
            any({c['task_id']: c['suite_sha256'] for c in r['cases']} != fingerprints['suites']
                for r in functional['models'].values())):
        raise ValueError('functional suites changed during evaluation')
    report['functional'] = functional
    report['parameters_unchanged'] = all(state_digest(model) == before[arm] for arm, model in models.items())
    if not report['parameters_unchanged']:
        raise ValueError('evaluation changed parameters')
    (output / 'report.json').write_bytes(encoded(report))
    print(json.dumps({'functional': {arm: r['passed'] for arm, r in functional['models'].items()},
                      'promotion': False}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'dataset', 'tasks', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    run(args.archive, args.dataset, args.tasks, args.output, args.image)
