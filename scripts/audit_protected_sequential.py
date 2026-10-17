"""Retain/reconstruct raw pilot replies, cursor events and every update trace."""
import argparse
import copy
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import protected_sequential as pilot


def retained(directory):
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    raw = (directory / 'report.json').read_bytes()
    if pilot.quality.digest(raw) != manifest['metadata_sha256']:
        raise ValueError('protected metadata hash differs')
    report = json.loads(raw)
    for item in manifest['chunks']:
        raw = (directory / item['file']).read_bytes()
        if pilot.quality.digest(raw) != item['sha256']:
            raise ValueError('protected evidence chunk hash differs')
        cases = report['arms'][item['arm']][item['stage']].setdefault(item['group'], [])
        if len(cases) != item['offset']:
            raise ValueError('protected chunk order differs')
        cases.extend(json.loads(raw))
    if pilot.quality.digest(pilot.quality.encoded(report)) != manifest['report_sha256']:
        raise ValueError('protected reconstructed report differs')
    return report


def retain(archive, directory, expected_sha256, run_id, artifact_id):
    raw = archive.read_bytes()
    if pilot.quality.digest(raw) != expected_sha256:
        raise ValueError('protected original artifact hash differs')
    dataset = pilot.ROOT / 'evaluation/instruction-v3'
    tasks = pilot.ROOT / 'evaluation/gated-copy-probes-v1/tasks.json'
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('protected archive CRC differs')
        report_raw = bundle.read('report.json')
        report = json.loads(report_raw)
        pilot.audit(report, dataset, tasks)
        if bundle.read('source/both64.nppai') and pilot.quality.digest(bundle.read('source/both64.nppai')) != pilot.quality.CHECKPOINTS['both64']:
            raise ValueError('protected source checkpoint differs')
        traces = {}
        for arm in pilot.ARMS:
            result = report['arms'][arm]
            trace = bundle.read(arm + '/0/trace.jsonl')
            weight = bundle.read(arm + '/0/experimental.nppaicopy')
            if (trace != bundle.read(arm + '/1/trace.jsonl') or
                    weight != bundle.read(arm + '/1/experimental.nppaicopy') or
                    pilot.quality.digest(weight) != result['identity']['checkpoint_sha256']):
                raise ValueError('protected artifact repeat/checkpoint differs')
            traces[arm] = pilot.audit_trace(trace, dataset, arm, result['identity'])
            # Verify the actual exported base, not only a reported unchanged flag.
            import io
            with zipfile.ZipFile(io.BytesIO(weight)) as model:
                if model.read('base.nppai') != bundle.read('source/both64.nppai'):
                    raise ValueError('protected exported base bytes differ from pinned initialization')
    directory.mkdir(parents=True, exist_ok=True)
    metadata = copy.deepcopy(report)
    manifest = {'run_id': run_id, 'artifact_id': artifact_id, 'archive_sha256': expected_sha256,
                'report_sha256': pilot.quality.digest(report_raw), 'chunks': [], 'traces': {},
                'verification': 'Original ZIP hash/CRC; both raw repeated bundles and traces; exported base bytes equal pinned source; all response metrics, retention, source cursor/exit decisions and training exposure/loss traces audited. Weights remain in 14-day Actions artifacts; hashes cannot reconstruct expired weights.'}
    for arm in pilot.ARMS:
        for stage in ('before', 'after'):
            metadata['arms'][arm][stage] = {}
            for group, cases in report['arms'][arm][stage].items():
                for offset in range(0, len(cases), 4):
                    name = f'{arm}-{stage}-{group.replace("/", "-")}-{offset//4:02d}.json'
                    raw = pilot.quality.encoded(cases[offset:offset+4])
                    (directory / name).write_bytes(raw)
                    manifest['chunks'].append({'arm':arm, 'stage':stage, 'group':group, 'offset':offset,
                                               'file':name, 'sha256':pilot.quality.digest(raw)})
        manifest['traces'][arm] = []
        for offset in range(0, len(traces[arm]), 64):
            name = f'{arm}-trace-{offset//64:02d}.json'
            raw = pilot.quality.encoded(traces[arm][offset:offset+64])
            (directory / name).write_bytes(raw)
            manifest['traces'][arm].append({'file':name, 'sha256':pilot.quality.digest(raw)})
    raw = pilot.quality.encoded(metadata)
    (directory / 'report.json').write_bytes(raw)
    manifest['metadata_sha256'] = pilot.quality.digest(raw)
    (directory / 'manifest.json').write_bytes(pilot.quality.encoded(manifest))
    reconstructed = retained(directory)
    pilot.audit(reconstructed, dataset, tasks)
    for arm in pilot.ARMS:
        pilot.audit_trace(pilot.curriculum.retained_trace(directory, arm), dataset, arm,
                          reconstructed['arms'][arm]['identity'])
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--directory', type=Path, default=pilot.ROOT/'docs/evidence/protected-sequential')
    parser.add_argument('--sha256')
    parser.add_argument('--run-id', type=int)
    parser.add_argument('--artifact-id', type=int)
    args = parser.parse_args()
    if args.archive:
        retain(args.archive, args.directory, args.sha256, args.run_id, args.artifact_id)
    else:
        report = retained(args.directory)
        pilot.audit(report, pilot.ROOT/'evaluation/instruction-v3', pilot.ROOT/'evaluation/gated-copy-probes-v1/tasks.json')
        for arm in pilot.ARMS:
            pilot.audit_trace(pilot.curriculum.retained_trace(args.directory, arm),
                              pilot.ROOT/'evaluation/instruction-v3', arm, report['arms'][arm]['identity'])
    print(json.dumps({'valid': True, 'arms': list(pilot.ARMS)}))
