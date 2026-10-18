"""Reconstruct retained oracle records and metrics without model inference."""
from collections import Counter
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import alias_oracle_diagnostic as experiment


def audit(directory):
    p = experiment.pilot
    metadata = json.loads((directory/'retention.json').read_bytes())
    archive = directory/'raw-records.zip'
    if p.quality.digest(archive.read_bytes()) != metadata['records_sha256']:
        raise ValueError('retained oracle archive differs')
    raw = (directory/'report.json').read_bytes()
    if p.quality.digest(raw) != metadata['report_sha256']:
        raise ValueError('retained oracle report differs')
    report = json.loads(raw)
    if not all(report[k] is True for k in ('complete', 'state_unchanged', 'baseline_replay_equal')) or report['promotion'] is not False:
        raise ValueError('incomplete or promoted diagnostic')
    groups, fingerprints = p.prior.inputs(p.ROOT/'evaluation/instruction-v3', p.ROOT/'evaluation/gated-copy-probes-v1/tasks.json')
    if report['inputs'] != fingerprints or set(report['modes']) != set(experiment.MODES):
        raise ValueError('oracle inputs/modes differ')
    old = experiment.retained(p.ROOT/'docs/evidence/protected-sequential')['arms']['frozen_sequential']
    if report['checkpoint_sha256'] != old['identity']['checkpoint_sha256']:
        raise ValueError('oracle checkpoint differs')
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read('manifest.json'))
        if set(bundle.namelist()) != (set(manifest)-{'frozen.nppaicopy'})|{'manifest.json'}:
            raise ValueError('oracle retained file set differs')
        for name, digest in manifest.items():
            if name == 'frozen.nppaicopy':
                if digest != report['checkpoint_sha256']:
                    raise ValueError('excluded checkpoint identity differs')
            elif p.quality.digest(bundle.read(name)) != digest:
                raise ValueError('oracle record hash differs')
        if bundle.read('report.json') != raw:
            raise ValueError('oracle report copy differs')
        for mode in experiment.MODES:
            if set(report['modes'][mode]) != set(groups):
                raise ValueError('oracle group coverage differs')
            for group, rows in groups.items():
                cases = json.loads(bundle.read(mode+'-'+group.replace('/', '-')+'.json'))
                if len(cases) != len(rows):
                    raise ValueError('oracle case count differs')
                for i, (case, row) in enumerate(zip(cases, rows)):
                    if case['id'] != row['id'] or case['index'] != i or case['mode'] != mode or case['assisted'] != (mode != 'autonomous'):
                        raise ValueError('oracle case identity differs')
                    if bytes(case['output_bytes']).decode('utf-8', errors='replace') != case['output'] or case['metrics'] != p.base.score(row, case['output']):
                        raise ValueError('oracle bytes/metrics differ')
                    p.audit_events(case, row)
                    if mode == 'autonomous' and any(case[k] != old['after'][group][i][k] for k in ('output', 'output_bytes', 'events')):
                        raise ValueError('oracle autonomous replay differs')
                summary = {'total': len(cases), 'metrics': dict(Counter(k for c in cases for k, v in c['metrics'].items() if v))}
                if summary != report['modes'][mode][group]:
                    raise ValueError('oracle summary differs')
            functional = report['functional'][mode]
            if functional['total'] != 6 or len(functional['cases']) != 6 or functional['passed'] != sum(c['category'] == 'passed' for c in functional['cases']):
                raise ValueError('oracle functional counts differ')
    return report


if __name__ == '__main__':
    report = audit(experiment.pilot.ROOT/'docs/evidence/alias-oracle')
    print(json.dumps({'valid': True, 'complete': report['complete'], 'modes': list(report['modes'])}))
