"""Retain and reconstruct paired start-only training evidence without inference."""
import argparse
import io
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import start_selection_training as experiment
from scripts.audit_protected_sequential import retained

p = experiment.pilot
ROOT = experiment.ROOT


def protected_entries(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        base = bundle.read('base.nppai')
        with zipfile.ZipFile(io.BytesIO(bundle.read('head.npz'))) as head:
            return {'base': base, 'exit_weight': head.read('exit_weight.npy'),
                    'exit_bias': head.read('exit_bias.npy')}


def retain(archive, directory, expected_sha256, run_id, artifact_id):
    raw = archive.read_bytes()
    if p.quality.digest(raw) != expected_sha256:
        raise ValueError('original start archive differs')
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('start archive CRC differs')
        manifest = json.loads(bundle.read('manifest.json'))
        for name, digest in manifest.items():
            if p.quality.digest(bundle.read(name)) != digest:
                raise ValueError('original start file hash differs')
        protected = protected_entries(bundle.read('frozen.nppaicopy'))
        report = json.loads(bundle.read('report.json'))
        for arm in experiment.ARMS:
            first = bundle.read(arm+'/0/experimental.nppaicopy')
            if first != bundle.read(arm+'/1/experimental.nppaicopy'):
                raise ValueError('repeated exported start weights differ')
            if protected_entries(first) != protected:
                raise ValueError('exported base or exit bytes changed')
            if bundle.read(arm+'/0/trace.jsonl') != bundle.read(arm+'/1/trace.jsonl'):
                raise ValueError('repeated start trace differs')
            if p.quality.digest(first) != report['arms'][arm]['identity']['checkpoint_sha256']:
                raise ValueError('exported start checkpoint hash differs')
        with zipfile.ZipFile(directory/'raw-records.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
            for name in bundle.namelist():
                if name.endswith(('.json', '.jsonl')):
                    target.writestr(name, bundle.read(name))
        metadata = {'run_id': run_id, 'artifact_id': artifact_id, 'archive_sha256': expected_sha256,
                    'records_sha256': p.quality.digest((directory/'raw-records.zip').read_bytes()),
                    'protected_export_bytes_equal': True,
                    'excluded_checkpoints': {name: digest for name, digest in manifest.items() if name.endswith('.nppaicopy')},
                    'limits': 'Every JSON/JSONL member retained byte-for-byte. Exported base/exit bytes, both repeated bundles and traces verified from original ZIP. Weight bytes remain in expiring Actions artifacts; hashes alone cannot reconstruct them.'}
        (directory/'retention.json').write_bytes(p.quality.encoded(metadata))
        (directory/'.gitattributes').write_text('raw-records.zip -text\n')
    audit(directory)
    return metadata


def audit(directory):
    metadata = json.loads((directory/'retention.json').read_bytes())
    raw = (directory/'raw-records.zip').read_bytes()
    if p.quality.digest(raw) != metadata['records_sha256']:
        raise ValueError('retained start records differ')
    dataset, tasks = ROOT/'evaluation/instruction-v3', ROOT/'evaluation/gated-copy-probes-v1/tasks.json'
    groups, fingerprints = p.prior.inputs(dataset, tasks)
    initial = retained(ROOT/'docs/evidence/protected-sequential')['arms']['frozen_sequential']
    schedule = list(experiment.schedule(dataset))
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        manifest = json.loads(bundle.read('manifest.json'))
        if set(bundle.namelist()) != (set(manifest)-set(metadata['excluded_checkpoints']))|{'manifest.json'}:
            raise ValueError('retained start member set differs')
        for name, digest in manifest.items():
            if name in metadata['excluded_checkpoints']:
                if metadata['excluded_checkpoints'][name] != digest:
                    raise ValueError('excluded checkpoint identity differs')
            elif p.quality.digest(bundle.read(name)) != digest:
                raise ValueError('start member hash differs')
        report = json.loads(bundle.read('report.json'))
        if (report['spec'] != experiment.SPEC or report['protocol'] != experiment.protocol(dataset) or
                report['inputs'] != fingerprints or report['initial_archive_sha256'] != experiment.SPEC['initial_archive_sha256'] or
                report['initial_checkpoint_sha256'] != initial['identity']['checkpoint_sha256'] or
                manifest['frozen.nppaicopy'] != initial['identity']['checkpoint_sha256'] or
                report.get('complete') is not True or report['baseline_replay_equal'] is not True or
                report['promotion'] is not False or set(report['arms']) != set(experiment.ARMS) or
                metadata['protected_export_bytes_equal'] is not True):
            raise ValueError('start provenance/completion differs')
        starts, protected = set(), set()
        for arm, result in report['arms'].items():
            if result['same_host_repeat_equal'] is not True or result['parameters_unchanged_by_evaluation'] is not True or result['identity']['roundtrip_equal'] is not True:
                raise ValueError('start repeat/invariance gate differs')
            traces = [bundle.read(arm+'/'+str(rep)+'/trace.jsonl') for rep in (0, 1)]
            if traces[0] != traces[1] or p.quality.digest(traces[0]) != result['identity']['trace_sha256']:
                raise ValueError('start retained repeat differs')
            rows = [json.loads(line) for line in traces[0].splitlines()]
            if len(rows) != experiment.STEPS+1 or [r['step'] for r in rows] != list(range(experiment.STEPS+1)):
                raise ValueError('start trace steps differ')
            starts.add(rows[0]['parameters'])
            protected.add((rows[0]['base'], rows[0]['exit']))
            if rows[0]['base'] != initial['identity']['base_state']:
                raise ValueError('start frozen base identity differs')
            for row, (records, flags) in zip(rows[1:], schedule):
                expected_batch = p.quality.digest(p.quality.encoded({'records': records, 'original': flags}))
                losses = row['losses']
                total = float.fromhex(row['loss_hex'])
                expected = losses['start_ce'] + (losses['hard_negative_margin'] if arm == 'start_margin' else 0)
                if (row['batch_sha256'] != expected_batch or row['base'] != rows[0]['base'] or row['exit'] != rows[0]['exit'] or
                        not math.isfinite(total) or not math.isclose(total, expected, rel_tol=1e-6, abs_tol=1e-6) or
                        not 0 <= losses['start_correct'] <= losses['examples'] or losses['examples'] != len(records)):
                    raise ValueError('start schedule/loss/frozen trace differs')
            identity = result['identity']
            if rows[-1]['parameters'] != identity['parameters'] or identity['protected'] != {'base': rows[0]['base'], 'exit': rows[0]['exit']}:
                raise ValueError('start final parameter identity differs')
            for rep in (0, 1):
                if manifest[arm+'/'+str(rep)+'/experimental.nppaicopy'] != identity['checkpoint_sha256']:
                    raise ValueError('start retained checkpoint repeat differs')
            if result['after_summary'] != p.prior.audit_views(result['after'], groups):
                raise ValueError('start score summary differs')
            for group, cases in result['after'].items():
                for case, original in zip(cases, groups[group]):
                    p.audit_events(case, original)
            if result['retention'] != p.base.retention(initial['after'], result['after']):
                raise ValueError('start exact retention differs')
            functional = result['functional']
            measurement = functional['models']['copy']
            if functional['manifest_sha256'] != fingerprints['tasks_sha256'] or measurement['tasks'] != 6 or len(measurement['cases']) != 6 or measurement['passed'] != sum(c['category'] == 'passed' for c in measurement['cases']):
                raise ValueError('start functional totals differ')
            for case, task in zip(measurement['cases'], json.loads(tasks.read_bytes())):
                if any(case[k] != v for k, v in task.items()) or case['suite_sha256'] != fingerprints['suites'][task['task_id']]:
                    raise ValueError('start functional suite differs')
        if len(starts) != 1 or len(protected) != 1:
            raise ValueError('start arms use different initialization/protected parameters')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'docs/evidence/start-selection')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--run-id', type=int)
    parser.add_argument('--artifact-id', type=int)
    args = parser.parse_args()
    if args.archive:
        retain(args.archive, args.directory, args.sha256, args.run_id, args.artifact_id)
    report = audit(args.directory)
    print(json.dumps({'valid': True, 'complete': report['complete'], 'arms': list(report['arms'])}))
