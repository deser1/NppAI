"""Audit retained actual CI readout; never execute generated model source."""
import hashlib
import json
import unittest
import zipfile

from scripts.neutral_binding_training import (ROOT, SOURCE, SOURCE_FILES, decision, groups,
    load_bank, schedule, score, sha, summary)
from scripts.factorial_repair import dataset as factorial_data


class NeutralBindingEvidenceTests(unittest.TestCase):
    def test_raw_paired_result_controls_retention_and_identity(self):
        directory = ROOT/'docs/evidence/neutral-binding-v1'
        manifest = json.loads((directory/'manifest.json').read_bytes())
        archive_path = directory/'raw.zip'
        self.assertEqual(sha(archive_path),manifest['archive_sha256'])
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(set(archive.namelist()),set(manifest['files']))
            self.assertFalse(any(n.endswith(('.nppai','.pth')) for n in archive.namelist()))
            raw = {n:archive.read(n) for n in archive.namelist()}
        for name, contents in raw.items():
            self.assertEqual(hashlib.sha256(contents).hexdigest(),manifest['files'][name])
        report, training = json.loads(raw['report.json']),json.loads(raw['training.json'])
        bank_manifest, bank, dev = load_bank()
        self.assertEqual(report['training_sha256'],manifest['files']['training.json'])
        self.assertEqual(report['source_commit'],manifest['tested_merge_commit'])
        self.assertEqual(training['source_commit'],manifest['tested_merge_commit'])
        self.assertEqual(training['source_sha256'],SOURCE)
        self.assertEqual(training['bank_manifest_sha256'],sha(ROOT/'evaluation/neutral-binding-v1.json'))
        self.assertEqual(training['source_sha256_files'],{f:sha(ROOT/f) for f in SOURCE_FILES})
        self.assertEqual(training['protocol'],bank_manifest['protocol'])
        self.assertTrue(training['initialization_unchanged'])
        self.assertEqual(training['schedule_sha256'],manifest['files']['schedule.json'])
        expected = groups()
        self.assertEqual(json.loads(raw['schedule.json']),schedule(bank_manifest,len(expected['original']),len(factorial_data()[0]),len(bank)))
        self.assertEqual(training['original_replay_sha256'],sha(ROOT/'evaluation/staged-protocol-v1/train.json'))
        self.assertEqual(training['factorial_training_sha256'],hashlib.sha256(json.dumps(factorial_data()[0],sort_keys=True).encode()).hexdigest())
        self.assertEqual(set(training['arms']),{'fixed_position','position_jitter'})
        for arm, item in training['arms'].items():
            self.assertEqual(item['trace_sha256'],manifest['files'][arm+'/trace.json'])
            trace=json.loads(raw[arm+'/trace.json'])
            self.assertEqual([r['step'] for r in trace],list(range(1,1025)))
            self.assertTrue(all(__import__('math').isfinite(r['loss']) for r in trace))
        self.assertRegex(report['validator_image'],r'^sha256:[0-9a-f]{64}$')
        self.assertEqual(report['controls'],dict(positive=[0]*(len(bank)+len(dev)),negative=[1]*len(dev)))
        self.assertTrue(report['complete'])
        self.assertFalse(report['promotion'])
        self.assertEqual(set(report['arms']),{'source','fixed_position','position_jitter'})
        for arm, data in report['arms'].items():
            self.assertTrue(data['weights_unchanged'])
            self.assertEqual(data['model_sha256'],SOURCE if arm=='source' else training['arms'][arm]['model_sha256'])
            self.assertEqual(data['model_sha256'],manifest['models'][arm])
            self.assertEqual(set(data['groups']),set(expected))
            for group, tasks in expected.items():
                rows = data['groups'][group]
                self.assertEqual([r['id'] for r in rows],[t['id'] for t in tasks])
                for result, task in zip(rows,tasks):
                    self.assertEqual(result['request'],task['request'])
                    self.assertEqual(result['output_sha256'],hashlib.sha256(result['output'].encode()).hexdigest())
                    self.assertEqual(result['metrics'],score(result['output'],task))
                    if group in ('development','factorial'):
                        self.assertIs(type(result['functional_status']),int)
                        self.assertIn(result['functional_status'],(0,1))
                        if result['error']: self.assertEqual(result['functional_status'],1)
                    else: self.assertIsNone(result['functional_status'])
                self.assertEqual(data['summary'][group],summary(rows))
            rows=data['groups']['development']
            self.assertEqual(data['cells'],{c:summary([r for r,t in zip(rows,dev) if t['cell']==c]) for c in {t['cell'] for t in dev}})
            self.assertEqual(data['cells'],manifest['cells'][arm])
            self.assertEqual(data['summary'],manifest['summary'][arm])
        self.assertEqual(report['decision'],decision(report,bank_manifest))
        self.assertEqual(report['decision'],manifest['decision'])
        self.assertEqual(report['decision']['status'],'rejected')
        for data in report['arms'].values():
            self.assertEqual(data['summary']['original']['exact'],96)
            self.assertEqual(data['summary']['factorial']['functional_passed'],20)
            self.assertEqual(sum(row['functional_status']==0 for row, task in zip(data['groups']['development'],dev) if task['primary']),0)
        for cell in ('name0-args0-shift1','name1-args0-shift1'):
            self.assertEqual(report['arms']['fixed_position']['cells'][cell]['functional_passed'],2)
            self.assertEqual(report['arms']['position_jitter']['cells'][cell]['functional_passed'],12)
        self.assertEqual(report['arms']['position_jitter']['cells']['name0-args0-shift0']['functional_passed'],11)
