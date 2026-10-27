"""Audit the permanent raw CI record without executing candidate source."""
import hashlib
import json
from pathlib import Path
import unittest
import zipfile

from scripts.function_body_training import decision, groups, load_bank, score, summary

ROOT = Path(__file__).resolve().parents[1]


class BodyEvidenceTests(unittest.TestCase):
    def test_original_rejected_run_is_bound_to_raw_outputs_and_traces(self):
        folder = ROOT/'docs/evidence/function-body-v1'
        metadata = json.loads((folder/'manifest.json').read_bytes())
        raw_zip = (folder/'pilot.zip').read_bytes()
        self.assertEqual(hashlib.sha256(raw_zip).hexdigest(), metadata['archive_sha256'])
        with zipfile.ZipFile(folder/'pilot.zip') as archive:
            self.assertEqual(set(archive.namelist()), set(metadata['files']))
            self.assertFalse(any(n.endswith(('.nppai', '.pth')) for n in archive.namelist()))
            raw = {n: archive.read(n) for n in archive.namelist()}
        for name, content in raw.items():
            self.assertEqual(hashlib.sha256(content).hexdigest(), metadata['files'][name])
        report, training = json.loads(raw['report.json']), json.loads(raw['training.json'])
        self.assertEqual(hashlib.sha256(raw['training.json']).hexdigest(), report['training_sha256'])
        self.assertEqual(report['source_commit'], metadata['tested_merge_commit'])
        tasks = groups()
        for arm, data in report['arms'].items():
            for group, expected in tasks.items():
                rows = data['groups'][group]
                self.assertEqual([r['id'] for r in rows], [r['id'] for r in expected])
                for actual, task in zip(rows, expected):
                    self.assertEqual(actual['request'], task['request'])
                    self.assertEqual(actual['output_sha256'], hashlib.sha256(actual['output'].encode()).hexdigest())
                    self.assertEqual(actual['metrics'], score(actual['output'], task))
                self.assertEqual(data['summary'][group], summary(rows))
            self.assertEqual(data['summary'], metadata['summary'][arm])
            self.assertTrue(data['weights_unchanged'])
        for arm, info in training['arms'].items():
            trace = raw[arm+'/trace.json']
            self.assertEqual(hashlib.sha256(trace).hexdigest(), info['trace_sha256'])
            self.assertEqual([r['step'] for r in json.loads(trace)], list(range(1,1025)))
            self.assertEqual(info['model_sha256'], metadata['models'][arm])
        self.assertEqual(report['controls'], dict(positive=[0]*168, negative=[1]*24))
        manifest, _, _ = load_bank()
        self.assertEqual(report['decision'], decision(report, manifest))
        self.assertEqual(report['decision'], metadata['decision'])
        self.assertEqual(report['decision']['status'], 'rejected')
        self.assertEqual(report['arms']['diverse_bodies']['summary']['development']['functional_passed'], 0)
        self.assertEqual(report['arms']['diverse_bodies']['summary']['factorial']['functional_passed'], 18)
        self.assertFalse(report['promotion'])
