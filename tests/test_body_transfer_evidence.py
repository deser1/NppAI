"""Check the exact retained CI grid; no model source executes in this test."""
import hashlib
import json
from pathlib import Path
import unittest
import zipfile

from scripts.diagnose_body_transfer import probes, probe_sha
from scripts.function_body_training import decision, load_bank, score, summary

ROOT = Path(__file__).resolve().parents[1]


class TransferEvidenceTests(unittest.TestCase):
    def test_complete_ci_grid_retains_failures_and_source_identity(self):
        directory = ROOT/'docs/evidence/function-body-v1'
        metadata = json.loads((directory/'transfer-manifest.json').read_bytes())
        archive_path = directory/'transfer.zip'
        self.assertEqual(hashlib.sha256(archive_path.read_bytes()).hexdigest(), metadata['archive_sha256'])
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(set(archive.namelist()), set(metadata['files']))
            self.assertFalse(any(n.endswith(('.nppai','.pth')) for n in archive.namelist()))
            raw = {n: archive.read(n) for n in archive.namelist()}
        for name, value in raw.items():
            self.assertEqual(hashlib.sha256(value).hexdigest(), metadata['files'][name])
        grid = json.loads(raw['transfer-diagnostic.json'])
        base = json.loads(raw['report.json'])
        training = json.loads(raw['training.json'])
        self.assertEqual(grid['pilot_report_sha256'], hashlib.sha256(raw['report.json']).hexdigest())
        self.assertEqual(base['training_sha256'], hashlib.sha256(raw['training.json']).hexdigest())
        self.assertEqual(base['source_commit'], metadata['tested_merge_commit'])
        self.assertEqual(grid['script_sha256'], hashlib.sha256((ROOT/'scripts/diagnose_body_transfer.py').read_bytes()).hexdigest())
        expected = probes()
        self.assertEqual(grid['probe_sha256'], probe_sha(expected))
        self.assertEqual(grid['controls'], dict(positive=[0]*96, negative=[1]*96))
        self.assertTrue(grid['complete'])
        for arm, data in grid['arms'].items():
            self.assertEqual(data['model_sha256'], base['arms'][arm]['model_sha256'])
            if arm != 'source':
                self.assertEqual(data['model_sha256'], training['arms'][arm]['model_sha256'])
            self.assertTrue(data['weights_unchanged'])
            self.assertEqual([r['id'] for r in data['rows']], [r['id'] for r in expected])
            for result, task in zip(data['rows'], expected):
                self.assertEqual(result['request'], task['request'])
                self.assertEqual(result['output_sha256'], hashlib.sha256(result['output'].encode()).hexdigest())
                self.assertEqual(result['metrics'], score(result['output'], task))
                self.assertIs(type(result['functional_status']), int)
                self.assertIn(result['functional_status'], (0,1))
            cells = {c: summary([r for r in data['rows'] if r['cell']==c]) for c in data['cells']}
            self.assertEqual(cells, data['cells'])
            self.assertEqual(cells, metadata['summary'][arm])
        candidate = grid['arms']['diverse_bodies']['cells']
        self.assertEqual(candidate['name0-args0-wording0']['functional_passed'], 12)
        self.assertEqual(candidate['name0-args0-wording1']['functional_passed'], 10)
        self.assertTrue(all(v['functional_passed']==0 for c,v in candidate.items()
                            if c not in ('name0-args0-wording0','name0-args0-wording1')))
        self.assertTrue(all(v['functional_passed']==0 for arm in ('source','replay_control')
                            for v in grid['arms'][arm]['cells'].values()))
        manifest,_,_ = load_bank()
        self.assertEqual(base['decision'], decision(base, manifest))
        self.assertEqual(base['decision']['status'], 'rejected')
        self.assertFalse(grid['promotion'])
