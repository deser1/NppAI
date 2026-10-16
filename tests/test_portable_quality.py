import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import portable_quality as quality

ROOT = Path(__file__).resolve().parents[1]


class PortableQualityTests(unittest.TestCase):
    def test_retained_raw_evidence_and_tampering(self):
        directory = ROOT / 'docs/evidence/portable-quality'
        report = quality.retained_report(directory)
        dataset = ROOT / 'evaluation/instruction-v3'
        tasks = ROOT / 'evaluation/gated-copy-probes-v1/tasks.json'
        self.assertTrue(quality.audit_report(report, dataset, tasks))
        mutations = [
            lambda r: r['models']['both64']['summary']['test/original'].__setitem__('requested_name', 1),
            lambda r: r['functional']['models']['both64'].__setitem__('passed', 1),
            lambda r: r['functional']['models']['both64']['cases'].pop(),
            lambda r: r.__setitem__('promotion', True),
            lambda r: r['inference'].__setitem__('arithmetic', 'both64 forward'),
        ]
        for mutate in mutations:
            broken = copy.deepcopy(report)
            mutate(broken)
            with self.assertRaises(ValueError):
                quality.audit_report(broken, dataset, tasks)

    def test_frozen_inputs_and_complete_response_scores(self):
        from scripts import aligned_name_copy as aligned
        from scripts import resampled_name_training as base
        groups, fingerprints = quality.inputs(ROOT / 'evaluation/instruction-v3',
                                              ROOT / 'evaluation/gated-copy-probes-v1/tasks.json')
        protocol = json.loads((ROOT / 'evaluation/portable-quality-v1.json').read_bytes())
        self.assertEqual(fingerprints, protocol['inputs'])
        self.assertEqual(sum(len(rows) for name, rows in groups.items() if not name.startswith('train/')), 132)
        self.assertEqual(sum(len(rows) for name, rows in groups.items() if name.startswith('train/')), 32)
        cases = {name: [{'index': i, 'id': row['id'], 'output': row['answer'],
                         'metrics': base.score(row, row['answer'])} for i, row in enumerate(rows)]
                 for name, rows in groups.items()}
        summary = quality.audit_groups(cases, groups)
        self.assertTrue(all(s['exact_response'] == s['total'] for s in summary.values()))
        broken = copy.deepcopy(cases)
        first = next(iter(broken.values()))[0]
        first['output'] = 'def unrelated():\n    pass\n'
        with self.assertRaisesRegex(ValueError, 'raw evaluation score'):
            quality.audit_groups(broken, groups)
        broken = copy.deepcopy(cases)
        next(iter(broken.values())).pop()
        with self.assertRaisesRegex(ValueError, 'case count'):
            quality.audit_groups(broken, groups)
        self.assertEqual(summary, aligned.summaries(cases))

    def test_checkpoint_archive_identity_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            archive = path / 'wrong.zip'
            archive.write_bytes(b'untrusted weights')
            with self.assertRaisesRegex(ValueError, 'pinned training archive'):
                quality.checkpoints(archive, path / 'weights')
            self.assertFalse((path / 'weights').exists())


if __name__ == '__main__':
    unittest.main()
