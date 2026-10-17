import json
from pathlib import Path
import tempfile
import unittest


class RetainedOracleTests(unittest.TestCase):
    def test_reconstruct_all_metrics_and_baseline(self):
        from scripts.audit_alias_oracle import audit
        directory = Path(__file__).resolve().parents[1]/'docs/evidence/alias-oracle'
        report = audit(directory)
        expected = {'autonomous': (27, 0), 'oracle_start': (57, 30),
                    'oracle_exit': (29, 0), 'oracle_both': (65, 36)}
        for mode, (dev, aliases) in expected.items():
            groups = report['modes'][mode]
            self.assertEqual(sum(v['metrics'].get('exact_response', 0) for k, v in groups.items()
                                 if k.startswith(('test/', 'validation/'))), dev)
            self.assertEqual(sum(v['metrics'].get('exact_response', 0) for k, v in groups.items()
                                 if k.startswith(('test/', 'validation/')) and not k.endswith('/original')), aliases)
            self.assertEqual(report['functional'][mode]['passed'], 0)

    def test_reject_changed_archive(self):
        from scripts.audit_alias_oracle import audit
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'retention.json').write_text(json.dumps({'records_sha256': '0'*64}))
            (directory/'raw-records.zip').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'archive differs'):
                audit(directory)
