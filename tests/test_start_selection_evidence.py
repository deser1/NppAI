import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from scripts import audit_start_selection as evidence

DIRECTORY = evidence.ROOT/'docs/evidence/start-selection'


class StartEvidenceTests(unittest.TestCase):
    def test_reconstruct_full_scores_traces_and_retention(self):
        report = evidence.audit(DIRECTORY)
        for arm in evidence.experiment.ARMS:
            result = report['arms'][arm]
            dev = {k: v for k, v in result['after_summary'].items() if k.startswith(('validation/', 'test/'))}
            self.assertEqual(sum(v['total'] for v in dev.values()), 132)
            self.assertEqual(sum(v['exact_response'] for v in dev.values()), 28)
            self.assertEqual(sum(v['exact_response'] for k, v in dev.items() if not k.endswith('/original')), 1)
            self.assertEqual(result['retention']['train/original']['exact_response'], {'retained': 14, 'lost': 0, 'gained': 1})
            self.assertEqual(result['functional']['models']['copy']['passed'], 0)

    def changed(self, filename, mutate):
        metadata = json.loads((DIRECTORY/'retention.json').read_bytes())
        with zipfile.ZipFile(DIRECTORY/'raw-records.zip') as bundle:
            members = {n: bundle.read(n) for n in bundle.namelist()}
        members[filename] = mutate(members[filename])
        manifest = json.loads(members['manifest.json'])
        manifest[filename] = evidence.p.quality.digest(members[filename])
        members['manifest.json'] = evidence.p.quality.encoded(manifest)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, raw in members.items(): bundle.writestr(name, raw)
        metadata['records_sha256'] = evidence.p.quality.digest(buffer.getvalue())
        return metadata, buffer.getvalue()

    def test_reject_rehashed_invented_score(self):
        def modify(raw):
            report = json.loads(raw)
            report['arms']['start_ce']['after_summary']['test/renamed']['exact_response'] += 1
            return evidence.p.quality.encoded(report)
        metadata, archive = self.changed('report.json', modify)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'retention.json').write_bytes(evidence.p.quality.encoded(metadata))
            (directory/'raw-records.zip').write_bytes(archive)
            with self.assertRaisesRegex(ValueError, 'score summary differs'):
                evidence.audit(directory)

    def test_reject_corrupted_archive(self):
        metadata = json.loads((DIRECTORY/'retention.json').read_bytes())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'retention.json').write_bytes(evidence.p.quality.encoded(metadata))
            (directory/'raw-records.zip').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'records differ'):
                evidence.audit(directory)
