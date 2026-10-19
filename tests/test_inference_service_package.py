import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.package_inference_service import package
from scripts.portable_notepad_smoke import verify_service
from scripts.publish_release_candidate import verify_installation


class InferenceServicePackageTests(unittest.TestCase):
    def test_extracted_runtime_integrity_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root/'service.zip'
            package(archive)
            with zipfile.ZipFile(archive) as contents:
                contents.extractall(root/'extracted')
                self.assertFalse(any(name.endswith(('.nppai', '.pth')) for name in contents.namelist()))
            extracted = root/'extracted'
            verify_service(extracted)
            (extracted/'model_backends.py').write_text('pass\n')
            with self.assertRaises(ValueError):
                verify_service(extracted)

    def test_published_assets_must_match_actual_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = dict(service_from_extracted_package=True, polish_prompt_round_trip=True,
                          apply_disabled_without_validation=True)
            for field, name in (('package_sha256', 'NppAI-x64.zip'),
                                ('service_archive_sha256', 'NppAI-service.zip'),
                                ('model_sha256', 'NppAI-research-staged-mix.nppai')):
                (root/name).write_bytes(name.encode())
                report[field] = hashlib.sha256(name.encode()).hexdigest()
            report['health'] = {'status': 'ready', 'model_id': 'sha256:'+report['model_sha256']}
            (root/'installation-report.json').write_text(json.dumps(report))
            verify_installation(root)
            (root/'NppAI-x64.zip').write_bytes(b'different build')
            with self.assertRaises(ValueError):
                verify_installation(root)


if __name__ == '__main__':
    unittest.main()
