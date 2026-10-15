import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.run_local_model_smoke import measure, pinned_backend


class LocalSmokeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manifest = Path(self.directory.name) / 'tasks.json'
        self.manifest.write_text(json.dumps([
            {'task_id': 'probe', 'prompt': 'Implement f()',
             'cases_file': 'examples/python-validation-cases.json'}]))
        self.backend = SimpleNamespace(weights={'context': 256}, model_id='sha256:model',
            generate=Mock(return_value=SimpleNamespace(output='def f(): return 1',
                model_id='sha256:model', generated_tokens=18, finish_reason='eos')))

    @patch('scripts.run_local_model_smoke.sandbox_validate.validate')
    def test_real_response_and_provenance_preserved(self, validate):
        validate.side_effect = [(0, 'syntax passed'), (1, 'wrong function')]
        result = measure(self.manifest, self.backend, 'sha256:' + 'a' * 64)
        self.assertEqual(result['counts']['functional_failure'], 1)
        entry = result['task_results'][0]
        self.assertEqual(entry['source'], 'def f(): return 1')
        self.assertEqual(entry['generated_tokens'], 18)
        self.assertEqual(len(entry['suite_sha256']), 64)
        request = self.backend.generate.call_args.args[0]
        self.assertNotIn('expected', request.prompt)
        self.assertEqual([c.args[1] for c in validate.call_args_list], ['syntax', 'tests'])

    @patch('scripts.run_local_model_smoke.sandbox_validate.validate', return_value=(1, 'syntax error'))
    def test_syntax_failure_skips_functional_tests(self, validate):
        result = measure(self.manifest, self.backend, 'sha256:' + 'a' * 64)
        self.assertEqual(result['counts']['syntax_failure'], 1)
        self.assertEqual(validate.call_count, 1)

    @patch('scripts.run_local_model_smoke.sandbox_validate.validate')
    def test_empty_output_counts_as_model_failure(self, validate):
        self.backend.generate.return_value.output = '  \n'
        result = measure(self.manifest, self.backend, 'sha256:' + 'a' * 64)
        self.assertEqual(result['counts']['empty_output'], 1)
        self.assertEqual(result['counts']['passed'], 0)
        validate.assert_not_called()

    @patch('scripts.run_local_model_smoke.sandbox_validate.validate', return_value=(2, 'Docker unavailable'))
    def test_infrastructure_aborts_instead_of_inventing_quality_result(self, validate):
        with self.assertRaisesRegex(RuntimeError, 'infrastructure'):
            measure(self.manifest, self.backend, 'sha256:' + 'a' * 64)

    def test_context_failure_precedes_generation(self):
        self.backend.weights['context'] = 1
        with self.assertRaisesRegex(ValueError, 'context'):
            measure(self.manifest, self.backend, 'sha256:' + 'a' * 64)
        self.backend.generate.assert_not_called()

    def test_changed_report_rejected_before_loading_checkpoint(self):
        root = Path(self.directory.name)
        (root / 'loss-report.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'report differs'):
            pinned_backend(root)


if __name__ == '__main__':
    unittest.main()
