import json
from unittest.mock import patch
import unittest
import evaluate_real_model_e2e as evaluation


def record(task, attempt, source):
    return json.dumps({'task_id': task, 'attempt': attempt, 'backend': 'local',
                       'source': source, 'cases_file': 'examples/python-validation-cases.json'})


class RealModelEvaluationTests(unittest.TestCase):
    @patch.object(evaluation.sandbox_validate, 'validate')
    def test_failure_then_repair_counts_real_attempts(self, validate):
        validate.side_effect = [(1, 'syntax failed'), (0, 'syntax passed'),
                                (0, 'tests passed'), (0, 'syntax passed'),
                                (1, 'wrong result')]
        result = evaluation.evaluate([record('one', 1, 'bad'),
                                      record('one', 2, 'fixed'),
                                      record('two', 1, 'wrong')], 'sha256:' + 'a' * 64)
        self.assertEqual((result['tasks'], result['first_attempt_pass'],
                          result['pass_within_three'], result['repaired_after_failure']), (2, 0, 1, 1))

    @patch.object(evaluation.sandbox_validate, 'validate', return_value=(2, 'Docker unavailable'))
    def test_infrastructure_failure_does_not_count_as_model_failure(self, validate):
        with self.assertRaises(RuntimeError):
            evaluation.evaluate([record('one', 1, 'code')], 'sha256:' + 'a' * 64)

    @patch.object(evaluation.sandbox_validate, 'validate', return_value=(1, 'bad'))
    def test_rejects_skipped_attempts(self, validate):
        with self.assertRaises(ValueError):
            evaluation.evaluate([record('one', 2, 'code')], 'sha256:' + 'a' * 64)


if __name__ == '__main__':
    unittest.main()
