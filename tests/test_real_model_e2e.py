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
        self.assertEqual(result['by_initial_backend']['local']['tasks'], 2)
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

    @patch.object(evaluation.sandbox_validate, 'validate', return_value=(1, 'failed'))
    def test_changed_suite_is_rejected(self, validate):
        with patch.object(evaluation.sandbox_validate, 'load_cases',
                          side_effect=[([], 'digest-a'), ([], 'digest-b')]):
            with self.assertRaisesRegex(ValueError, 'test suite changed'):
                evaluation.evaluate([record('one', 1, 'bad'), record('one', 2, 'still bad')],
                                    'sha256:' + 'a' * 64)

    @patch.object(evaluation.sandbox_validate, 'validate', return_value=(0, 'passed'))
    def test_separate_local_and_online_metrics(self, validate):
        local = json.loads(record('local-task', 1, 'def f(): return 1'))
        online = json.loads(record('online-task', 1, 'def f(): return 1'))
        online['backend'] = 'online'
        result = evaluation.evaluate([json.dumps(local), json.dumps(online)],
                                     'sha256:' + 'a' * 64)
        self.assertEqual(result['by_initial_backend']['local']['first_attempt_pass'], 1)
        self.assertEqual(result['by_initial_backend']['online']['first_attempt_pass'], 1)

    @patch.object(evaluation.sandbox_validate, 'validate')
    def test_empty_and_non_text_sources_never_reach_validator(self, validate):
        for source in ('', None, 42, []):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'nonempty Python text'):
                evaluation.evaluate([record('one', 1, source)], 'sha256:' + 'a' * 64)
        validate.assert_not_called()

    @patch.object(evaluation.sandbox_validate, 'validate')
    def test_online_repair_is_counted_under_initial_local_backend(self, validate):
        validate.side_effect = [(1, 'syntax failed'), (0, 'syntax passed'), (0, 'tests passed')]
        repair = json.loads(record('one', 2, 'fixed'))
        repair['backend'] = 'online'
        result = evaluation.evaluate([record('one', 1, 'bad'), json.dumps(repair)],
                                     'sha256:' + 'a' * 64)
        self.assertEqual(result['by_initial_backend'], {'local': {
            'tasks': 1, 'first_attempt_pass': 0, 'pass_within_three': 1,
            'repaired_after_failure': 1}})
        self.assertEqual(result['task_results']['one'][1]['backend'], 'online')


if __name__ == '__main__':
    unittest.main()
