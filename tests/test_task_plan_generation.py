import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from model_backends import BackendUnavailable, DenseLocalBackend, GenerationResponse
from staged_generation import run_staged
from scripts.evaluate_task_plan import evaluate
from test_model_backends import checkpoint


class ScriptedBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def generate(self, payload):
        self.requests.append(payload)
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, GenerationResponse):
            return value
        return GenerationResponse(model_id="fixture", output=value,
                                  generated_tokens=1, finish_reason="eos")


class TaskPlanTests(unittest.TestCase):
    def test_carries_exact_unicode_task_and_prior_artifacts_without_reference(self):
        code = 'def silnia(n):\n    return 1\n'
        backend = ScriptedBackend(['Wejście: n; wynik: n!', 'Krótki plan: pętla.', code])
        events = []
        result = run_staged(backend, 'Wygeneruj silnię 🧪', context='Kontekst', on_stage=events.append)
        self.assertTrue(result.completed)
        self.assertEqual(result.code, code)
        self.assertEqual(events, ['specification', 'plan', 'code'])
        self.assertEqual(result.execution_status, 'not_run')
        self.assertEqual(result.generated_tokens, 3)
        self.assertTrue(all('Wygeneruj silnię 🧪' in r.prompt for r in backend.requests))
        prior = json.loads(backend.requests[2].context.split('data):\n')[1])
        self.assertEqual(prior, {'specification': 'Wejście: n; wynik: n!', 'plan': 'Krótki plan: pętla.'})
        self.assertTrue(all(not r.assistant_prefix for r in backend.requests))

    def test_failed_or_truncated_stage_stops_and_retains_raw_response(self):
        for failure in ['', BackendUnavailable('no checkpoint'),
                GenerationResponse(model_id='fixture', output='partial', generated_tokens=128, finish_reason='length')]:
            with self.subTest(failure=failure):
                backend = ScriptedBackend(['spec', failure, 'must not run'])
                result = run_staged(backend, 'task')
                self.assertFalse(result.completed)
                self.assertEqual(result.code, '')
                self.assertEqual(len(backend.requests), 2)
                self.assertTrue(result.stages[-1].error)
                self.assertEqual(result.stages[0].response['output'], 'spec')

    def test_identity_budget_and_request_overflow_fail_closed(self):
        for response in (
            GenerationResponse(model_id='other', output='plan', generated_tokens=1, finish_reason='eos'),
            GenerationResponse(model_id='fixture', output='plan', generated_tokens=129, finish_reason='eos')):
            result = run_staged(ScriptedBackend(['spec', response]), 'task')
            self.assertFalse(result.completed)
            self.assertTrue(result.stages[-1].error)
        result = run_staged(ScriptedBackend(['spec']), 'task', context='x' * 100_000)
        self.assertEqual(len(result.stages), 2)
        self.assertTrue(result.stages[-1].error)
        for invalid in (0, 171, True):
            with self.assertRaises(ValueError):
                run_staged(ScriptedBackend([]), 'task', tokens_per_stage=invalid)

    def test_reference_only_used_after_generation_and_paired_controls(self):
        code = 'def f():\n    return 7\n'
        backend = ScriptedBackend(['spec', 'plan', code, '', code])
        report = evaluate(backend, [{'id': 'one', 'prompt': 'write f', 'function': 'f', 'answer': code}])
        self.assertEqual([r.max_new_tokens for r in backend.requests], [128,128,128,128,384])
        self.assertTrue(all(code not in r.prompt + r.context for r in backend.requests))
        self.assertEqual(report['summary']['staged']['exact_response'], 1)
        self.assertEqual(report['summary']['direct_code_budget']['exact_response'], 0)
        self.assertEqual(report['execution_status'], 'not_run')
        with self.assertRaises(ValueError):
            evaluate(backend, [])

    def test_real_dense_runtime_reports_context_failure_without_truncating_task(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'fixture.nppai'
            checkpoint(model)
            backend = DenseLocalBackend(str(model))
            result = run_staged(backend, 'ż' * 100)
            self.assertFalse(result.completed)
            self.assertIn('byte context', result.stages[0].error)
            self.assertIn('ż' * 100, result.stages[0].request['prompt'])

    def test_cli_retains_failure_report_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, dataset, output = root/'model.nppai', root/'tasks.json', root/'report.json'
            checkpoint(model, token=0)
            dataset.write_text(json.dumps([{'id': 'a', 'prompt': 'task', 'function': 'f',
                'answer': 'def f():\n    return 1\n'}]), encoding='utf-8')
            run = subprocess.run([sys.executable, 'scripts/evaluate_task_plan.py', '--model', str(model),
                '--dataset', str(dataset), '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['summary']['staged']['errors'], 1)
            self.assertTrue(report['dataset_sha256'])
            self.assertEqual(report['cases'][0]['arms']['staged']['trace']['stages'][0]['response']['output'], '')


if __name__ == '__main__':
    unittest.main()
