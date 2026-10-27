"""Prevent omitted failures, fabricated validation, and reference leakage."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_backends import GenerationResponse
from scripts.evaluate_released_model import SUITE, audit, load_suite, metrics, run, sha, summarize


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.tasks = load_suite()

    def report(self, image=None):
        m, tasks = self.manifest, self.tasks
        arms = {}
        for arm, settings in m['arms'].items():
            rows = [dict(id=t['id'], group=t['group'], request=t['request'], output='',
                         output_sha256=hashlib.sha256(b'').hexdigest(), metrics=metrics('', t),
                         error=None, response_model_id='sha256:' + m['checkpoint_sha256'],
                         functional_status=1 if image else None) for t in tasks]
            arms[arm] = dict(settings=settings, weights_unchanged=True, rows=rows, summary=summarize(rows))
        return dict(checkpoint_sha256=m['checkpoint_sha256'], tasks_sha256=m['tasks_sha256'],
                    manifest_sha256=sha(SUITE / 'manifest.json'), promotion=False, complete=True, arms=arms,
                    validator_image=image, controls=dict(positive=[0]*len(tasks), negative=[1]*len(tasks)) if image else None)

    def test_suite_is_frozen_and_separated_from_known_training_requests(self):
        self.assertEqual(len(self.tasks), 40)
        self.assertEqual(sum(t['group'] == 'new_operations' for t in self.tasks), 24)
        train = json.loads((SUITE.parent / 'staged-protocol-v1/train.json').read_bytes())
        from scripts.factorial_repair import dataset
        factorial, _ = dataset()
        known = {r['prompt'] for r in train} | {r['request']['prompt'] for r in factorial}
        self.assertFalse(known & {t['request']['prompt'] for t in self.tasks})
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'manifest.json').write_bytes((SUITE/'manifest.json').read_bytes())
            (root/'tasks.json').write_bytes((SUITE/'tasks.json').read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                load_suite(root)

    def test_retained_ci_evidence_matches_raw_digest_and_complete_suite(self):
        directory = SUITE.parents[1] / 'docs/evidence/released-generalization-v1'
        metadata = json.loads((directory / 'manifest.json').read_bytes())
        report = json.loads((directory / 'report.json').read_bytes())
        self.assertEqual(sha(directory / 'report.json'), metadata['report_sha256'])
        self.assertEqual(report['source_commit'], metadata['tested_merge_commit'])
        self.assertTrue(audit(report, self.manifest, self.tasks))
        self.assertEqual(report['arms']['released_profile']['summary']['new_operations']['functional_passed'], 0)
        self.assertEqual(report['arms']['released_profile']['summary']['exposed_bilingual_regression']['functional_passed'], 1)

    def test_ast_success_does_not_mean_functional_success(self):
        task = self.tasks[0]
        wrong = 'def ' + task['function'] + '(' + task['arguments'][0] + '):\n    return None\n'
        self.assertTrue(metrics(wrong, task)['signature'])
        self.assertFalse(metrics(wrong.replace(task['arguments'][0], 'wrong_arg'), task)['signature'])
        self.assertFalse(metrics('def broken(', task)['syntax'])
        self.assertEqual(summarize(self.report()['arms']['released_profile']['rows'])['new_operations']['functional_measured'], 0)

    def test_audit_rejects_missing_rows_forged_scores_or_unavailable_docker(self):
        good = self.report()
        self.assertTrue(audit(good, self.manifest, self.tasks))
        for mutation in ('omit', 'score', 'functional', 'checkpoint', 'complete', 'weights'):
            bad = copy.deepcopy(good)
            arm = bad['arms']['released_profile']
            if mutation == 'omit': arm['rows'].pop()
            if mutation == 'score': arm['rows'][0]['metrics']['signature'] = True
            if mutation == 'functional': arm['rows'][0]['functional_status'] = 0
            if mutation == 'checkpoint': arm['rows'][0]['response_model_id'] = 'different'
            if mutation == 'complete': bad['complete'] = False
            if mutation == 'weights': arm['weights_unchanged'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit(bad, self.manifest, self.tasks)
        docker = self.report('sha256:' + 'a'*64)
        self.assertTrue(audit(docker, self.manifest, self.tasks))
        docker['controls']['negative'][0] = 0
        with self.assertRaisesRegex(ValueError, 'controls'):
            audit(docker, self.manifest, self.tasks)

    def test_generator_receives_only_requests_and_errors_stay_in_denominator(self):
        calls = []
        class Backend:
            def __init__(self, *args, **kwargs): self.weights = {}; self.count = 0
            def generate(self, payload):
                calls.append(payload.model_dump())
                self.count += 1
                if self.count == 1: raise ValueError('bounded request failed')
                return GenerationResponse(model_id='sha256:' + self_manifest['checkpoint_sha256'],
                                          output='not python!', generated_tokens=11, finish_reason='eos')
        self_manifest = self.manifest
        with tempfile.TemporaryDirectory() as d, patch('scripts.evaluate_released_model.load_suite', return_value=(self.manifest, self.tasks)), patch('scripts.evaluate_released_model.sha', return_value=self.manifest['checkpoint_sha256']), patch('scripts.evaluate_released_model.DenseLocalBackend', Backend), patch('scripts.evaluate_released_model.audit', return_value=True):
            report = run(Path(d)/'model', Path(d)/'output')
        self.assertEqual(len(calls), 80)
        for request, task in zip(calls[:40], self.tasks):
            self.assertEqual(request['prompt'], task['request']['prompt'])
            self.assertEqual(request['context'], '')
            self.assertEqual(request['assistant_prefix'], '')
        for arm in report['arms'].values():
            self.assertEqual(len(arm['rows']), 40)
            self.assertEqual(arm['summary']['new_operations']['generation_errors'], 1)
