import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.function_body_training import (DATA, SOURCE, decision, load_bank, measure,
                                            samples, schedule, score, selected_samples, sha)


class BodyTrainingTests(unittest.TestCase):
    def test_bank_is_frozen_bilingual_and_has_no_released_evaluation_prompts(self):
        manifest, train, dev = load_bank()
        self.assertEqual((len(train), len(dev)), (144, 24))
        self.assertEqual({r['language'] for r in train}, {'en', 'pl'})
        self.assertEqual({r['family'] for r in train}, {r['family'] for r in dev})
        released = json.loads((DATA.parent/'released-generalization-v1/tasks.json').read_bytes())
        self.assertFalse({r['request']['prompt'] for r in train} & {r['request']['prompt'] for r in released})
        self.assertFalse({r['function'] for r in train} & {r['function'] for r in dev})
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ('manifest.json', 'train.json', 'development.json'):
                (root/name).write_bytes((DATA/name).read_bytes())
            (root/'train.json').write_bytes((root/'train.json').read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError, 'bytes changed'): load_bank(root)

    def test_changed_development_names_or_arguments_cannot_leak(self):
        manifest, train, dev = load_bank()
        for change in ('name', 'argument'):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                rows = copy.deepcopy(dev)
                if change == 'name': rows[0]['function'] = train[0]['function']
                else: rows[0]['arguments'][0] = train[0]['arguments'][0]
                # Update the control as well, so this tests split isolation, not only parsing.
                old = dev[0]
                rows[0]['answer'] = old['answer'].replace(old['function'], rows[0]['function'])
                for a, b in zip(old['arguments'], rows[0]['arguments']):
                    rows[0]['answer'] = rows[0]['answer'].replace(a, b)
                for case in rows[0]['cases']: case['function'] = rows[0]['function']
                (root/'train.json').write_bytes((DATA/'train.json').read_bytes())
                (root/'development.json').write_text(json.dumps(rows), encoding='utf-8')
                m = copy.deepcopy(manifest)
                m['files']['development.json']['sha256'] = sha(root/'development.json')
                (root/'manifest.json').write_text(json.dumps(m))
                with self.assertRaisesRegex(ValueError, 'leak'): load_bank(root)

    def test_fixed_sampling_and_training_arm_isolation(self):
        m, bank, _ = load_bank()
        draws = schedule(m, 96, 532, len(bank))
        self.assertEqual(draws, schedule(m, 96, 532, len(bank)))
        self.assertEqual(len(draws), 1024)
        old, fact, bodies = ([f'o-{i}' for i in range(96)], [f'f-{i}' for i in range(532)],
                             [f'b-{i}' for i in range(144)])
        for draw in draws:
            control = selected_samples('replay_control', draw, old, fact, bodies)
            candidate = selected_samples('diverse_bodies', draw, old, fact, bodies)
            self.assertEqual(len(control), 8)
            self.assertFalse(any(r.startswith('b-') for r in control))
            self.assertEqual([sum(r.startswith(p) for r in candidate) for p in ('o-', 'f-', 'b-')], [2, 2, 4])
        with self.assertRaises(ValueError): selected_samples('development', draws[0], old, fact, bodies)

    def test_ast_return_is_not_execution_and_nested_return_does_not_count(self):
        row = dict(function='f', arguments=['x'], answer='def f(x):\n    return x\n')
        missing = 'def f(x):\n    def helper():\n        return x\n'
        self.assertTrue(score(missing, row)['syntax'])
        self.assertFalse(score(missing, row)['value_return'])
        self.assertTrue(score('def f(x):\n    return None\n', row)['value_return'])
        self.assertFalse(score('def f(y):\n    return y\n', row)['signature'])
        self.assertFalse(score('def broken(', row)['syntax'])

    def test_gate_preserves_each_hit_and_never_promotes_on_development(self):
        m, _, _ = load_bank()
        def arm(passed):
            return dict(summary=dict(development=dict(functional_passed=passed)), groups=dict(
                original=[dict(id='old-a', metrics=dict(exact=True)), dict(id='old-b', metrics=dict(exact=False))],
                factorial=[dict(id='factorial-a', functional_status=0)]))
        report = dict(validator_image='sha256:'+'a'*64, arms=dict(source=arm(0), replay_control=arm(1), diverse_bodies=arm(12)))
        result = decision(report, m)
        self.assertTrue(result['eligible_for_fresh_evaluation'])
        self.assertFalse(result['promotion'])
        # Equal total exact hits with a loss and a gain must still fail retention.
        report['arms']['diverse_bodies']['groups']['original'][0]['metrics']['exact'] = False
        report['arms']['diverse_bodies']['groups']['original'][1]['metrics']['exact'] = True
        self.assertIn('lost prior original hit: old-a', decision(report, m)['reasons'])
        report['validator_image'] = None
        self.assertEqual(decision(report, m)['status'], 'not_measured')
        self.assertFalse(decision(report, m)['eligible_for_fresh_evaluation'])

    def test_changed_initialization_cannot_enter_inference(self):
        with tempfile.TemporaryDirectory() as d, patch('scripts.function_body_training.DenseLocalBackend') as backend:
            root = Path(d)
            (root/'training.json').write_text('{}')
            (root/'source.nppai').write_bytes(b'wrong source')
            with self.assertRaisesRegex(ValueError, 'source'): measure(root/'source.nppai', root)
            backend.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Training workflow installs PyTorch')
    def test_completion_supervision_excludes_user_bytes_and_includes_body_and_eos(self):
        import torch
        _, bank, _ = load_bank()
        row = bank[0]
        x, y = samples(row, 1024, 256)
        self.assertTrue(torch.all(y[:255] == -100))
        self.assertEqual(bytes(y[255:-1].tolist()).decode(), row['answer'])
        self.assertEqual(int(y[-1]), 0)
        with self.assertRaisesRegex(ValueError, 'truncation'): samples(row, 260, 256)
