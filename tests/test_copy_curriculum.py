import importlib.util
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import copy_curriculum as experiment
from scripts import resampled_name_training as base

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'evaluation/instruction-v3'


class CurriculumContracts(unittest.TestCase):
    def test_retained_raw_evidence_traces_and_tampering(self):
        directory = ROOT / 'docs/evidence/copy-curriculum'
        report = experiment.retained(directory)
        tasks = ROOT / 'evaluation/gated-copy-probes-v1/tasks.json'
        self.assertTrue(experiment.audit(report, DATASET, tasks))
        starts = []
        for arm in experiment.ARMS:
            raw = experiment.retained_trace(directory, arm)
            rows = experiment.audit_trace(raw, DATASET, arm, report['arms'][arm]['identity']['trace_sha256'])
            starts.append(rows[0])
            rows[1]['batch_sha256'] = '0' * 64
            broken = b''.join(experiment.quality.encoded(row) for row in rows)
            with self.assertRaisesRegex(ValueError, 'trace batch'):
                experiment.audit_trace(broken, DATASET, arm, experiment.quality.digest(broken))
        self.assertEqual(starts[0], starts[1])
        mutations = [
            lambda r: r['arms']['curriculum']['summary']['test/original'].__setitem__('exact_response', 28),
            lambda r: r['arms']['curriculum']['functional']['models']['copy'].__setitem__('passed', 1),
            lambda r: r['arms']['curriculum']['training_only_gate_diagnostic'][0]['name_gate'].__setitem__(0, 2),
            lambda r: r['arms']['curriculum']['retention']['train/original']['exact_response'].__setitem__('lost', 0),
            lambda r: r.__setitem__('promotion', True),
        ]
        for mutate in mutations:
            broken = copy.deepcopy(report)
            mutate(broken)
            with self.assertRaises(ValueError):
                experiment.audit(broken, DATASET, tasks)

    def test_train_only_schedule_and_stage_boundary(self):
        original = base.read_split
        calls = []
        def read(path, split):
            calls.append(split)
            return original(path, split)
        with patch.object(base, 'read_split', side_effect=read):
            control = list(experiment.schedule(DATASET, 'control'))
            curriculum = list(experiment.schedule(DATASET, 'curriculum'))
        self.assertEqual(calls, ['train', 'train'])
        self.assertEqual(len(control), 1024)
        for step, (left, right) in enumerate(zip(control, curriculum)):
            for a, b in zip(left, right):
                self.assertEqual({k:v for k,v in a.items() if k != 'answer'},
                                 {k:v for k,v in b.items() if k != 'answer'})
                self.assertEqual(b['answer'], 'def ' + b['function'] + '(' if step < 256 else a['answer'])
        self.assertEqual(curriculum[256:], control[256:])
        with self.assertRaises(ValueError):
            list(experiment.schedule(DATASET, 'invalid'))

    def test_frozen_protocol_exposure(self):
        protocol = json.loads((ROOT / 'evaluation/copy-curriculum-v1.json').read_bytes())
        self.assertEqual(protocol['spec'], experiment.SPEC)
        for arm in experiment.ARMS:
            self.assertEqual(protocol['schedules'][arm], experiment.schedule_evidence(DATASET, arm))
        self.assertLess(protocol['schedules']['curriculum']['response_targets'],
                        protocol['schedules']['control']['response_targets'])
        _, inputs = experiment.quality.inputs(DATASET, ROOT / 'evaluation/gated-copy-probes-v1/tasks.json')
        self.assertEqual(inputs, protocol['evaluation_inputs'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class CurriculumTensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def model(self):
        from train_nppai import NppAIModel
        from scripts.gated_copy_model import GatedCopyModel
        return GatedCopyModel(NppAIModel(256, 8, 16, 1, 256))

    def sample(self):
        import torch
        record = {'prompt': 'Implement føø_77(x) returning x.',
                  'function': 'føø_77', 'answer': 'def føø_77(x):\n    return x\n'}
        x, y = base.sample(record)
        return record, x[None, :], y[None, :]

    def test_utf8_causal_labels_and_non_name_gate_targets(self):
        import torch
        record, x, y = self.sample()
        gate, points = experiment.supervision([record], x, y)
        self.assertEqual(int(gate.sum()), len(record['function'].encode()))
        for batch, query, source in points:
            self.assertLess(source, query)
            self.assertEqual(int(x[batch, source]), int(y[batch, query]))
            self.assertEqual(float(gate[batch, query]), 1)
        self.assertTrue(torch.all(gate[y == -100] == 0))
        corrupted = y.clone()
        corrupted[0, points[0][1]] = 0
        with self.assertRaisesRegex(ValueError, 'causal copy supervision'):
            experiment.supervision([record], x, corrupted)

    def test_auxiliary_reaches_head_with_frozen_base_and_exact_control_nll(self):
        import torch
        from torch.nn import functional as F
        from scripts.gated_copy_model import source_mask
        record, x, y = self.sample()
        model = self.model()
        for parameter in model.base.parameters():
            parameter.requires_grad_(False)
        control, _ = experiment.losses(model, [record], x, y, False)
        expected = F.nll_loss(model(x, source_mask(x, record['prompt'])).reshape(-1, 256),
                              y.reshape(-1), ignore_index=-100)
        self.assertTrue(torch.equal(control, expected))
        loss, parts = experiment.losses(model, [record], x, y, True)
        self.assertAlmostEqual(float(loss.detach()),
                               parts['response_nll'] + parts['pointer_nll'] + 0.1 * parts['gate_bce'], places=5)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        for parameter in (model.copy_q.weight, model.copy_k.weight, model.gate.weight, model.gate.bias):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        # Training supervision does not grant the model future-token attention.
        with torch.no_grad():
            changed = x.clone()
            changed[:, -1] = 90
            mask = source_mask(x, record['prompt'])
            self.assertTrue(torch.allclose(model(x, mask)[:, :-1], model(changed, mask)[:, :-1], atol=1e-5))


if __name__ == '__main__':
    unittest.main()
