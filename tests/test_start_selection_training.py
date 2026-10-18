import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts import start_selection_training as experiment


class StartProtocolTests(unittest.TestCase):
    def test_fixed_train_only_schedule(self):
        dataset = experiment.ROOT/'evaluation/instruction-v3'
        seen = []
        original_read = experiment.pilot.base.read_split
        def read(root, split):
            seen.append(split)
            return original_read(root, split)
        with patch.object(experiment.pilot.base, 'read_split', side_effect=read):
            protocol = experiment.protocol(dataset)
        self.assertEqual(seen, ['train'])
        self.assertEqual(protocol['batches'], 256)
        self.assertEqual((protocol['originals'], protocol['aliases']), (512, 512))
        self.assertEqual(json.loads((experiment.ROOT/'evaluation/start-selection-v1.json').read_bytes()), protocol)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class StartObjectiveTests(unittest.TestCase):
    def test_hard_negative_excludes_gold_generated_and_padding_positions(self):
        import torch
        scores = torch.tensor([[100., 2., 1.5, 900.]], requires_grad=True)
        eligible = torch.tensor([[False, True, True, False]])
        gold = torch.tensor([1])
        control, _ = experiment.pointer_objective(scores, eligible, gold, 'start_ce')
        treatment, parts = experiment.pointer_objective(scores, eligible, gold, 'start_margin')
        self.assertAlmostEqual(float(treatment-control), .5, places=6)
        self.assertAlmostEqual(parts['hard_negative_margin'], .5)
        treatment.backward()
        self.assertEqual(float(scores.grad[0, 0]), 0)
        self.assertEqual(float(scores.grad[0, 3]), 0)
        self.assertLess(float(scores.grad[0, 1]), 0)
        self.assertGreater(float(scores.grad[0, 2]), 0)

    def test_no_hinge_penalty_when_gold_has_margin(self):
        import torch
        scores = torch.tensor([[3., 1., -2.]])
        mask = torch.tensor([[True, True, True]])
        gold = torch.tensor([0])
        control, _ = experiment.pointer_objective(scores, mask, gold, 'start_ce')
        treatment, parts = experiment.pointer_objective(scores, mask, gold, 'start_margin')
        self.assertEqual(float(control), float(treatment))
        self.assertEqual(parts['hard_negative_margin'], 0)
        with self.assertRaises(ValueError):
            experiment.pointer_objective(scores, torch.tensor([[False, True, True]]), gold, 'start_margin')

    def test_real_pointer_gradients_protect_base_exit_and_causal_source(self):
        import torch
        from scripts.sequential_copy_model import SequentialCopyModel
        from scripts.gated_copy_model import GatedCopyModel
        from train_nppai import NppAIModel
        from scripts.training_repro import state_digest
        torch.set_num_threads(1)
        torch.manual_seed(4242)
        model = SequentialCopyModel(GatedCopyModel(NppAIModel(256, 8, 16, 1, 256)))
        model.requires_grad_(False)
        model.copy_q.requires_grad_(True)
        model.copy_k.requires_grad_(True)
        row = {'id': 'train-fixture', 'function': 'føø_2', 'prompt': 'Implement føø_2(x).',
               'answer': 'def føø_2(x):\n    return x\n'}
        x = experiment.pilot.base.sample(row)[0][None, :]
        base, exit_state = state_digest(model.base), state_digest(model.exit)
        loss, parts = experiment.losses(model, [row], x, 'start_margin')
        loss.backward()
        self.assertEqual(parts['examples'], 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.copy_q.parameters()), 0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.copy_k.parameters()), 0)
        for name, parameter in model.named_parameters():
            if name not in experiment.SPEC['trainable']:
                self.assertIsNone(parameter.grad)
        optimizer = torch.optim.AdamW([*model.copy_q.parameters(), *model.copy_k.parameters()], lr=.001)
        optimizer.step()
        self.assertEqual(state_digest(model.base), base)
        self.assertEqual(state_digest(model.exit), exit_state)
