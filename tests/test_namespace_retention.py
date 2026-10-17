import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import namespace_retention as pilot

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'evaluation/instruction-v3'


class NamespaceContracts(unittest.TestCase):
    def test_retained_report_traces_and_false_retention_claims(self):
        directory = ROOT / 'docs/evidence/namespace-retention'
        report = pilot.previous.retained(directory)
        self.assertTrue(pilot.audit(report, DATASET, ROOT / 'evaluation/gated-copy-probes-v1/tasks.json'))
        starts = []
        for arm in pilot.ARMS:
            raw = pilot.previous.retained_trace(directory, arm)
            rows = pilot.audit_trace(raw, DATASET, arm, report['arms'][arm]['identity']['trace_sha256'])
            starts.append(rows[0])
            rows[1]['batch_sha256'] = '0' * 64
            broken = b''.join(pilot.quality.encoded(row) for row in rows)
            with self.assertRaisesRegex(ValueError, 'trace exposure'):
                pilot.audit_trace(broken, DATASET, arm, pilot.quality.digest(broken))
        self.assertTrue(all(row == starts[0] for row in starts))
        mutations = [
            lambda r: r['arms']['retained']['retention']['train/original']['exact_response'].__setitem__('retained', 4),
            lambda r: r['arms']['retained']['prefix_summary']['train-family/q_'].__setitem__('name_prefix', 16),
            lambda r: r['arms']['retained']['functional']['models']['copy'].__setitem__('passed', 1),
            lambda r: r['arms']['retained']['identity'].__setitem__('teacher_unchanged', False),
            lambda r: r.__setitem__('promotion', True),
        ]
        for mutate in mutations:
            broken = copy.deepcopy(report)
            mutate(broken)
            with self.assertRaises(ValueError):
                pilot.audit(broken, DATASET, ROOT / 'evaluation/gated-copy-probes-v1/tasks.json')

    def test_name_prefix_is_separate_from_complete_function(self):
        row = {'id': 'training-probe', 'function': 'q_1234', 'prompt': 'Implement q_1234(x).',
               'answer': 'def q_1234(x):\n    return x\n'}
        output = 'def q_1234(x):'
        cases = {'probe': [{'id': row['id'], 'index': 0, 'output': output,
                            'metrics': pilot.base.score(row, output), 'name_prefix': True}]}
        summary = pilot.audit_views(cases, {'probe': [row]})
        self.assertEqual(summary['probe']['requested_signature'], 0)
        self.assertEqual(pilot.prefix_summary(cases)['probe']['name_prefix'], 1)
        cases['probe'][0]['name_prefix'] = False
        with self.assertRaisesRegex(ValueError, 'prefix score'):
            pilot.audit_views(cases, {'probe': [row]})

    def test_train_only_factor_schedule_and_original_targets(self):
        original = pilot.base.read_split
        calls = []
        def read(root, split):
            calls.append(split)
            return original(root, split)
        with patch.object(pilot.base, 'read_split', side_effect=read):
            arms = {a: list(pilot.schedule(DATASET, a)) for a in pilot.ARMS}
            probes = pilot.namespace_probe(DATASET)
        self.assertTrue(calls and all(split == 'train' for split in calls))
        self.assertEqual(arms['fixed'], arms['retained'])
        self.assertEqual(arms['diverse'], arms['diverse_retained'])
        self.assertEqual(len(arms['fixed']), 512)
        seen = set()
        for step, ((left, originals), (right, right_originals)) in enumerate(zip(arms['fixed'], arms['diverse'])):
            self.assertEqual(originals, right_originals)
            self.assertEqual(sum(originals), 2)
            for a, b, is_original in zip(left, right, originals):
                self.assertEqual(a['id'], b['id'])
                if is_original:
                    self.assertEqual(a, b)
                    self.assertIn('return', a['answer'])
                else:
                    self.assertTrue(any(b['function'].startswith(prefix) for prefix in pilot.PREFIXES))
                    seen.add(b['function'].rsplit('_', 1)[0] + '_')
                    if step < 128:
                        self.assertEqual(b['answer'], 'def ' + b['function'] + '(')
        self.assertEqual(seen, set(pilot.PREFIXES))
        self.assertEqual({k:len(v) for k,v in probes.items()},
                         {f'train-family/{p}':16 for p in ('nc_', 'id_', 'q_', 'rename_')})

    def test_frozen_protocol_and_no_probe_alias_exposure(self):
        protocol = json.loads((ROOT / 'evaluation/namespace-retention-v1.json').read_bytes())
        _, inputs = pilot.inputs(DATASET, ROOT / 'evaluation/gated-copy-probes-v1/tasks.json')
        self.assertEqual(protocol, {'spec': pilot.SPEC, 'inputs': inputs,
                                   'schedules': {a:pilot.schedule_evidence(DATASET, a) for a in pilot.ARMS}})
        self.assertEqual(protocol['schedules']['fixed'], protocol['schedules']['retained'])
        self.assertEqual(protocol['schedules']['diverse'], protocol['schedules']['diverse_retained'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class TeacherTests(unittest.TestCase):
    def test_teacher_kl_excludes_prompt_and_padding_positions(self):
        import torch
        from torch import nn
        torch.set_num_threads(1)
        records, original = next(pilot.schedule(DATASET, 'retained'))
        samples = [pilot.base.sample(r) for r in records]
        x = torch.nn.utils.rnn.pad_sequence([s[0] for s in samples], batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence([s[1] for s in samples], batch_first=True, padding_value=-100)
        indices = [i for i, flag in enumerate(original) if flag]
        class PositionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(2, x.shape[1], 256))
            def forward(self, tokens, mask):
                return self.logits.log_softmax(-1)
        teacher = PositionModel().eval()
        teacher.requires_grad_(False)
        student = PositionModel()
        with torch.no_grad():
            change = torch.zeros_like(student.logits)
            change[:, :, 90] = 8
            student.logits.add_(change * (y[indices] == -100)[:, :, None])
        loss = pilot.distillation(student, teacher, x, y, records, original)
        self.assertEqual(float(loss.detach()), 0)

    def test_frozen_teacher_and_original_only_kl(self):
        import torch
        from train_nppai import NppAIModel
        from scripts.gated_copy_model import GatedCopyModel
        torch.set_num_threads(1)
        torch.manual_seed(42)
        teacher = GatedCopyModel(NppAIModel(256, 8, 16, 1, 256)).eval()
        student = copy.deepcopy(teacher)
        for p in teacher.parameters():
            p.requires_grad_(False)
        records, original = next(pilot.schedule(DATASET, 'diverse_retained'))
        samples = [pilot.base.sample(r) for r in records]
        x = torch.nn.utils.rnn.pad_sequence([s[0] for s in samples], batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence([s[1] for s in samples], batch_first=True, padding_value=-100)
        zero = pilot.distillation(student, teacher, x, y, records, original)
        self.assertAlmostEqual(float(zero.detach()), 0, places=6)
        with torch.no_grad():
            student.base.output.weight.add_(torch.randn_like(student.base.output.weight) * 0.1)
        loss = pilot.distillation(student, teacher, x, y, records, original)
        changed_x, changed_y = x.clone(), y.clone()
        changed_x[[i for i, flag in enumerate(original) if not flag]] = 90
        changed_y[[i for i, flag in enumerate(original) if not flag]] = 0
        self.assertTrue(torch.equal(loss, pilot.distillation(student, teacher, changed_x, changed_y, records, original)))
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertGreater(float(student.base.output.weight.grad.abs().sum()), 0)
        teacher.train()
        with self.assertRaisesRegex(ValueError, 'teacher must be frozen'):
            pilot.distillation(student, teacher, x, y, records, original)


if __name__ == '__main__':
    unittest.main()
