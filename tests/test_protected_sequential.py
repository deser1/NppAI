import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import protected_sequential as pilot

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'evaluation/instruction-v3'
TASKS = ROOT / 'evaluation/gated-copy-probes-v1/tasks.json'


class ProtectedContracts(unittest.TestCase):
    def test_frozen_protocol_and_train_only_exposure(self):
        self.assertEqual(json.loads((ROOT / 'evaluation/protected-sequential-v1.json').read_bytes()),
                         pilot.protocol(DATASET, TASKS))
        read = pilot.base.read_split
        splits = []
        def tracked(root, split):
            splits.append(split)
            return read(root, split)
        with patch.object(pilot.base, 'read_split', side_effect=tracked):
            records = list(pilot.schedule(DATASET))
        self.assertEqual(splits, ['train'])
        self.assertEqual(len(records), 512)
        self.assertEqual(records, list(pilot.prior.schedule(DATASET, 'diverse')))
        self.assertTrue(all(sum(original) == 2 for _, original in records))

    def test_source_cursor_audit_rejects_generated_source_and_wrong_exit(self):
        row = {'prompt': 'Implement aa(x).'}
        cursor = len(b'[USER]: Implement ')
        case = {'output': 'def aa(', 'output_bytes': list(b'def aa('), 'events': [
            {'response_query': 4, 'cursor': cursor, 'kind': 'copy', 'source_byte': 97,
             'exit_probability': 0.1, 'base_byte': 90, 'chosen': 97},
            {'response_query': 5, 'cursor': cursor+1, 'kind': 'copy', 'source_byte': 97,
             'exit_probability': 0.1, 'base_byte': 90, 'chosen': 97},
            {'response_query': 6, 'cursor': cursor+2, 'kind': 'exit', 'source_byte': 40,
             'exit_probability': 0.9, 'base_byte': 40, 'chosen': 40}]}
        pilot.audit_events(case, row)
        for mutate in (
            lambda c: c['events'][0].update(cursor=200),
            lambda c: c['events'][1].update(cursor=cursor),
            lambda c: c['events'][2].update(exit_probability=0.1),
            lambda c: c['output_bytes'].__setitem__(4, 98),
        ):
            changed = copy.deepcopy(case)
            mutate(changed)
            with self.assertRaises(ValueError):
                pilot.audit_events(changed, row)

    def test_retained_evidence_when_available(self):
        directory = ROOT / 'docs/evidence/protected-sequential'
        if not directory.exists():
            self.skipTest('measurement not yet retained')
        from scripts.audit_protected_sequential import retained
        report = retained(directory)
        self.assertTrue(pilot.audit(report, DATASET, TASKS))
        for arm in pilot.ARMS:
            raw = pilot.curriculum.retained_trace(directory, arm)
            identity = report['arms'][arm]['identity']
            rows = pilot.audit_trace(raw, DATASET, arm, identity)
            rows[1]['base'] = '0'*64
            changed = b''.join(pilot.quality.encoded(r) for r in rows)
            with self.assertRaisesRegex(ValueError, 'frozen base'):
                pilot.audit_trace(changed, DATASET, arm, {**identity, 'trace_sha256':pilot.quality.digest(changed)})
        for mutate in (
            lambda r: r['arms']['frozen_sequential']['retention']['train/original']['exact_response'].update(retained=99),
            lambda r: r['arms']['frozen_soft']['identity'].update(base_unchanged=False),
            lambda r: r['arms']['frozen_sequential']['functional']['models']['copy'].update(passed=99),
        ):
            changed = copy.deepcopy(report)
            mutate(changed)
            with self.assertRaises(ValueError):
                pilot.audit(changed, DATASET, TASKS)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class ProtectedTorchTests(unittest.TestCase):
    def model(self):
        import torch
        from train_nppai import NppAIModel
        from scripts.gated_copy_model import GatedCopyModel
        from scripts.sequential_copy_model import SequentialCopyModel
        torch.set_num_threads(1)
        torch.manual_seed(4242)
        return SequentialCopyModel(GatedCopyModel(NppAIModel(256, 8, 16, 1, 256))).eval()

    def test_causal_source_mask_and_frozen_vocabulary(self):
        import torch
        from scripts.gated_copy_model import source_mask, source_bounds
        model = self.model()
        prompt = 'Implement føø(x).'
        raw = ('[USER]: '+prompt+'\n[AI]:\ndef føø(').encode()
        tokens = torch.tensor([list(raw)])
        mask = source_mask(tokens, prompt)
        with torch.no_grad():
            hidden, vocab, attention = model.components(tokens, mask)
            self.assertTrue(torch.equal(vocab, model.base(tokens).softmax(-1)))
            left, right = source_bounds(prompt)
            self.assertEqual(float(attention[:, :, :left].sum()), 0)
            self.assertEqual(float(attention[:, :, right:].sum()), 0)
            changed = tokens.clone()
            changed[0, -1] = 90
            other = model.components(changed, mask)
            self.assertTrue(torch.allclose(hidden[:, :-1], other[0][:, :-1], atol=1e-6))
            self.assertTrue(torch.allclose(attention[:, :-1], other[2][:, :-1], atol=1e-6))
            self.assertEqual(float(model.components(tokens, torch.zeros_like(mask))[2].sum()), 0)

    def test_real_gradients_and_immutable_base_after_update(self):
        import torch
        from scripts.training_repro import state_digest
        model = self.model().train()
        records, _ = next(pilot.schedule(DATASET))
        samples = [pilot.base.sample(r) for r in records]
        x = torch.nn.utils.rnn.pad_sequence([s[0] for s in samples], batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence([s[1] for s in samples], batch_first=True, padding_value=-100)
        before = state_digest(model.base)
        loss, _ = pilot.sequential_loss(model, records, x, y)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
        loss.backward()
        for parameter in (model.copy_q.weight, model.copy_k.weight, model.exit.weight, model.exit.bias):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.base.parameters()))
        optimizer.step()
        self.assertEqual(state_digest(model.base), before)

    def test_cursor_duplicate_utf8_exit_boundary_and_no_reentry(self):
        import torch
        from scripts.sequential_copy_model import step_decision
        class Stub:
            def components(self, x, mask):
                h = torch.zeros(1, x.shape[1], 2)
                vocab = torch.zeros(1, x.shape[1], 256);vocab[:, :, 40] = 1
                att = torch.zeros(1, x.shape[1], x.shape[1]);att[:, :, 1] = 1
                return h, vocab, att
            def exit_probability(self, q, s):
                return torch.tensor(0.0)
        model = Stub()
        tokens = torch.tensor([[99, 97, 97, 195, 184, 0, 90]])
        mask = torch.tensor([[False, True, True, True, True, True, False]])
        state = {'mode':'waiting', 'cursor':None}
        output = b'def '
        copied = []
        for _ in range(4):
            token, event = step_decision(model, tokens, mask, 0, output, state)
            copied.append(token);output += bytes([token])
            self.assertEqual(event['kind'], 'copy')
        self.assertEqual(bytes(copied), 'aaø'.encode())
        token, event = step_decision(model, tokens, mask, 0, output, state)
        self.assertEqual((token, event['kind'], state['mode']), (40, 'boundary', 'ended'))
        self.assertEqual(step_decision(model, tokens, mask, 0, b'def ', state), (40, None))
        state = {'mode':'copy', 'cursor':2}
        model.exit_probability = lambda q, s: torch.tensor(.9)
        self.assertEqual(step_decision(model, tokens, mask, 0, b'def a', state)[1]['kind'], 'exit')
        self.assertEqual(state['mode'], 'ended')
        state = {'mode':'copy', 'cursor':1}
        self.assertEqual(step_decision(model, tokens, mask, 2, b'def a', state)[1]['kind'], 'boundary')
        state = {'mode':'waiting', 'cursor':None}
        self.assertEqual(step_decision(model, tokens, mask, 0, b'', state), (40, None))

    def test_bundle_roundtrip_and_semantic_format_separation(self):
        import torch
        from scripts import sequential_copy_model as sequential
        from scripts import gated_copy_model as soft
        from scripts.training_repro import state_digest
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.nppaicopy'
            sequential.export_bundle(model, path, 'test-source')
            raw = path.read_bytes()
            sequential.export_bundle(model, path, 'test-source')
            self.assertEqual(path.read_bytes(), raw)
            loaded, manifest = sequential.load_bundle(path)
            self.assertEqual(state_digest(model), state_digest(loaded))
            self.assertTrue(all(not p.requires_grad for p in loaded.base.parameters()))
            self.assertFalse(manifest['native_compatible'])
            with self.assertRaisesRegex(ValueError, 'unsupported experimental bundle'):
                soft.load_bundle(path)
            entries = soft.bounded_zip(raw, ('base.nppai','head.npz','manifest.json'))
            entries['head.npz'] += b'bad'
            path.write_bytes(soft.zip_bytes(entries))
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                sequential.load_bundle(path)
