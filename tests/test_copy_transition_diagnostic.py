import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import copy_transition_diagnostic as diagnostic

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'evaluation/instruction-v3'


class TransitionContracts(unittest.TestCase):
    def test_retained_raw_points_and_false_diagnostic_claims(self):
        report = diagnostic.retained(ROOT / 'docs/evidence/copy-transition')
        self.assertTrue(diagnostic.audit(report, DATASET))
        mutations = [
            lambda r: r['models']['fixed']['summary'].__setitem__('scaffolded_exact_names', 64),
            lambda r: r['models']['fixed']['cases'][0]['conditional'][0].__setitem__('gate', 2),
            lambda r: r['models']['fixed']['cases'][0]['conditional'][0].__setitem__('aligned_source', 0),
            lambda r: r['models']['fixed']['cases'][0]['conditional'][0].__setitem__('causal_max_delta', 1),
            lambda r: r['models']['fixed']['cases'][0].__setitem__('cached_output_equal', False),
            lambda r: r['models']['fixed'].__setitem__('parameters_after', '0' * 64),
            lambda r: r.__setitem__('promotion', True),
        ]
        for mutate in mutations:
            broken = copy.deepcopy(report)
            mutate(broken)
            with self.assertRaises(ValueError):
                diagnostic.audit(broken, DATASET)

    def test_frozen_cases_open_training_references_only(self):
        read = diagnostic.base.read_split
        opened = []
        def split(root, name):
            opened.append(name)
            return read(root, name)
        with patch.object(diagnostic.base, 'read_split', side_effect=split):
            cases = diagnostic.cases(DATASET)
        self.assertTrue(opened and set(opened) == {'train'})
        self.assertEqual(sum(c['role'] == 'new_alias' for c in cases), 64)
        self.assertEqual([c['index'] for c in cases if c['role'] == 'initial_exact_training'], [2, 12, 13, 14])
        protocol = json.loads((ROOT / 'evaluation/copy-transition-v1.json').read_bytes())
        self.assertEqual(protocol, {'spec': diagnostic.SPEC, 'cases_sha256': diagnostic.quality.digest(diagnostic.quality.encoded(cases))})

    def test_utf8_byte_alignment_and_spaced_source_terminator(self):
        record = {'function': 'føø_7', 'prompt': 'Implement føø_7 (x) returning x.',
                  'answer': 'def føø_7(x):\n    return x\n'}
        prefix, target, sources, start = diagnostic.plan(record)
        self.assertEqual(target, 'føø_7('.encode())
        for source, byte in zip(sources, target):
            self.assertEqual(prefix[source], byte)
        self.assertEqual(sources[0], start)
        self.assertEqual(diagnostic.phase(record, 4), 'name')
        self.assertEqual(diagnostic.phase(record, 4 + len('føø_7'.encode())), 'name_termination')
        self.assertEqual(diagnostic.phase(record, len(record['answer'].encode())), 'eos')

    def test_wrong_archive_rejected_before_weight_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'wrong.zip'
            archive.write_bytes(b'wrong source')
            with self.assertRaisesRegex(ValueError, 'pinned diagnostic archive'):
                diagnostic.prepare(archive, root / 'weights')
            self.assertFalse((root / 'weights').exists())


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class TransitionTensorTests(unittest.TestCase):
    def test_conditional_causality_and_duplicate_utf8_bytes(self):
        import torch
        from train_nppai import NppAIModel
        from scripts.gated_copy_model import GatedCopyModel, source_bounds
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = GatedCopyModel(NppAIModel(256, 8, 16, 1, 256)).eval()
        record = {'function': 'føø_7', 'prompt': 'Implement føø_7(x) returning x.',
                  'answer': 'def føø_7(x):\n    return x\n'}
        with torch.no_grad():
            points = diagnostic.conditional(model, record)
        self.assertEqual([p['target'] for p in points], list('føø_7('.encode()))
        left, right = source_bounds(record['prompt'])
        for point in points:
            self.assertLess(point['causal_max_delta'], 5e-5)
            self.assertTrue(left <= point['peak_position'] < right)
            self.assertAlmostEqual(point['target_probability'],
                (1-point['gate'])*point['vocab_target_probability']+point['gate']*point['pointer_target_probability'], places=6)

    def test_instrumented_full_decoder_matches_existing_with_context_roll(self):
        import torch
        from train_nppai import NppAIModel
        from scripts.gated_copy_model import GatedCopyModel, generate
        from scripts.training_repro import state_digest
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = GatedCopyModel(NppAIModel(256, 8, 16, 1, 64)).eval()
        with torch.no_grad():
            model.base.output.weight.zero_()
        record = {'function': 'x', 'prompt': 'Implement x(x).', 'answer': 'def x(x):\n    return x\n'}
        before = state_digest(model)
        with torch.no_grad():
            measured = diagnostic.full_generate(model, record)
            existing = generate(model, record['prompt'])
        self.assertEqual(measured['output'], existing)
        self.assertGreater(len(measured['bytes']), 32)
        self.assertEqual(measured['finish'], 'eos')
        self.assertEqual(state_digest(model), before)


if __name__ == '__main__':
    unittest.main()
