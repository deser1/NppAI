import importlib.util
from types import SimpleNamespace
import unittest
from scripts import alias_oracle_diagnostic as diagnostic


class OracleContracts(unittest.TestCase):
    def test_unique_utf8_span(self):
        self.assertEqual(diagnostic.oracle_span('é Implement føø_2(x).', 'føø_2'), (21, 7))
        for prompt in ('missing', 'aa aa'):
            with self.assertRaises(ValueError):
                diagnostic.oracle_span(prompt, 'aa')

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            diagnostic.ReadoutProxy(SimpleNamespace(base=None), 'aa', 'aa', 'training')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class OracleDecoderTests(unittest.TestCase):
    def model(self, prompt, name, prefix=b'def '):
        import torch
        frame = ('[USER]: '+prompt+'\n[AI]:\n').encode()
        answer = prefix+b'?'*len(name.encode())+b'(x):\n    return x\n'
        class Toy:
            base = SimpleNamespace(max_seq_len=256)
            def components(self, tokens, mask):
                q = tokens.shape[1]-len(frame)
                byte = answer[q] if q < len(answer) else 0
                hidden = torch.zeros(1, tokens.shape[1], 2)
                vocab = torch.zeros(1, tokens.shape[1], 256)
                vocab[0, -1, byte] = 1
                attention = torch.zeros(1, tokens.shape[1], tokens.shape[1])
                attention[0, -1, 8] = 1  # deliberately wrong source start
                return hidden, vocab, attention
            def exit_probability(self, query, source):
                return torch.tensor([.9])
        return Toy()

    def test_oracles_are_separate_and_baseline_unchanged(self):
        from scripts.sequential_copy_model import generate
        prompt, name = 'Implement aa(x).', 'aa'
        model = self.model(prompt, name)
        baseline = diagnostic.generate(model, prompt, name, 'autonomous')
        original = generate(model, prompt, with_events=True)
        self.assertEqual((baseline['output'], baseline['events'], baseline['output_bytes']), original)
        start = diagnostic.generate(model, prompt, name, 'oracle_start')
        end = diagnostic.generate(model, prompt, name, 'oracle_exit')
        both = diagnostic.generate(model, prompt, name, 'oracle_both')
        self.assertTrue(start['output'].startswith('def ??('))  # learned exit still fires
        self.assertTrue(end['output'].startswith('def Im('))  # start still wrong
        self.assertTrue(both['output'].startswith('def aa('))
        self.assertEqual(both['events'][0]['cursor'], 18)
        self.assertEqual(both['events'][2]['kind'], 'exit')
        self.assertAlmostEqual(both['readouts'][4]['learned_exit_probability'], .9, places=6)
        self.assertEqual(both['readouts'][4]['effective_exit_probability'], 0)

    def test_no_supplied_declaration_trigger(self):
        prompt, name = 'Implement aa(x).', 'aa'
        result = diagnostic.generate(self.model(prompt, name, b'xyz '), prompt, name, 'oracle_both')
        self.assertFalse(result['triggered'])
        self.assertEqual(result['events'], [])
        self.assertTrue(result['output'].startswith('xyz ??('))

    def test_utf8_oracle_counts_bytes(self):
        prompt, name = 'Implement føø_2(x).', 'føø_2'
        result = diagnostic.generate(self.model(prompt, name), prompt, name, 'oracle_both')
        self.assertTrue(result['output'].startswith('def føø_2('))
        self.assertEqual(result['events'][-1]['response_query'], 4+len(name.encode()))
