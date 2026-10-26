import ast
import tempfile
from pathlib import Path
import unittest
import numpy as np
from model_backends import GenerationRequest,format_prompt
from python_function_decoding import CachedLogits,acceptable,declaration
from scripts.factorial_repair import dataset
from scripts.check_dense_runtime_parity import load_weights,numpy_logits

class RepairTests(unittest.TestCase):
    def test_fixed_boundary_preserves_unicode_and_default(self):
        p=GenerationRequest(prompt='Zażółć gęślą jaźń')
        self.assertEqual(format_prompt(p),'[USER]: Zażółć gęślą jaźń\n[AI]:\n')
        self.assertEqual(len(format_prompt(p,256).encode()),256)
        with self.assertRaises(ValueError):format_prompt(p,10)
    def test_dataset_separation_and_cases(self):
        train,ev=dataset();self.assertEqual((len(train),len(ev)),(532,18))
        keys=lambda rows:{(r['request']['prompt'],r['request']['assistant_prefix']) for r in rows}
        self.assertFalse(keys(train)&keys(ev))
        for row in train+ev:ast.parse(row['answer']);self.assertEqual(row['cases'][0]['expected'],1)
    def test_structure_is_not_execution_or_correctness(self):
        p=GenerationRequest(prompt='Implement f(n). Return factorial.')
        header,signature=declaration(p);self.assertEqual(header,'def f(n):\n    ')
        self.assertTrue(acceptable('def f(n):\n    return 1\n',signature))
        self.assertFalse(acceptable('def other(n):\n    return 1\n',signature))
        self.assertFalse(acceptable('def f(n):\n    return unknown\n',signature))
        self.assertFalse(acceptable('print("side effect")\ndef f(n):\n    return n\n',signature))
        self.assertEqual(declaration(GenerationRequest(prompt='Compare f(n) with g(n)'))[0],'')
    def test_cached_logits_forks_and_sliding_match_full(self):
        # The same serialized fixture exercised by existing runtime tests.
        from test_model_backends import checkpoint
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.nppai';checkpoint(path);w=load_weights(path);cache=CachedLogits(w)
            prefix=[1,2,3]
            np.testing.assert_allclose(cache(prefix),numpy_logits(w,prefix).reshape(-1),rtol=2e-5,atol=2e-5)
            cache.next_depth()
            for suffix in (4,5):
                tokens=prefix+[suffix];np.testing.assert_allclose(cache(tokens),numpy_logits(w,tokens).reshape(-1),rtol=2e-5,atol=2e-5)
            cache.next_depth();tokens=prefix+[4,6]
            np.testing.assert_allclose(cache(tokens),numpy_logits(w,tokens).reshape(-1),rtol=2e-5,atol=2e-5)
            tokens=list(range(129));np.testing.assert_allclose(cache(tokens),numpy_logits(w,tokens).reshape(-1),rtol=2e-5,atol=2e-5)
    def test_beam_uses_neural_alternative_without_reference_or_execution(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from python_function_decoding import generate
        payload=GenerationRequest(prompt='Implement f(n).',max_new_tokens=32)
        header,_=declaration(payload);prefix=(format_prompt(payload)+header).encode()
        class Logits:
            def __init__(self,w):pass
            def next_depth(self):pass
            def __call__(self,tokens):
                body=bytes(tokens[len(prefix):]);values=np.full(256,-100.,dtype=np.float32)
                for text,bias in [(b'return unknown\n',0.),(b'return n\n',-.3)]:
                    if text.startswith(body):values[text[len(body)] if len(body)<len(text) else 0]=bias
                return values
        backend=SimpleNamespace(weights={'context':256},prompt_bytes=0,model_id='fixture')
        with patch('python_function_decoding.CachedLogits',Logits):
            response=generate(backend,payload)
        self.assertEqual(response.output,header+'return n\n')
        self.assertEqual(response.finish_reason,'eos')
