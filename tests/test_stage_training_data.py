import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_stage_dataset import build, derive_training
from scripts.train_staged_protocol import supervision
from model_backends import format_prompt
from staged_generation import stage_values

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'evaluation/staged-protocol-v1'


class StageDataTests(unittest.TestCase):
    def test_frozen_dataset_rebuild_and_split_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            build(Path(directory))
            for name in ('train.json','new-evaluation.json','manifest.json'):
                self.assertEqual((Path(directory)/name).read_bytes(),(DATA/name).read_bytes())
        train=json.loads((DATA/'train.json').read_bytes())
        evaluation=json.loads((DATA/'new-evaluation.json').read_bytes())
        self.assertEqual(len(train),96)
        self.assertEqual(len(evaluation),12)
        self.assertTrue({r['function'] for r in train}.isdisjoint(r['function'] for r in evaluation))
        with self.assertRaises(ValueError):
            derive_training(evaluation)

    def test_exact_runtime_stage_format_and_full_context_fit(self):
        train=json.loads((DATA/'train.json').read_bytes())
        for r in train:
            prior={}
            for s in ('specification','plan','code'):
                p,target=supervision(r,s)
                self.assertEqual(p.model_dump(),type(p)(**stage_values(s,r['prompt'],'',prior,128)).model_dump())
                self.assertLess(len(target.encode())+1,128)
                self.assertLessEqual(len(format_prompt(p).encode())+len(target.encode())+1,1024)
                prior[s]=target
            with self.assertRaises(ValueError):
                supervision(r,'unknown')

    def test_audit_rejects_score_denominator_and_request_tampering(self):
        import copy
        from scripts.audit_staged_pilot import audit_report
        from scripts.evaluate_task_plan import evaluate
        from model_backends import GenerationResponse
        code='def f():\n    return 1\n'
        class Backend:
            def __init__(self): self.outputs=iter(['spec','plan',code,code,code])
            def generate(self,payload):
                return GenerationResponse(model_id='fixture',output=next(self.outputs),generated_tokens=1,finish_reason='eos')
        records=[{'id':'one','prompt':'write f','function':'f','answer':code}]
        original=evaluate(Backend(),records)
        self.assertTrue(audit_report(original,records))
        for kind in ('score','count','prompt'):
            r=copy.deepcopy(original)
            if kind=='score': r['summary']['staged']['exact_response']=0
            if kind=='count': r['total']=0
            if kind=='prompt': r['cases'][0]['arms']['staged']['trace']['stages'][0]['request']['prompt']=code
            with self.assertRaises(ValueError): audit_report(r,records)

    def test_authored_cases_are_nonempty_and_match_reference_signatures(self):
        # Parse references only; tests never execute candidates or references on host.
        for r in json.loads((DATA/'new-evaluation.json').read_bytes()):
            tree=ast.parse(r['answer'])
            self.assertEqual(tree.body[0].name,r['function'])
            self.assertGreaterEqual(len(r['cases']),3)
            self.assertTrue(all(c['function']==r['function'] for c in r['cases']))
        manifest=json.loads((DATA/'manifest.json').read_bytes())
        for path,info in manifest['files'].items():
            self.assertEqual(hashlib.sha256((DATA/path).read_bytes()).hexdigest(),info['sha256'])


if __name__=='__main__': unittest.main()
