"""Reference byte offsets and conditional diagnostics must not imply free generation."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import unittest
from scripts.diagnose_body_bytes import (plan,visible_plan,metrics,infer,build,score,summarize,audit,
    MODELS,SOURCE_RUN,HASHES)

DATASET=Path(__file__).resolve().parents[1]/'evaluation/instruction-v3'


class BodyByteTests(unittest.TestCase):
    def record(self,answer='def foo(abcd):\n    return abcd + abcd\n'):
        return {'id':'fixture','function':'foo','prompt':'Define foo(abcd) to return twice abcd.','answer':answer}

    def test_shifted_target_offsets_eos_and_no_header_spans(self):
        r=self.record();p=plan(r);header=b'def foo(abcd):\n'
        prefix=('[USER]: '+r['prompt']+'\n[AI]:\n').encode()
        self.assertEqual(p['start'],len(prefix)+len(header)-1)
        self.assertEqual(p['expected'],list(b'    return abcd + abcd\n')+[0])
        self.assertEqual(p['categories'][-1],'eos')
        for s in p['argument_spans']:
            self.assertEqual(bytes(p['expected'][s['start']:s['end']]),b'abcd')
            index=p['start']+s['start']
            self.assertEqual(p['sequence'][index+1],ord('a'))
            self.assertEqual(p['sequence'][:index+1][-1],ord(' '))
        for invalid in ({**r,'prompt':'x\0y'},{**r,'prompt':'x'*300}):
            with self.assertRaises(ValueError):plan(invalid)

    def test_utf8_literals_attributes_keyword_names_not_argument_uses(self):
        r={'id':'utf8','function':'foo','prompt':'Define foo(x). é',
           'answer':'def foo(x):\n    return "é x" + x.x(x=x)\n'}
        p=plan(r);body=bytes(p['expected'][:-1])
        self.assertEqual(len(p['argument_spans']),2)
        self.assertEqual([s['start'] for s in p['argument_spans']],[body.index(b'x.x'),body.index(b'x=x')+2])
        literal=body.index(b'x"')
        self.assertEqual(p['categories'][literal],'other_body')
        self.assertEqual(p['categories'][body.index(b'x.x')+2],'operation_name')
        self.assertEqual(p['categories'][body.index(b'x=x')],'operation_name')

    def test_gold_initial_byte_can_hide_failure_to_begin_copying(self):
        p=plan(self.record());pred=list(p['expected']);probs=[1.0]*len(pred)
        first=p['argument_spans'][0]['start'];pred[first]=ord('z');probs[first]=0.25
        m=metrics(p,pred,probs)
        self.assertEqual(m['argument_occurrences'],{'total':2,'all_bytes_correct':1,
            'initial_byte_correct':1,'with_suffix':2,'suffix_all_correct':2})
        self.assertEqual(m['cases_all_argument_bytes_correct'],0)
        self.assertAlmostEqual(m['by_category']['argument_first_byte']['nll_sum'],math.log(4))
        for bad_pred,bad_probs in ((pred[:-1],probs),([256]*len(pred),probs),(pred,[float('nan')]*len(probs))):
            with self.assertRaises(ValueError):metrics(p,bad_pred,bad_probs)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required for causal prefix logits')
    def test_actual_causal_logits_match_truncated_gold_prefix(self):
        import torch
        from train_nppai import NppAIModel
        torch.manual_seed(42);torch.set_num_threads(2)
        model=NppAIModel(256,64,128,2,256).eval();p=plan(self.record())
        with torch.no_grad():result=infer(model,p)
        self.assertTrue(result['prefix_check']['consistent'])
        self.assertEqual(result['prefix_check']['count'],2)
        self.assertEqual(len(result['predictions']),len(p['expected']))

    def fixture(self):
        groups={};before={};after={}
        for split,rows in build(DATASET,'varied_arguments')[2].items():
            cases=[];cache=[]
            for r in rows:
                p=plan(r);cached={'id':r['id'],'task_signature_assisted':score(r,''),'single_model':score(r,'')}
                cache.append(cached)
                item={'predictions':p['expected'],'target_probabilities':[1.0]*len(p['expected']),
                      'metrics':metrics(p,p['expected'],[1.0]*len(p['expected'])),
                      'prefix_check':{'count':len(p['argument_spans']),'consistent':True,'max_logit_delta':0.0},
                      'cached_assisted':cached['task_signature_assisted'],'cached_free':cached['single_model']}
                cases.append({**r,'layout':visible_plan(p),'models':{n:copy.deepcopy(item) for n in MODELS}})
            groups[split]=cases;before[split]={'cases':cache};after[split]={'cases':copy.deepcopy(cache)}
        sources={p:{'model_sha256':p,'source':{'model_sha256':'source20k'},
                    'evaluation':{'before':before,'after':after}} for p in HASHES}
        report={'source_run_id':SOURCE_RUN,'source_report_sha256':HASHES,'model_sha256':{n:n for n in MODELS},
                'mode':'teacher_forced_reference_prefixes','groups':groups,'summary':{s:summarize(c) for s,c in groups.items()}}
        return report,sources

    def test_raw_json_roundtrip_tampering_and_no_generation_credit(self):
        baseline,sources=self.fixture()
        self.assertTrue(audit(json.loads(json.dumps(baseline)),DATASET,sources)['valid'])
        summary=baseline['summary']['test/argument_aliases']['body_only']
        self.assertEqual(summary['cases_all_argument_bytes_correct'],336)
        self.assertEqual(summary['cached_assisted_exact'],0)
        for mutate in (
            lambda r:r['model_sha256'].update(body_only='different'),
            lambda r:r['groups']['test/argument_aliases'][0]['layout'].update(body_target_start=0),
            lambda r:r['groups']['test/argument_aliases'][0]['models']['body_only']['metrics'].update(cases_all_argument_bytes_correct=0),
            lambda r:r['groups']['test/argument_aliases'][0]['models']['body_only']['prefix_check'].update(consistent=False),
            lambda r:r['groups']['test/argument_aliases'][0]['models']['body_only']['cached_assisted'].update(exact_response=True),
            lambda r:r['groups']['test/argument_aliases'].pop(),
        ):
            report=copy.deepcopy(baseline);mutate(report)
            with self.assertRaises(ValueError):audit(report,DATASET,sources)
