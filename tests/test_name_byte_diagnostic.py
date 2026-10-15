import copy
import importlib.util
import json
from pathlib import Path
import unittest

from scripts import diagnose_name_bytes as diagnostic

ROOT=Path(__file__).resolve().parents[1]
DATASET=ROOT/'evaluation/instruction-v3'


class NameByteTests(unittest.TestCase):
    def test_utf8_target_offsets_and_fixed_scaffold(self):
        prefix,target=diagnostic.plan({'function':'føø','prompt':'Implement føø(x).'})
        self.assertEqual(target,'føø'.encode())
        self.assertTrue(prefix.endswith(b'\n[AI]:\ndef '))
        self.assertNotIn(b'def '+target,prefix)
        with self.assertRaises(ValueError):
            diagnostic.plan({'function':'foo','prompt':'foo'+'x'*256})

    def test_conditional_name_accuracy_is_separate_from_greedy_decode(self):
        target=b'nc_ab12'
        m=diagnostic.metrics(target,list(target),[0.5]*len(target),list(b'wrong('))
        self.assertEqual(m['all_name_bytes_correct'],1)
        self.assertEqual(m['suffix_correct'],4)
        self.assertEqual(m['scaffolded_exact_name'],0)
        with self.assertRaises(ValueError):diagnostic.metrics(target,[0],[0.5],[])
        with self.assertRaises(ValueError):diagnostic.metrics(target,list(target),[float('nan')]*len(target),[])

    def test_training_views_and_same_suffix_prefix_control(self):
        groups=diagnostic.build(DATASET)
        self.assertEqual(sum(map(len,groups.values())),180)
        for split in ('validation','test'):
            for original,control in zip(groups[split+'/renamed'],groups[split+'/matched_prefix']):
                self.assertTrue(original['function'].startswith('ev_'))
                self.assertEqual(control['function'],'nc_'+original['function'][3:])
                self.assertEqual(len(diagnostic.plan(original)[0]),len(diagnostic.plan(control)[0]))
        for policy in ('fixed','resampled'):
            self.assertEqual(len(groups['train/'+policy]),16)
        self.assertEqual([r['id'] for r in groups['train/fixed']],
                         [r['id'] for r in groups['train/resampled']])

    def fixture(self):
        reports={p:json.loads((ROOT/'docs/evidence/resampled-name-training'/f'{p}.json').read_bytes())
                 for p in ('fixed','resampled')}
        groups={}
        for group,rows in diagnostic.build(DATASET).items():
            cases=[]
            for index,r in enumerate(rows):
                _,target=diagnostic.plan(r)
                models={}
                for name in diagnostic.MODELS:
                    predictions=list(target);probabilities=[0.5]*len(target);greedy=list(target+b'(')
                    models[name]={'predictions':predictions,'target_probabilities':probabilities,
                                  'scaffolded_bytes':greedy,'metrics':diagnostic.metrics(target,predictions,probabilities,greedy),
                                  'prefix_check':{'count':len(target),'consistent':True,'max_logit_delta':0.0},
                                  'cached_free':diagnostic.cached(reports,group,index,name)}
                cases.append({**{k:r[k] for k in ('id','prompt','function')},'index':index,'models':models})
            groups[group]=cases
        report={'training_run_id':diagnostic.TRAINING_RUN,'source_reports':diagnostic.REPORT_HASHES,
                'model_sha256':{'initial':diagnostic.training.SOURCE_MODEL_ID[7:],
                                **{p:r['model_sha256'] for p,r in reports.items()}},
                'mode':diagnostic.MODE,'groups':groups,
                'summary':{g:diagnostic.summarize(c) for g,c in groups.items()}}
        return report,reports

    def test_audit_rejects_raw_predictions_caches_and_prefix_tampering(self):
        report,reports=self.fixture()
        self.assertTrue(diagnostic.audit(report,DATASET,reports)['valid'])
        for mutate in (
                lambda r:r['groups']['test/renamed'][0]['models']['fixed']['predictions'].__setitem__(0,0),
                lambda r:r['groups']['test/original'][0]['models']['initial'].update(cached_free=None),
                lambda r:r['groups']['train/fixed'][0]['models']['fixed']['prefix_check'].update(count=0),
                lambda r:r.update(mode='autonomous'),
                lambda r:r['summary']['train/original']['fixed'].update(cases=999)):
            changed=copy.deepcopy(report);mutate(changed)
            with self.assertRaises(ValueError):diagnostic.audit(changed,DATASET,reports)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required for actual causal inference')
    def test_real_transformer_prefill_matches_truncated_prefixes(self):
        import torch
        from train_nppai import NppAIModel
        torch.manual_seed(42)
        model=NppAIModel(256,8,16,1,256).eval()
        with torch.no_grad():
            result=diagnostic.infer(model,b'[USER]: Define foo(x).\n[AI]:\ndef ',b'foo')
        self.assertEqual(result['prefix_check']['count'],3)
        self.assertTrue(result['prefix_check']['consistent'])
        self.assertEqual(len(result['predictions']),3)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required for noncausal rejection')
    def test_future_dependent_logits_are_rejected(self):
        import torch
        class Noncausal(torch.nn.Module):
            def forward(self,x):
                out=torch.zeros((1,x.shape[1],256))
                out[:,:,1]=x.sum()/1000
                return out
        with torch.no_grad(),self.assertRaisesRegex(RuntimeError,'causal prefix'):
            diagnostic.infer(Noncausal(),b'def ',b'foo')


if __name__=='__main__':unittest.main()
