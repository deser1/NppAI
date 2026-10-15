import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import aligned_name_copy as experiment

ROOT=Path(__file__).resolve().parents[1]
DATASET=ROOT/'evaluation/instruction-v3'
HAS_TORCH=bool(importlib.util.find_spec('torch'))


class AlignedCopyTests(unittest.TestCase):
    def record(self):
        return {'id':'fixture','function':'føø','prompt':'Żółć: Implement føø(x).',
                'answer':'def føø(x):\n    return x\n'}

    def test_source_and_shifted_target_offsets_are_utf8_bytes(self):
        r=self.record();prefix,raw=experiment.base.frame(r);pairs=experiment.alignment(r)
        self.assertEqual(len(pairs),len('føø'.encode()))
        self.assertEqual(pairs[0][0],len(prefix)+3)
        self.assertEqual(pairs[0][1],len('[USER]: Żółć: Implement '.encode()))
        for query,source in pairs:
            self.assertLess(source,query)
            self.assertEqual(raw[source],raw[query+1])
        for prompt in ('Implement x(x).','Implement føø(x) and føø(y).'):
            with self.assertRaises(ValueError):experiment.alignment({**r,'prompt':prompt})

    def test_full_training_schedule_has_valid_alignment_without_development_reads(self):
        rows=experiment.base.training_data(DATASET)
        count=0
        for batch in experiment.base.schedule(rows,'resampled'):
            for _,r in batch:count+=len(experiment.alignment(r))
        self.assertGreater(count,0)
        self.assertEqual(sum(map(len,experiment.development(DATASET).values())),164)

    @unittest.skipUnless(HAS_TORCH,'PyTorch required for real attention gradients')
    def test_attention_loss_backpropagates_and_masks_future_keys(self):
        import torch
        torch.manual_seed(42)
        q=torch.randn(1,9,4,requires_grad=True);k=torch.randn(1,9,4,requires_grad=True)
        loss=experiment.attention_loss({'wq':q,'wk':k},[[(7,2)]],[9]);loss.backward()
        self.assertGreater(float(q.grad[0,7].abs().sum()),0)
        self.assertEqual(float(q.grad[0,:7].abs().sum()),0)
        self.assertEqual(float(k.grad[0,8].abs().sum()),0)
        self.assertGreater(float(k.grad[0,2].abs().sum()),0)
        with self.assertRaises(ValueError):experiment.attention_loss({'wq':q,'wk':k},[[(2,7)]],[9])

    @unittest.skipUnless(HAS_TORCH,'PyTorch required for actual forward and hook cleanup')
    def test_hooks_preserve_logits_export_parameters_and_remove_on_failure(self):
        import torch
        from train_nppai import NppAIModel
        torch.manual_seed(42);model=NppAIModel(256,8,16,1,256).eval()
        x,_=experiment.base.sample(self.record());x=x[None,:]
        baseline=model(x);names=list(model.state_dict())
        with experiment.capture_qk(model) as captured:observed=model(x)
        self.assertTrue(torch.equal(baseline,observed));self.assertEqual(names,list(model.state_dict()))
        loss=experiment.attention_loss(captured,[experiment.alignment(self.record())],[x.shape[1]])
        loss.backward();self.assertGreater(float(model.layers[-1].wq.weight.grad.abs().sum()),0)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with experiment.capture_qk(model):raise RuntimeError('fixture')
        self.assertFalse(model.layers[-1].wq._forward_hooks)
        self.assertFalse(model.layers[-1].wk._forward_hooks)

    @unittest.skipUnless(HAS_TORCH,'PyTorch required for exact control gradient equality')
    def test_control_has_the_original_graph_and_identical_gradients(self):
        import torch
        import torch.nn.functional as F
        from train_nppai import NppAIModel
        torch.manual_seed(42);plain=NppAIModel(256,8,16,1,256);control=copy.deepcopy(plain)
        x,y=experiment.base.sample(self.record());x=x[None,:];y=y[None,:]
        expected=F.cross_entropy(plain(x).reshape(-1,256),y.reshape(-1),ignore_index=-100)
        expected.backward()
        with patch.object(experiment,'capture_qk',side_effect=AssertionError('control must not capture')):
            actual,aux=experiment.training_losses(control,x,y,[self.record()],[x.shape[1]],False)
        actual.backward();self.assertIsNone(aux)
        self.assertTrue(torch.equal(expected,actual))
        self.assertTrue(all(torch.equal(a.grad,b.grad) for a,b in zip(plain.parameters(),control.parameters())))

    def fixture(self):
        base=experiment.base;groups=experiment.development(DATASET);old=experiment.control_report()
        phases={}
        for stage in ('before','after'):
            data={}
            for group,refs in groups.items():
                cases=[]
                for index,r in enumerate(refs):
                    output=old[stage][group][index]['output'] if group in old[stage] else r['answer']
                    case={'index':index,'id':r['id'],'output':output,'metrics':base.score(r,output)}
                    if stage=='after':
                        _,target=experiment.diagnostic.plan(r);pred=list(target);probs=[0.5]*len(target);greedy=list(target+b'(')
                        case['name_bytes']={'predictions':pred,'target_probabilities':probs,'scaffolded_bytes':greedy,
                            'metrics':experiment.diagnostic.metrics(target,pred,probs,greedy),
                            'prefix_check':{'count':len(target),'consistent':True,'max_logit_delta':0}}
                    cases.append(case)
                data[group]=cases
            phases[stage]=data
        return {**phases,'arm':'control','steps':experiment.STEPS,'coefficient':0.0,
                'initial_model_id':base.SOURCE_MODEL_ID,'source_run_id':base.SOURCE_RUN,
                'source_report_sha256':base.REPORT_SHA256,'optimizer':base.OPTIMIZER,
                'model_sha256':experiment.CONTROL_MODEL_SHA256,'exported_weights_equal':True,
                'historical_checkpoint_equal':True,
                'runtime':{'fixture':True},
                'development_sha256':base.digest(base.encoded(groups)),
                'schedule':base.schedule_evidence(base.training_data(DATASET),'resampled'),
                'summary':{s:experiment.summaries(d) for s,d in phases.items()},
                'retention':base.retention(phases['before'],phases['after']),
                'loss_endpoints':[{'response_ce':1.0}]*2}

    def test_evidence_audit_rejects_alignment_budget_and_raw_output_tampering(self):
        report=self.fixture();self.assertTrue(experiment.audit(report,DATASET)['valid'])
        for mutate in (
            lambda r:r.update(coefficient=1.0),
            lambda r:r.update(model_sha256='wrong'),
            lambda r:r['after']['test/original'][0].update(output='tampered'),
            lambda r:r['after']['test/matched_prefix'][0]['name_bytes']['predictions'].__setitem__(0,0),
            lambda r:r['schedule'].update(response_targets=0)):
            changed=copy.deepcopy(report);mutate(changed)
            with self.assertRaises(ValueError):experiment.audit(changed,DATASET)

    def test_same_host_pair_requires_exact_control_repeat_and_shared_exposure(self):
        control=self.fixture();repeat=copy.deepcopy(control);aligned=copy.deepcopy(control)
        self.assertTrue(experiment.check_pair(control,repeat,aligned)['same_host_control_repeat_identical'])
        repeat['model_sha256']='different'
        with self.assertRaisesRegex(RuntimeError,'control repeat'):experiment.check_pair(control,repeat,aligned)
        repeat=copy.deepcopy(control);aligned['schedule']['response_targets']=0
        with self.assertRaisesRegex(RuntimeError,'exposure'):experiment.check_pair(control,repeat,aligned)


if __name__=='__main__':unittest.main()
