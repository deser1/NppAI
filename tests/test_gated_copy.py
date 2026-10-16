import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import train_gated_copy as experiment

ROOT=Path(__file__).resolve().parents[1]
DATASET=ROOT/'evaluation/instruction-v3'
HAS_TORCH=bool(importlib.util.find_spec('torch'))


class GatedCopyContracts(unittest.TestCase):
    def test_training_schedule_does_not_open_independent_probes(self):
        original=experiment.base.read_split;calls=[]
        def read(root,split):calls.append(split);return original(root,split)
        with patch.object(experiment.base,'read_split',side_effect=read):
            rows=experiment.base.training_data(DATASET)
            meta=experiment.base.schedule_evidence(rows,'resampled')
        self.assertEqual(calls,['train'])
        self.assertEqual(meta['draws'],{'original':2048,'renamed':2048})

    def fixture(self):
        groups=experiment.aligned.development(DATASET);old=experiment.aligned.control_report();phases={}
        for stage in ('before','after'):
            result={}
            for group,rows in groups.items():
                cases=[]
                for index,r in enumerate(rows):
                    output=old[stage][group][index]['output'] if group in old[stage] else r['answer']
                    item={'id':r['id'],'index':index,'output':output,'metrics':experiment.base.score(r,output)}
                    if stage=='after':
                        _,target=experiment.diagnostic.plan(r);pred=list(target);probs=[0.5]*len(target);greedy=list(target+b'(')
                        item['name_bytes']={'predictions':pred,'target_probabilities':probs,'scaffolded_bytes':greedy,
                            'metrics':experiment.diagnostic.metrics(target,pred,probs,greedy),
                            'prefix_check':{'count':len(target),'consistent':True,'max_logit_delta':0}}
                    cases.append(item)
                result[group]=cases
            phases[stage]=result
        return {**phases,'arm':'control','steps':experiment.base.STEPS,'head_seed':experiment.HEAD_SEED,
            'initial_model_id':experiment.base.SOURCE_MODEL_ID,'optimizer':experiment.base.OPTIMIZER,
            'schedule':experiment.base.schedule_evidence(experiment.base.training_data(DATASET),'resampled'),
            'development_sha256':experiment.base.digest(experiment.base.encoded(groups)),
            'checkpoint_sha256':experiment.aligned.CONTROL_MODEL_SHA256,'exported_parameters_equal':True,
            'summary':{s:experiment.aligned.summaries(v) for s,v in phases.items()},
            'retention':experiment.base.retention(phases['before'],phases['after']),
            'historical_checkpoint_equal':True,'loss_endpoints':[1.0,0.5],'runtime':{},'limits':'fixture'}

    def test_raw_report_audit_rejects_wrong_outputs_budgets_and_byte_scores(self):
        report=self.fixture();self.assertTrue(experiment.audit(report,DATASET)['valid'])
        for mutate in (
            lambda r:r.update(head_seed=0),lambda r:r['schedule'].update(response_targets=0),
            lambda r:r['after']['test/original'][0].update(output='tampered'),
            lambda r:r['after']['test/renamed'][0]['name_bytes']['predictions'].__setitem__(0,0)):
            changed=copy.deepcopy(report);mutate(changed)
            with self.assertRaises(ValueError):experiment.audit(changed,DATASET)


@unittest.skipUnless(HAS_TORCH,'PyTorch required for actual pointer model and format tests')
class GatedCopyTorchTests(unittest.TestCase):
    def model(self):
        import torch
        from train_nppai import NppAIModel
        from scripts.gated_copy_model import GatedCopyModel
        torch.manual_seed(42)
        return GatedCopyModel(NppAIModel(256,8,16,1,256)).eval()

    def test_duplicate_aggregation_empty_source_normalization_and_eos_path(self):
        import torch
        from scripts.gated_copy_model import aggregate
        self.assertTrue(torch.equal(aggregate(torch.tensor([[[0.2,0.3,0.5]]]),torch.tensor([[65,65,66]]))[0,0,65:67],torch.tensor([0.5,0.5])))
        model=self.model();tokens=torch.tensor([[0,65,65,66]])
        mask=torch.tensor([[True,True,True,False]])
        logits,pointer,gate,attention,mixture=model.components(tokens,mask)
        self.assertTrue(torch.allclose(mixture.sum(-1),torch.ones(1,4),atol=1e-6))
        self.assertEqual(float(pointer[:,:,0].abs().sum()),0)
        self.assertEqual(float(attention[:,:,3].abs().sum()),0)
        self.assertTrue(torch.allclose(mixture[:,:,0],(1-gate[:,:,0])*logits.softmax(-1)[:,:,0]))
        empty=model.components(tokens,torch.zeros_like(mask))
        self.assertTrue(torch.equal(empty[-1],empty[0].softmax(-1)))
        self.assertTrue(torch.isfinite(model(tokens,torch.zeros_like(mask))).all())

    def test_utf8_source_only_mask_future_causality_and_unchanged_base_logits(self):
        import torch
        from scripts.gated_copy_model import source_mask,source_bounds,PromptAdapter
        from scripts.diagnose_name_bytes import infer
        model=self.model();prompt='Copy føø(x).'
        raw=('[USER]: '+prompt+'\n[AI]:\ndef føø(x):').encode();tokens=torch.tensor([list(raw)])
        mask=source_mask(tokens,prompt);left,right=source_bounds(prompt)
        self.assertEqual(int(mask.sum()),len(prompt.encode()))
        with torch.no_grad():
            parts=model.components(tokens,mask)
            self.assertTrue(torch.equal(parts[0],model.base(tokens)))
            self.assertEqual(float(parts[3][:,:,:left].abs().sum()),0)
            self.assertEqual(float(parts[3][:,:,right:].abs().sum()),0)
            changed=tokens.clone();changed[:,-1]=90
            self.assertTrue(torch.allclose(model(tokens,mask)[:,:-1],model(changed,mask)[:,:-1],atol=1e-5))
            check=infer(PromptAdapter(model,prompt),('[USER]: '+prompt+'\n[AI]:\ndef ').encode(),'føø'.encode())
        self.assertTrue(check['prefix_check']['consistent'])

    def test_response_nll_reaches_pointer_and_gate_gradients(self):
        import torch
        import torch.nn.functional as F
        from scripts.gated_copy_model import source_mask
        model=self.model();prompt='Implement copy_7(x).'
        r={'prompt':prompt,'answer':'def copy_7(x):\n    return x\n'}
        x,y=experiment.base.sample(r);x=x[None,:];y=y[None,:]
        F.nll_loss(model(x,source_mask(x,prompt)).reshape(-1,256),y.reshape(-1),ignore_index=-100).backward()
        for parameter in (model.copy_q.weight,model.copy_k.weight,model.gate.weight,model.gate.bias):
            self.assertTrue(torch.isfinite(parameter.grad).all());self.assertGreater(float(parameter.grad.abs().sum()),0)

    def test_versioned_roundtrip_rejects_legacy_loader_and_tampered_head(self):
        import torch
        from scripts.gated_copy_model import export_bundle,load_bundle,bounded_zip,zip_bytes
        from scripts.check_dense_runtime_parity import load_weights
        model=self.model()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.nppaicopy';export_bundle(model,path,experiment.base.SOURCE_MODEL_ID)
            loaded,manifest=load_bundle(path)
            self.assertFalse(manifest['native_compatible'])
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.parameters(),loaded.parameters())))
            x=torch.tensor([[65,66,65]]);mask=torch.tensor([[True,True,False]])
            with torch.no_grad():self.assertTrue(torch.allclose(model(x,mask),loaded(x,mask),atol=1e-6))
            with self.assertRaises(ValueError):load_weights(path)
            entries=bounded_zip(path.read_bytes(),('base.nppai','head.npz','manifest.json'))
            entries['head.npz']=b'invalid';path.write_bytes(zip_bytes(entries))
            with self.assertRaisesRegex(ValueError,'digest'):load_bundle(path)

    def test_pickle_backed_head_arrays_are_rejected_even_with_valid_digest(self):
        import numpy as np
        from scripts.gated_copy_model import export_bundle,load_bundle,bounded_zip,zip_bytes
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.nppaicopy';export_bundle(self.model(),path,experiment.base.SOURCE_MODEL_ID)
            outer=bounded_zip(path.read_bytes(),('base.nppai','head.npz','manifest.json'))
            inner=bounded_zip(outer['head.npz'],('copy_q.npy','copy_k.npy','gate_weight.npy','gate_bias.npy'))
            stream=io.BytesIO();np.save(stream,np.array([{'fixture':'object'}],dtype=object),allow_pickle=True)
            inner['copy_q.npy']=stream.getvalue();outer['head.npz']=zip_bytes(inner)
            manifest=json.loads(outer['manifest.json']);manifest['head_sha256']=experiment.base.digest(outer['head.npz'])
            outer['manifest.json']=json.dumps(manifest).encode();path.write_bytes(zip_bytes(outer))
            with self.assertRaises(ValueError):load_bundle(path)


if __name__=='__main__':unittest.main()
