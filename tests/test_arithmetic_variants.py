import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from scripts.arithmetic_variants import SPECS,agreement,validate_input


class ArithmeticContracts(unittest.TestCase):
    def test_frozen_inputs_reject_size_and_hash_changes(self):
        for spec in SPECS.values():
            with self.assertRaisesRegex(ValueError,'size or digest'):validate_input(b'',spec)
            with self.assertRaisesRegex(ValueError,'size or digest'):validate_input(b'\0'*spec['bytes'],spec)

    def test_complete_step_agreement_requires_all_stages_not_just_weights(self):
        a={'isolated':{'outputs':dict.fromkeys(('softmax_forward','softmax_backward','sqrt'),'same')},
           'complete':{'state':dict.fromkeys(('initial_parameters','logits','loss_hex','raw_gradients','clipped_gradients','updated_parameters'),'same')}}
        b=copy.deepcopy(a);self.assertTrue(agreement(a,b)['complete_step_identical'])
        b['complete']['state']['raw_gradients']='different'
        self.assertFalse(agreement(a,b)['complete_step_identical'])
        self.assertTrue(agreement(a,b)['step']['updated_parameters'])
        self.assertFalse(agreement(a,b)['step']['raw_gradients'])


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required for actual mixed-precision arithmetic')
class RealArithmetic(unittest.TestCase):
    def test_softmax_forward_and_saved_fp32_backward_formula(self):
        import torch
        from scripts.arithmetic_variants import arithmetic_mode
        x=torch.tensor([[1.,2.,float('-inf')],[-1.,0.,3.]])
        g=torch.tensor([[.5,-.25,1.],[-1.,.25,.5]])
        mode=arithmetic_mode('softmax64')
        with mode:
            mode.phase='forward';out=torch.softmax(x,-1)
            mode.phase='backward';back=torch.ops.aten._softmax_backward_data.default(g,out,-1,torch.float32)
        self.assertTrue(torch.equal(out,torch.softmax(x.double(),-1).float()))
        reference=((g.double()-(g.double()*out.double()).sum(-1,keepdim=True))*out.double()).float()
        torch.testing.assert_close(back,reference,rtol=1e-6,atol=1e-7)
        self.assertEqual(float(out[0,2]),0);self.assertEqual(float(back[0,2]),0)
        self.assertGreater(mode.overrides.get('aten._softmax_backward_data.default',0),0)

    def test_sqrt_only_replaced_in_optimizer_and_context_restores_native(self):
        import torch
        from scripts.arithmetic_variants import arithmetic_mode
        x=torch.tensor([0.,1e-20,.1,2.,7.]);native=torch.sqrt(x);mode=arithmetic_mode('sqrt64')
        with mode:
            mode.phase='forward';untouched=torch.sqrt(x)
            self.assertFalse(mode.overrides)
            mode.phase='optimizer';changed=torch.sqrt(x)
        self.assertTrue(torch.equal(untouched,native));self.assertTrue(torch.equal(torch.sqrt(x),native))
        self.assertTrue(torch.equal(changed,x.double().sqrt().float()))
        self.assertEqual(mode.overrides,{'aten.sqrt.default':1})

    def test_native_mode_preserves_complete_step_and_variants_start_identically(self):
        import torch
        from train_nppai import NppAIModel,export_to_bin
        from scripts.check_dense_runtime_parity import load_weights
        from scripts.first_step_ops import step
        from scripts.arithmetic_variants import complete_step
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.nppai';torch.manual_seed(42);export_to_bin(NppAIModel(256,8,16,1,64),path)
            weights=load_weights(path);x=torch.tensor([[65,66,67,68],[68,67,66,65]])
            y=torch.tensor([[-100,67,68,0],[-100,66,65,0]])
            reference=step(weights,x,y,'historical-v1')
            native=complete_step(weights,x,y,'historical-v1','native')
            self.assertEqual(native['state'],reference);self.assertFalse(native['overrides'])
            changed=complete_step(weights,x,y,'historical-v1','both64')
            self.assertEqual(changed['state']['initial_parameters'],reference['initial_parameters'])
            self.assertGreater(changed['overrides'].get('aten.sqrt.default',0),0)
            self.assertGreater(changed['overrides'].get('aten._softmax_backward_data.default',0),0)


if __name__=='__main__':unittest.main()
