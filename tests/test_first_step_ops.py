import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from scripts.first_step_ops import TensorStore,first_difference


class OperatorTraceContracts(unittest.TestCase):
    def test_output_difference_is_only_isolated_for_identical_call_and_inputs(self):
        event={'phase':'forward','op':'aten.mm.default','schema':'mm','args':{'x':'same'},'kwargs':{},'output':'a'}
        other={**event,'output':'b'}
        result=first_difference([event],[other]);self.assertTrue(result['same_inputs']);self.assertTrue(result['output_differs'])
        other['args']={'x':'different'}
        self.assertFalse(first_difference([event],[other])['same_inputs'])
        self.assertTrue(first_difference([event],[event])['identical'])
        self.assertFalse(first_difference([event],[])['identical'])

    def test_snapshot_store_deduplicates_and_binds_dtype_and_shape_with_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            store=TensorStore(directory,budget=8)
            a=store.put(b'1234','float32',[1]);b=store.put(b'1234','float32',[1]);self.assertEqual(a,b)
            other=store.put(b'1234','int32',[1]);self.assertNotEqual(a['tensor'],other['tensor'])
            self.assertEqual(store.bytes,8)
            with self.assertRaisesRegex(ValueError,'budget'):store.put(b'1','uint8',[1])
            self.assertEqual((Path(directory)/(a['tensor']+'.bin')).read_bytes(),b'1234')


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required for actual traced autograd')
class RealOperatorTrace(unittest.TestCase):
    def test_recorded_training_step_matches_untraced_step(self):
        import torch
        from train_nppai import NppAIModel,export_to_bin
        from scripts.check_dense_runtime_parity import load_weights
        from scripts.first_step_ops import step
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);weights_path=root/'source.nppai'
            export_to_bin(NppAIModel(256,8,16,1,64),weights_path)
            weights=load_weights(weights_path);x=torch.tensor([[65,66,67,68],[68,67,66,65]])
            y=torch.tensor([[-100,67,68,0],[-100,66,65,0]])
            reference=step(weights,x,y,'historical-v1')
            observed=step(weights,x,y,'historical-v1',root/'trace')
            self.assertEqual(reference,observed)
            events=json.loads((root/'trace/events.json').read_text())
            self.assertEqual(set(e['phase'] for e in events),{'forward','loss','backward','clip','optimizer'})
            self.assertTrue(any(e['op'].startswith('aten.') for e in events))


if __name__=='__main__':unittest.main()
