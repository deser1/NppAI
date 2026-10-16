import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.portable_full_training import first_step_gate,trace_difference,compare,PROFILE,VARIANTS


class FullTrainingContracts(unittest.TestCase):
    def test_fixed_gate_requires_unmodified_cross_vendor_first_step_evidence(self):
        root=Path(__file__).resolve().parents[1]/'docs/evidence/both64-trace'
        result=first_step_gate(root)
        self.assertEqual([r['cross_vendor_pairs'] for r in result],[4,3])
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            for file in ('comparison.json','validation-comparison.json'):(p/file).write_bytes((root/file).read_bytes())
            (p/'comparison.json').write_bytes((p/'comparison.json').read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'evidence differs'):first_step_gate(p)

    def test_trace_difference_detects_intermediate_mismatch_even_with_same_final_weights(self):
        a=[{'step':0,'parameters':'start'},{'step':1,'parameters':'a'},{'step':2,'parameters':'final'}]
        b=copy.deepcopy(a);b[1]['parameters']='b'
        self.assertEqual(trace_difference(a,b)['first_different_step'],1)
        self.assertTrue(trace_difference(a,a)['identical'])
        self.assertFalse(trace_difference(a,a[:-1])['identical'])

    def test_aggregation_validates_raw_repeats_and_input_exposure(self):
        from scripts.first_step_ops import encoded
        from scripts.arithmetic_variants import digest
        with tempfile.TemporaryDirectory() as directory,patch('scripts.portable_full_training.base.STEPS',2):
            root=Path(directory)
            for i in range(1,5):
                p=root/f'portable-full-{i}';arms={}
                for variant in VARIANTS:
                    rows=[{'step':0,'parameters':'same'},{'step':1,'parameters':'middle','inputs':'same'},{'step':2,'parameters':'final','inputs':'same'}]
                    raw=b''.join(encoded(r) for r in rows)
                    val={'trace_sha256':digest(raw),'initial_parameters':'same','final_parameters':'final','checkpoint_sha256':digest(b'checkpoint')}
                    arms[variant]={'repeats':[val,val],'same_host_identical':True}
                    for rep in range(2):
                        target=p/variant/str(rep);target.mkdir(parents=True);(target/'trace.jsonl').write_bytes(raw);(target/'experimental.nppai').write_bytes(b'checkpoint')
                report={'version':1,'profile':PROFILE,'steps':2,'promotion':False,'arms':arms,
                    'source_model_id':'same','gate':'same','schedule':'same','training_sha256':'same','optimizer':'same',
                    'environment':{'cpu':{'model name':'fixture','vendor_id':'AMD' if i%2 else 'Intel'}},'limits':'fixture'}
                (p/'report.json').write_bytes(encoded(report))
            compare(root);result=json.loads((root/'comparison.json').read_bytes())
            self.assertEqual(result['coverage']['cross_vendor_pairs'],4)
            self.assertTrue(result['summary']['both64']['all_training_traces_and_checkpoints_identical'])
            p=root/'portable-full-4/report.json';r=json.loads(p.read_bytes());r['training_sha256']='changed';p.write_bytes(encoded(r))
            with self.assertRaisesRegex(ValueError,'exposure'):compare(root)
            r['training_sha256']='same';p.write_bytes(encoded(r))
            (root/'portable-full-1/native/1/trace.jsonl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'trace repeat'):compare(root)


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required')
class ActualFullTraining(unittest.TestCase):
    def test_actual_multi_update_training_repeats_and_preserves_fixed_first_step(self):
        import torch
        from train_nppai import NppAIModel,export_to_bin
        from scripts.check_dense_runtime_parity import load_weights
        from scripts.portable_full_training import train
        from scripts.arithmetic_variants import complete_step
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.nppai'
            torch.manual_seed(42);export_to_bin(NppAIModel(256,8,16,1,64),source);weights=load_weights(source)
            x=torch.tensor([[65,66,67,68],[68,67,66,65]]);y=torch.tensor([[-100,67,68,0],[-100,66,65,0]])
            batches=[(x,y)]*3
            for variant in VARIANTS:
                a=train(weights,batches,root/variant/'a',variant);b=train(weights,batches,root/variant/'b',variant)
                self.assertEqual(a,b);self.assertEqual(a['steps'],3)
                self.assertEqual(a['first_step'],complete_step(weights,x,y,PROFILE,variant))
                for file in ('trace.jsonl','experimental.nppai'):
                    self.assertEqual((root/variant/'a'/file).read_bytes(),(root/variant/'b'/file).read_bytes())
                rows=(root/variant/'a'/'trace.jsonl').read_text().splitlines();self.assertEqual(len(rows),4)


if __name__=='__main__':unittest.main()
