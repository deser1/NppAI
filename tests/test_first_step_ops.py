import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.first_step_ops import TensorStore,first_difference,compare,encoded


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

    def test_aggregation_retains_boundary_evidence_and_rejects_different_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for profile in ('historical-v1','portable-cpu-v1'):
                for replica in range(1,5):
                    p=root/f'first-step-{profile}-{replica}';(p/'0').mkdir(parents=True)
                    report={'trace_preserves_untraced_step':True,'trace_repeat':{'identical':True},
                        'profile':profile,'source_model_id':'same','batch_sha256':'same','x_sha256':'same','y_sha256':'same',
                        'reference':{'initial_parameters':'same'},'environment':{'cpu':{'model name':'fixture','vendor_id':'fixture'}}}
                    (p/'report.json').write_text(json.dumps(report))
                    event={'phase':'forward','op':'aten._softmax.default','schema':'softmax','args':['same'],'kwargs':{},
                        'output':'other' if replica==4 else 'same'}
                    (p/'0/events.json').write_text(json.dumps([event]))
            with patch('builtins.print'):compare(root)
            result=json.loads((root/'comparison.json').read_text())
            pair=result['historical-v1']['pairs'][2]
            self.assertEqual(pair['replicas'],[1,4]);self.assertTrue(pair['same_call']);self.assertTrue(pair['same_inputs'])
            p=root/'first-step-portable-cpu-v1-4/report.json';report=json.loads(p.read_text());report['x_sha256']='changed';p.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError,'source or batch differs'):compare(root)


    def test_retained_operator_evidence_has_verified_reports_and_first_difference(self):
        root=Path(__file__).resolve().parents[1]/'docs/evidence/first-step-operators'
        for prefix in ('','validation-'):
            manifest=json.loads((root/(prefix+'manifest.json')).read_bytes())
            comparison=json.loads((root/(prefix+'comparison.json')).read_bytes());digests={}
            for record in manifest['records']:
                raw=(root/record['report_file']).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),record['files']['report.json']['sha256'])
                report=json.loads(raw);self.assertTrue(all(item==report['reference'] for item in report['traced']))
                self.assertTrue(report['trace_preserves_untraced_step']);self.assertTrue(report['trace_repeat']['identical'])
                rows=[]
                for name in record['event_digest_chunks']:rows.extend(json.loads((root/name).read_bytes()))
                self.assertEqual(len(rows),report['trace_repeat']['counts'][0]);digests[record['artifact_name']]=rows
            self.assertEqual(len(digests),8)
            for profile,result in comparison.items():
                for pair in result['pairs']:
                    a,b=[digests[f'first-step-{profile}-{i}'] for i in pair['replicas']]
                    differences=[i for i,(left,right) in enumerate(zip(a,b)) if left!=right]
                    self.assertEqual(not differences,pair['identical'])
                    if differences:
                        self.assertEqual(differences[0],pair['event'])
                        self.assertEqual(hashlib.sha256(encoded(pair['left'])).hexdigest(),a[pair['event']])
                        self.assertEqual(hashlib.sha256(encoded(pair['right'])).hexdigest(),b[pair['event']])
            if not prefix:
                for replica in (1,3):
                    events=json.loads((root/f'historical-v1-{replica}-prefix.json').read_bytes())
                    self.assertEqual([hashlib.sha256(encoded(e)).hexdigest() for e in events],digests[f'first-step-historical-v1-{replica}'][:37])


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
