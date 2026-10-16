import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.portable_full_training import first_step_gate,trace_difference,compare,PROFILE,VARIANTS


class FullTrainingContracts(unittest.TestCase):
    def test_aggregation_cli_imports_with_standard_library_only(self):
        import subprocess
        import sys
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,'-S',str(root/'scripts/portable_full_training.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

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
        with tempfile.TemporaryDirectory() as directory,patch('scripts.portable_full_training.STEPS',2):
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


    def test_retained_full_traces_reconstruct_exact_bytes_and_pairwise_results(self):
        from scripts.arithmetic_variants import digest
        from scripts.first_step_ops import encoded
        root=Path(__file__).resolve().parents[1]/'docs/evidence/portable-full-training'
        manifest=json.loads((root/'manifest.json').read_bytes());comparison=json.loads((root/'comparison.json').read_bytes())
        self.assertEqual(digest((root/'comparison.json').read_bytes()),manifest['comparison_sha256'])
        traces={}
        for sha,record in manifest['traces'].items():
            rows=[]
            for file in record['chunks']:rows.extend(json.loads((root/file).read_bytes()))
            raw=b''.join(encoded(row) for row in rows)
            self.assertEqual(digest(raw),sha);self.assertEqual(len(raw),record['bytes'])
            self.assertEqual([row['step'] for row in rows],list(range(1025)));traces[sha]=rows
        self.assertEqual(len(manifest['records']),4);reports={}
        for record in manifest['records']:
            raw=(root/record['report_file']).read_bytes();self.assertEqual(digest(raw),record['report_sha256'])
            report=json.loads(raw);reports[record['artifact_name']]=report
            raw=(root/record['preflight_file']).read_bytes();self.assertEqual(digest(raw),record['preflight_sha256'])
            preflight=json.loads(raw);self.assertTrue(preflight['matches_frozen_candidate'])
            self.assertTrue(all(preflight['first_step']==g['reference'] for g in report['gate']))
            for variant,arm in report['arms'].items():
                self.assertTrue(arm['same_host_identical']);self.assertEqual(arm['repeats'][0],arm['repeats'][1])
                r=arm['repeats'][0];self.assertEqual(r['trace_sha256'],record['arms'][variant]['trace_sha256'])
                self.assertEqual(traces[r['trace_sha256']][0]['parameters'],r['initial_parameters'])
                self.assertEqual(traces[r['trace_sha256']][-1]['parameters'],r['final_parameters'])
        self.assertEqual(comparison['reports'],[reports[f'portable-full-{i}'] for i in range(1,5)])
        for pair in comparison['pairs']:
            a,b=[reports[f'portable-full-{i}'] for i in pair['replicas']]
            for variant,result in pair['arms'].items():
                x,y=[r['arms'][variant]['repeats'][0] for r in (a,b)]
                self.assertEqual(result['trace'],trace_difference(traces[x['trace_sha256']],traces[y['trace_sha256']]))
                self.assertEqual(result['checkpoint_identical'],x['checkpoint_sha256']==y['checkpoint_sha256'])


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
