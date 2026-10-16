import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from scripts.both64_trace import saved_difference,compare


class TraceContracts(unittest.TestCase):
    def test_saved_difference_includes_value_and_save_location(self):
        a=[{'index':0,'phase':'forward','after_events':2,'value':'a'}]
        self.assertTrue(saved_difference(a,a)['identical'])
        result=saved_difference(a,[{**a[0],'value':'b'}])
        self.assertEqual(result['index'],0);self.assertEqual(result['left'],a[0])
        self.assertFalse(saved_difference(a,[])['identical'])

    def test_compare_rejects_changed_exposure_or_unreliable_instrumentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for profile in ('historical-v1','portable-cpu-v1'):
                for i in range(1,5):
                    p=root/f'both64-report-{profile}-{i}';(p/'0').mkdir(parents=True)
                    ref={'state':{'initial_parameters':'same','raw_gradients':str(i)}}
                    report={'version':1,'variant':'both64','profile':profile,'steps':1,'promotion':False,
                        'source_model_id':'same','batch_sha256':'same','x_sha256':'same','y_sha256':'same',
                        'reference':ref,'traced':[ref,ref],'trace_preserves_untraced_step':True,
                        'trace_repeat':{'identical':True},'saved_repeat':True,'saved_counts':[1,1],
                        'environment':{'cpu':{'vendor_id':'AMD' if i==1 else 'Intel','model name':'fixture'}}}
                    (p/'report.json').write_text(json.dumps(report))
                    (p/'0/events.json').write_text(json.dumps([{'phase':'backward','op':'test','args':'same','output':str(i)}]))
                    (p/'0/saved.json').write_text(json.dumps([{'value':str(i)}]))
            compare(root);result=json.loads((root/'comparison.json').read_bytes())
            pair=result['historical-v1']['pairs'][0]
            self.assertTrue(pair['cross_vendor']);self.assertTrue(pair['first_operator_difference']['same_inputs'])
            self.assertFalse(pair['first_saved_difference']['identical'])
            p=root/'both64-report-portable-cpu-v1-4/report.json';r=json.loads(p.read_bytes());r['x_sha256']='different';p.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'exposure'):compare(root)
            r['x_sha256']='same';r['traced'][0]={'changed':True};p.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'unreliable'):compare(root)


    def test_retained_evidence_binds_reports_saved_tensors_and_first_operator_difference(self):
        import hashlib
        from scripts.first_step_ops import encoded
        root=Path(__file__).resolve().parents[1]/'docs/evidence/both64-trace'
        manifest=json.loads((root/'manifest.json').read_bytes());comparison=json.loads((root/'comparison.json').read_bytes())
        self.assertEqual(len(manifest['records']),8);events={};saved={};reports={}
        for record in manifest['records']:
            name=record['artifact_name'];raw=(root/record['report_file']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),record['files']['report.json']['sha256'])
            report=json.loads(raw);reports[name]=report
            self.assertEqual(report['traced'],[report['reference']]*2)
            self.assertTrue(report['trace_preserves_untraced_step']);self.assertTrue(report['saved_repeat'])
            self.assertEqual(report['environment']['git_revision'],manifest['checkout_merge_sha'])
            rows=[]
            for file in record['event_digest_chunks']:rows.extend(json.loads((root/file).read_bytes()))
            events[name]=rows;self.assertEqual(len(rows),report['trace_repeat']['counts'][0])
            raw=(root/record['saved_file']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),record['files']['0/saved.json']['sha256'])
            saved[name]=json.loads(raw);self.assertEqual([len(saved[name])]*2,report['saved_counts'])
        for profile,result in comparison.items():
            self.assertEqual(result['reports'],[reports[f'both64-report-{profile}-{i}'] for i in range(1,5)])
            for pair in result['pairs']:
                a,b=[f'both64-report-{profile}-{i}' for i in pair['replicas']]
                differences=[i for i,(x,y) in enumerate(zip(events[a],events[b])) if x!=y]
                finding=pair['first_operator_difference'];self.assertEqual(finding['identical'],not differences)
                if differences:
                    self.assertEqual(finding['event'],differences[0])
                    self.assertEqual(hashlib.sha256(encoded(finding['left'])).hexdigest(),events[a][differences[0]])
                    self.assertEqual(hashlib.sha256(encoded(finding['right'])).hexdigest(),events[b][differences[0]])
                self.assertEqual(pair['first_saved_difference'],saved_difference(saved[a],saved[b]))
                self.assertEqual(pair['step'],{k:reports[a]['reference']['state'][k]==reports[b]['reference']['state'][k] for k in reports[a]['reference']['state']})
        for replica in (1,4):
            prefix=json.loads((root/f'historical-v1-{replica}-prefix.json').read_bytes())
            self.assertEqual([hashlib.sha256(encoded(e)).hexdigest() for e in prefix],events[f'both64-report-historical-v1-{replica}'][:len(prefix)])


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required')
class RealTrace(unittest.TestCase):
    def test_both64_trace_and_saved_hooks_preserve_full_step_and_effective_softmax(self):
        import torch
        from train_nppai import NppAIModel,export_to_bin
        from scripts.check_dense_runtime_parity import load_weights
        from scripts.arithmetic_variants import complete_step
        from scripts.both64_trace import traced_step
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'source.nppai'
            export_to_bin(NppAIModel(256,8,16,1,64),path);weights=load_weights(path)
            x=torch.tensor([[65,66,67,68],[68,67,66,65]]);y=torch.tensor([[-100,67,68,0],[-100,66,65,0]])
            reference=complete_step(weights,x,y,'historical-v1','both64')
            for i in range(2):
                p=root/str(i);p.mkdir();observed=traced_step(weights,x,y,'historical-v1',p)
                self.assertEqual(reference,observed)
                events=json.loads((p/'events.json').read_bytes());saved=json.loads((p/'saved.json').read_bytes())
                self.assertTrue(saved);self.assertTrue(any(e['phase']=='backward' for e in events))
                soft=next(e for e in events if e['op']=='aten._softmax.default')
                record=soft['output'];self.assertEqual(record['dtype'],'torch.float32')
                import numpy as np
                arg=soft['args']['items'][0]
                raw=(p/'tensors'/(arg['tensor']+'.bin')).read_bytes()
                inp=torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy().reshape(arg['shape']))
                self.assertEqual((p/'tensors'/(record['tensor']+'.bin')).read_bytes(),torch.softmax(inp.double(),-1).float().numpy().tobytes())
            self.assertEqual((root/'0/events.json').read_bytes(),(root/'1/events.json').read_bytes())
            self.assertEqual((root/'0/saved.json').read_bytes(),(root/'1/saved.json').read_bytes())


if __name__=='__main__':unittest.main()
