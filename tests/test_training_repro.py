import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch
from scripts.training_repro import compare_traces,compare_replicas,configure


class TrainingReproTests(unittest.TestCase):
    def test_trace_reports_first_divergence_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            left=Path(directory)/'left';right=Path(directory)/'right'
            rows=[{'step':i,'parameters':str(i)} for i in range(3)]
            def write(path,items):path.write_text(''.join(json.dumps(x)+'\n' for x in items))
            write(left,rows);write(right,rows)
            self.assertTrue(compare_traces(left,right)['identical'])
            write(right,rows[:2]);self.assertEqual(compare_traces(left,right)['first_different_step'],2)
            write(right,[rows[0],{'step':1,'parameters':'different'},rows[2]])
            self.assertEqual(compare_traces(left,right)['first_different_step'],1)

    def test_portable_profile_requires_environment_before_configuring_torch(self):
        torch=MagicMock()
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(ValueError,'MKL_CBWR'):configure(torch,'portable-cpu-v1')
        torch.set_num_threads.assert_not_called()
        with patch.dict('os.environ',{'MKL_CBWR':'COMPATIBLE','ATEN_CPU_CAPABILITY':'default','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},clear=True):
            configure(torch,'portable-cpu-v1')
        torch.set_num_threads.assert_called_once_with(1)
        torch.use_deterministic_algorithms.assert_called_once_with(True)
        self.assertFalse(torch.backends.mkldnn.enabled)

    def test_pair_rejects_nonidentical_historical_checkpoint_before_copy(self):
        from scripts import train_gated_copy as experiment
        with patch.object(experiment,'run',return_value=({'historical_checkpoint_equal':False},None)) as run:
            with self.assertRaisesRegex(RuntimeError,'historical control checkpoint differs'):
                experiment.run_pair(None,None,Path('unused'),None,None)
        self.assertEqual(run.call_count,1)

    def test_portability_failure_is_retained_and_not_converted_to_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for profile in ('historical-v1','portable-cpu-v1'):
                for replica in (1,2):
                    p=root/f'training-repro-{profile}-{replica}';(p/'0').mkdir(parents=True)
                    (p/'comparison.json').write_text(json.dumps({'same_host_identical':True,'outcomes':[],
                        'checkpoints':['same' if profile=='historical-v1' else str(replica)]}))
                    (p/'0/trace.jsonl').write_text(json.dumps({'step':0,'parameters':'same'})+'\n')
                    (p/'0/report.json').write_text(json.dumps({'before':{},'after':{}}))
                    (p/'0/environment.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'portable profile did not reproduce'):
                compare_replicas(root)
            result=json.loads((root/'comparison.json').read_text())
            self.assertTrue(result['historical-v1']['checkpoint_identical'])
            self.assertFalse(result['portable-cpu-v1']['checkpoint_identical'])

    def test_retained_traces_reconstruct_verified_artifact_bytes(self):
        root=Path(__file__).resolve().parents[1]/'docs/evidence/training-reproducibility'
        manifest=json.loads((root/'manifest.json').read_text());initial=set()
        self.assertEqual(len(manifest['records']),4)
        for record in manifest['records']:
            raw=b''.join((root/name).read_bytes() for name in record['trace_chunks'])
            rows=[json.loads(line) for line in raw.splitlines()]
            self.assertEqual([r['step'] for r in rows],list(range(1025)))
            initial.add(rows[0]['parameters'])
            for repetition in (0,1):
                expected=record['files'][str(repetition)+'/trace.jsonl']
                self.assertEqual(len(raw),expected['bytes'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(),expected['sha256'])
        self.assertEqual(len(initial),1)


if __name__=='__main__':unittest.main()
