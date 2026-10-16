import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch
from scripts.training_repro import compare_traces,configure


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


if __name__=='__main__':unittest.main()
