import tempfile
from pathlib import Path
import unittest
import torch
from train_nppai import NppAIModel
from scripts.all_stage_protection import train_all
from scripts.training_repro import state_digest


class AllStageGradientTests(unittest.TestCase):
    def test_all_stage_optimization_changes_only_the_copy(self):
        torch.manual_seed(43);torch.set_num_threads(1)
        initial=NppAIModel(256,4,8,1,1024);before=state_digest(initial)
        record={'prompt':'write f','answer':'def f():\n    return 1\n','specification':'integer one','plan':'return one'}
        with tempfile.TemporaryDirectory() as directory:
            model=train_all(initial,[record.copy() for _ in range(4)],[[0,1,2,3]]*2,Path(directory))
            self.assertEqual(state_digest(initial),before);self.assertNotEqual(state_digest(model),before)
            self.assertTrue((Path(directory)/'trace.json').is_file())


if __name__=='__main__':unittest.main()
