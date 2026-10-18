import tempfile
from pathlib import Path
import unittest
import torch
from train_nppai import NppAIModel
from scripts.staged_code_fit import train
from scripts.training_repro import state_digest


class CodeFitGradientTests(unittest.TestCase):
    def test_real_objectives_update_copies_without_mutating_initialization(self):
        torch.manual_seed(43);torch.set_num_threads(1)
        model=NppAIModel(256,4,8,1,1024)
        before=state_digest(model)
        row={'prompt':'write f','answer':'def f():\n    return 1\n','specification':'integer output','plan':'return one'}
        records=[row.copy() for _ in range(4)];states=[]
        with tempfile.TemporaryDirectory() as directory:
            for arm in ('code_only','code_with_direct_replay'):
                after,stats=train(model,records,[[0,1,2,3]]*2,arm,Path(directory)/arm)
                self.assertEqual(state_digest(model),before)
                self.assertNotEqual(state_digest(after),before)
                self.assertEqual(stats['steps'],2)
                self.assertGreater(stats['target_tokens'],0)
                states.append(state_digest(after))
        self.assertNotEqual(*states)


if __name__=='__main__':unittest.main()
