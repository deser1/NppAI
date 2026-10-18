import copy
from pathlib import Path
import tempfile
import unittest

import torch
from model_backends import GenerationRequest
from scripts.train_staged_protocol import encoded_sample,expanded_initial,supervision,train_arm,TorchBackend
from scripts.training_repro import state_digest
from scripts.check_dense_runtime_parity import load_weights
from test_model_backends import checkpoint


class StageOptimizationTests(unittest.TestCase):
    def test_context_expansion_preserves_old_positions_and_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.nppai'; checkpoint(path)
            model=expanded_initial(path,1024)
            old=load_weights(path)
            self.assertTrue(torch.equal(model.pos_embeddings.weight[:128],torch.from_numpy(old['position'])))
            self.assertEqual(model.max_seq_len,1024)
            self.assertTrue(all(m.eps==1e-5 for m in model.modules() if isinstance(m,torch.nn.RMSNorm)))
            p=GenerationRequest(prompt='task')
            _,y=encoded_sample(p,'abc',1024)
            self.assertEqual(int((y!=-100).sum()),4) # response and EOS only
            with self.assertRaises(ValueError): encoded_sample(p,'abc',2)

    def test_real_updates_leave_initial_state_untouched_and_share_task_schedule(self):
        torch.set_num_threads(1)
        row={'prompt':'task','answer':'def f():\n    return 1\n','specification':'integer output','plan':'return one'}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source.nppai'; checkpoint(source)
            model=expanded_initial(source,1024)
            before=state_digest(model)
            rows=[copy.deepcopy(row) for _ in range(4)]
            schedule=[[0,1,2,3]]*2
            states=[]
            for arm in ('direct','staged_mix'):
                trained,summary=train_arm(model,rows,schedule,arm,1024,root/arm)
                self.assertEqual(state_digest(model),before)
                self.assertNotEqual(state_digest(trained),before)
                self.assertEqual(summary['steps'],2)
                self.assertTrue((root/arm/'trace.json').is_file())
                states.append(state_digest(trained))
            self.assertNotEqual(*states)


if __name__=='__main__': unittest.main()
