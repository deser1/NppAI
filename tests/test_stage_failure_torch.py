import unittest
import torch
from model_backends import GenerationRequest,format_prompt
from scripts.stage_failure_diagnostic import conditional_readout


class ConditionalReadoutTests(unittest.TestCase):
    def test_response_only_alignment_and_eos_are_counted(self):
        payload=GenerationRequest(prompt='task');target='abc'
        seq=list(format_prompt(payload).encode())+list(target.encode())+[0]
        class Model:
            max_seq_len=512
            def __call__(self,tokens):
                logits=torch.full((1,len(seq)-1,256),-10.)
                for i,t in enumerate(seq[1:]): logits[0,i,t]=10.
                return logits
        r=conditional_readout(Model(),payload,target)
        self.assertEqual(r['target_tokens_including_eos'],[97,98,99,0])
        self.assertEqual(r['predicted_tokens_under_gold_prefixes'],[97,98,99,0])
        self.assertEqual(r['hits'],4);self.assertTrue(r['sequence_exact_under_gold_prefixes'])
        self.assertIsNone(r['first_target_mismatch'])
        class Short(Model): max_seq_len=2
        with self.assertRaises(ValueError): conditional_readout(Short(),payload,target)


if __name__=='__main__': unittest.main()
