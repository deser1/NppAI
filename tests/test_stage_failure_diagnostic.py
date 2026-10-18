import copy
import json
from pathlib import Path
import tempfile
import unittest

from model_backends import GenerationResponse
from scripts.stage_failure_diagnostic import intervention,compare_training,source,code_mismatch_region
from scripts.audit_stage_failure import audit_mode


class Backend:
    def __init__(self,outputs): self.outputs=iter(outputs);self.calls=[]
    def generate(self,payload):
        self.calls.append(payload)
        return GenerationResponse(model_id='fixture',output=next(self.outputs),generated_tokens=1,finish_reason='eos')


class StageFailureTests(unittest.TestCase):
    def setUp(self):
        self.record={'id':'train-fixture','prompt':'write f','function':'f','answer':'def f():\n    return 1\n',
            'specification':'return integer one','plan':'return one'}

    def test_gold_priors_are_explicit_and_skip_generation_of_inserted_stages(self):
        b=Backend([self.record['answer']])
        result=intervention(b,self.record,'gold_specification_and_plan')
        self.assertEqual(len(b.calls),1)
        self.assertEqual(result['assistance'],{'specification':'return integer one','plan':'return one'})
        self.assertTrue(result['score']['exact_response'])
        audit_mode(result,self.record,'gold_specification_and_plan','fixture')
        self.assertEqual(b.calls[0].assistant_prefix,'')
        self.assertNotIn(self.record['answer'],b.calls[0].prompt+b.calls[0].context)

    def test_spec_only_keeps_raw_generated_plan_and_empty_plan_stops(self):
        b=Backend(['generated imperfect plan',self.record['answer']])
        r=intervention(b,self.record,'gold_specification')
        self.assertIn('generated imperfect plan',b.calls[1].context)
        self.assertNotIn('return one',b.calls[1].context)
        audit_mode(r,self.record,'gold_specification','fixture')
        b=Backend(['']);r=intervention(b,self.record,'gold_specification')
        self.assertEqual(len(b.calls),1);self.assertEqual(r['output'],'')
        audit_mode(r,self.record,'gold_specification','fixture')

    def test_audit_rejects_assistance_hidden_as_autonomous_or_fabricated_scores(self):
        r=intervention(Backend(['spec','plan',self.record['answer']]),self.record,'autonomous')
        audit_mode(r,self.record,'autonomous','fixture')
        bad=copy.deepcopy(r);bad['assistance']={'plan':'gold'}
        with self.assertRaises(ValueError): audit_mode(bad,self.record,'autonomous','fixture')
        bad=copy.deepcopy(r);bad['score']['exact_response']=False
        with self.assertRaises(ValueError): audit_mode(bad,self.record,'autonomous','fixture')

    def test_saved_loss_divergence_does_not_claim_an_operator(self):
        metadata={'initial_expanded_state_sha256':'same','schedule_sha256':'same',
            'arms':{a:{'checkpoint_sha256':'one'} for a in ('direct','staged_mix')}}
        trace=[{'step':i+1,'base_indices':[0,1,2,3],'stages':['direct']*4,'loss':1.0} for i in range(1024)]
        a={'training-report.json':json.dumps(metadata),**{arm+'/trace.json':json.dumps(trace) for arm in ('direct','staged_mix')}}
        b=copy.deepcopy(a);other=copy.deepcopy(trace);other[2]['loss']=0.999
        b['direct/trace.json']=json.dumps(other)
        r=compare_training(a,b)
        self.assertEqual(r['arms']['direct']['first_saved_loss_difference']['step'],3)
        self.assertIsNone(r['arms']['staged_mix']['first_saved_loss_difference'])
        self.assertIn('Does not identify',r['limit'])
        other[0]['base_indices']=[4,5,6,7];b['direct/trace.json']=json.dumps(other)
        with self.assertRaises(ValueError): compare_training(a,b)

    def test_code_mismatch_regions_distinguish_name_body_and_eos(self):
        record=self.record
        self.assertEqual(code_mismatch_region(record,None),'none')
        self.assertEqual(code_mismatch_region(record,0),'header_keyword')
        self.assertEqual(code_mismatch_region(record,4),'function_name')
        self.assertEqual(code_mismatch_region(record,5),'signature_after_name')
        self.assertEqual(code_mismatch_region(record,12),'body')
        self.assertEqual(code_mismatch_region(record,len(record['answer'].encode())),'eos')

    def test_wrong_archive_fails_before_loading_any_model(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'bad.zip';p.write_bytes(b'not trusted')
            with self.assertRaises(ValueError): source(p,12149182157)


if __name__=='__main__': unittest.main()
