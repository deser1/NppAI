import copy
import unittest
from model_backends import GenerationResponse
from scripts.staged_code_fit import assignments,retention,MODES
from scripts.audit_staged_code_fit import audit_training
from scripts.stage_failure_diagnostic import intervention
from scripts.train_staged_protocol import supervision


class CodeFitTests(unittest.TestCase):
    def test_objective_assignment_rotates_replay_without_changing_batch_size(self):
        for step in range(4):
            self.assertEqual(assignments('code_only',step),['code']*4)
            self.assertEqual(assignments('code_with_direct_replay',step).count('code'),2)
        for position in range(4):
            self.assertEqual(sum(assignments('code_with_direct_replay',s)[position]=='direct' for s in range(4)),2)
        with self.assertRaises(ValueError): assignments('unknown',0)

    def test_retention_reports_losses_even_if_new_hits_raise_total(self):
        def rows(values):return {'cases':[{'id':k,'modes':{m:{'score':{'exact_response':v}} for m in MODES}} for k,v in values.items()]}
        a=rows({'old':True,'new1':False,'new2':False});b=rows({'old':False,'new1':True,'new2':True})
        r=retention(a,b)['direct']
        self.assertEqual(r['initial_exact'],1);self.assertEqual(r['retained'],0)
        self.assertEqual(r['lost_ids'],['old']);self.assertEqual(r['new_ids'],['new1','new2'])
        with self.assertRaises(ValueError): retention(a,rows({'different':True}))

    def test_audit_rejects_changed_scores_and_conditional_eos(self):
        record={'id':'train-fixture','prompt':'write f','function':'f','answer':'def f():\n    return 1\n',
            'specification':'integer output','plan':'return one'}
        class Backend:
            def __init__(self):self.outputs=iter([record['answer'],'spec','plan',record['answer'],record['answer']])
            def generate(self,payload):return GenerationResponse(model_id='fixture',output=next(self.outputs),generated_tokens=1,finish_reason='eos')
        backend=Backend();modes={m:intervention(backend,record,m) for m in MODES}
        payload,target=supervision(record,'code');tokens=list(target.encode())+[0]
        conditional={'request':payload.model_dump(),'target':target,'target_tokens_including_eos':tokens,
            'predicted_tokens_under_gold_prefixes':tokens,'target_log_probabilities':[-1.0]*len(tokens),
            'hits':len(tokens),'total':len(tokens),'sequence_exact_under_gold_prefixes':True,
            'mean_negative_log_probability':1.0,'first_target_mismatch':None}
        case={'id':record['id'],'record':record,'modes':modes,'conditional_code':conditional}
        value={'total':1,'cases':[case],'summary':{m:v['score'].copy() for m,v in modes.items()},'conditional_code_exact':1}
        for m in MODES:value['summary'][m].pop('output')
        self.assertTrue(audit_training(value,[record],'fixture'))
        bad=copy.deepcopy(value);bad['cases'][0]['conditional_code']['target_tokens_including_eos'][-1]=65
        with self.assertRaises(ValueError):audit_training(bad,[record],'fixture')
        bad=copy.deepcopy(value);bad['summary']['direct']['exact_response']=0
        with self.assertRaises(ValueError):audit_training(bad,[record],'fixture')


if __name__=='__main__':unittest.main()
