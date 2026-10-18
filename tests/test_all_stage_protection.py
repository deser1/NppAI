import copy
import unittest
from model_backends import GenerationResponse
from experimental_stage_router import run_routed
from scripts.all_stage_protection import objective_assignment,retained,frozen_artifacts_equal
from scripts.audit_all_stage_protection import audit_trace


class Backend:
    def __init__(self,name,outputs):self.model_id=name;self.outputs=iter(outputs);self.requests=[]
    def generate(self,payload):
        self.requests.append(payload)
        x=next(self.outputs)
        return x if isinstance(x,GenerationResponse) else GenerationResponse(model_id=self.model_id,output=x,generated_tokens=1,finish_reason='eos')


class StageProtectionTests(unittest.TestCase):
    def test_explicit_roles_route_only_code_to_the_second_own_checkpoint(self):
        planner=Backend('planner',['spec 🧪','plan']);coder=Backend('coder',['def f():\n    return 1\n'])
        result=run_routed(planner,coder,'Wygeneruj f')
        self.assertTrue(result['completed']);self.assertEqual(len(planner.requests),2);self.assertEqual(len(coder.requests),1)
        self.assertEqual([c['expected_model_id'] for c in result['stages']],['planner','planner','coder'])
        self.assertIn('spec 🧪',coder.requests[0].context);self.assertEqual(coder.requests[0].assistant_prefix,'')
        self.assertTrue(all('Wygeneruj f' in r.prompt for r in planner.requests+coder.requests))

    def test_empty_or_truncated_planning_never_calls_coder(self):
        for bad in ('',GenerationResponse(model_id='planner',output='partial',generated_tokens=128,finish_reason='length')):
            planner=Backend('planner',[bad]);coder=Backend('coder',[])
            r=run_routed(planner,coder,'task')
            self.assertFalse(r['completed']);self.assertEqual(coder.requests,[]);self.assertTrue(r['stages'][0]['error'])

    def test_wrong_model_and_mutation_are_rejected_without_weakening_identity_checks(self):
        wrong=GenerationResponse(model_id='unexpected',output='spec',generated_tokens=1,finish_reason='eos')
        r=run_routed(Backend('planner',[wrong]),Backend('coder',[]),'task')
        self.assertFalse(r['completed']);self.assertIn('checkpoint',r['stages'][0]['error'])
        class Mutating(Backend):
            def generate(self,payload):
                self.model_id='changed'
                return GenerationResponse(model_id='changed',output='spec',generated_tokens=1,finish_reason='eos')
        r=run_routed(Mutating('planner',[]),Backend('coder',[]),'task')
        self.assertFalse(r['completed'])

    def test_rotation_has_one_supervised_example_for_each_stage(self):
        for step in range(4):self.assertEqual(set(objective_assignment(step)),{'direct','specification','plan','code'})
        for position in range(4):self.assertEqual({objective_assignment(s)[position] for s in range(4)},{'direct','specification','plan','code'})

    def test_retention_tracks_lost_originals_instead_of_just_totals(self):
        def read(values):return {'cases':[{'id':k,**{m:{'score':{'exact_response':v}} for m in ('direct','autonomous','gold_code')}} for k,v in values.items()]}
        a=read({'old':True,'new':False});b=read({'old':False,'new':True})
        x=retained(a,b)['autonomous'];self.assertEqual(x['retained'],0);self.assertEqual(x['lost_ids'],['old'])
        with self.assertRaises(ValueError):retained(a,read({'other':True}))

    def test_raw_audit_rejects_role_and_output_tampering(self):
        r=run_routed(Backend('planner',['spec','plan']),Backend('coder',['code']),'task')
        self.assertEqual(audit_trace(r,'task','planner','coder'),'code')
        wrong=copy.deepcopy(r);wrong['stages'][0]['role']='coder'
        with self.assertRaises(ValueError):audit_trace(wrong,'task','planner','coder')
        wrong=copy.deepcopy(r);wrong['code']='invented successful output'
        with self.assertRaises(ValueError):audit_trace(wrong,'task','planner','coder')

    def test_frozen_planner_comparison_rejects_changed_raw_artifacts(self):
        p=Backend('planner',['spec','plan']);c=Backend('coder',['code'])
        trace=run_routed(p,c,'task');a={'cases':[{'id':'one','autonomous':{'trace':trace}}]}
        b=copy.deepcopy(a);b['cases'][0]['autonomous']['trace']['code']='different code'
        self.assertTrue(frozen_artifacts_equal(a,b))
        b['cases'][0]['autonomous']['trace']['stages'][0]['response']['output']='different spec'
        self.assertFalse(frozen_artifacts_equal(a,b))


if __name__=='__main__':unittest.main()
