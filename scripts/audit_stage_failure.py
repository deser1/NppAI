"""Reconstruct frozen stage diagnostics from raw records without model execution."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_backends import GenerationRequest
from staged_generation import stage_values
from scripts.train_staged_protocol import supervision
from scripts.stage_failure_diagnostic import digest,MODELS,MODES,SOURCES,summarize,compare_training
from scripts.evaluate_staged_generation import score_output


def accepted(call):
    response=call['response']
    reason='token budget exceeded' if response['generated_tokens']>128 else 'empty stage' if not response['output'].strip() else 'stage token limit' if response['finish_reason']!='eos' else ''
    if call['error']!=reason: raise ValueError('Accepted/failure decision differs')
    return '' if reason else response['output']


def audit_mode(value,record,mode,model_id):
    if mode=='direct':
        calls=value['calls']
        if len(calls)!=1 or calls[0]['request']!=GenerationRequest(prompt=record['prompt'],max_new_tokens=128).model_dump() or value['assistance']:
            raise ValueError('Direct control request differs')
        expected=calls[0]['response']['output']
    elif mode=='autonomous':
        trace=value['trace']; prior={};total=0
        if value['assistance']: raise ValueError('Autonomous arm received assistance')
        for i,call in enumerate(trace['stages']):
            name=('specification','plan','code')[i]
            if call['stage']!=name or call['request']!=stage_values(name,record['prompt'],'',prior,128):
                raise ValueError('Autonomous stage request differs')
            response=call['response']
            if response:
                total+=response['generated_tokens']
                if response['model_id']!=model_id: raise ValueError('Checkpoint identity differs')
                if not call['error']:
                    if response['generated_tokens']>128 or not response['output'].strip() or response['finish_reason']!='eos':
                        raise ValueError('Invalid accepted autonomous stage')
                    prior[name]=response['output']
            elif not call['error']: raise ValueError('Missing stage response without failure')
            if call['error'] and i!=len(trace['stages'])-1: raise ValueError('Continued failed autonomous stage')
        expected=prior.get('code','') if trace['completed'] else ''
        if trace['code']!=expected or (trace['completed'] and len(prior)!=3) or trace['generated_tokens']!=total or trace['execution_status']!='not_run':
            raise ValueError('Autonomous completion/code differs')
        calls=trace['stages']
    else:
        prior={'specification':record['specification']}; assistance=dict(prior); calls=value['calls']
        index=0
        if mode=='gold_specification_and_plan':
            prior['plan']=record['plan'];assistance['plan']=record['plan']
        else:
            if not calls or calls[0]['request']!=GenerationRequest(**stage_values('plan',record['prompt'],'',prior,128)).model_dump():
                raise ValueError('Assisted plan request differs')
            plan=accepted(calls[0]);index=1
            if plan: prior['plan']=plan
        if value['assistance']!=assistance: raise ValueError('Assistance provenance differs')
        if 'plan' in prior:
            if len(calls)!=index+1 or calls[index]['request']!=GenerationRequest(**stage_values('code',record['prompt'],'',prior,128)).model_dump():
                raise ValueError('Assisted code request differs')
            expected=accepted(calls[index])
        else:
            if len(calls)!=1: raise ValueError('Generated code after a failed plan')
            expected=''
    if any(c['response'] and c['response']['model_id']!=model_id for c in calls): raise ValueError('Response model identity differs')
    if value['output']!=expected or value['score']!=score_output(record,expected): raise ValueError('Raw generation score differs')


def audit(report,comparison=None):
    records=json.loads((ROOT/'evaluation/staged-protocol-v1/train.json').read_bytes())
    if report['train_sha256']!=digest((ROOT/'evaluation/staged-protocol-v1/train.json').read_bytes()) or set(report['models'])!=set(MODELS):
        raise ValueError('Training source/model set differs')
    if report['optimization_performed'] is not False or report['generated_source_executed'] is not False or report['promotion'] is not False:
        raise ValueError('Frozen experiment scope differs')
    if report['checkpoints']!=MODELS or {str(k):v for k,v in report['source_artifacts'].items()}!={str(k):v for k,v in SOURCES.items()}:
        raise ValueError('Pinned model/archive identity differs')
    for arm,value in report['models'].items():
        if len(value['cases'])!=96 or not value['parameters_unchanged']: raise ValueError('Case count or frozen state differs')
        for case,record in zip(value['cases'],records):
            if case['id']!=record['id'] or case['record']!=record or set(case['modes'])!=set(MODES): raise ValueError('Case source differs')
            for mode in MODES: audit_mode(case['modes'][mode],record,mode,'sha256:'+MODELS[arm])
            if set(case['conditional'])!={'direct','specification','plan','code'}: raise ValueError('Conditional stage set differs')
            for stage,conditional in case['conditional'].items():
                payload,target=supervision(record,stage)
                expected=list(target.encode('utf-8'))+[0]
                prediction=conditional['predicted_tokens_under_gold_prefixes']; logp=conditional['target_log_probabilities']
                if conditional['request']!=payload.model_dump() or conditional['target']!=target or conditional['target_tokens_including_eos']!=expected or len(prediction)!=len(expected) or len(logp)!=len(expected):
                    raise ValueError('Conditional target/prefix/denominator differs')
                if any(type(t) is not int or not 0<=t<256 for t in prediction) or any(not math.isfinite(v) or v>0 for v in logp):
                    raise ValueError('Invalid conditional readouts')
                if (conditional['hits']!=sum(x==y for x,y in zip(expected,prediction)) or conditional['total']!=len(expected) or
                    conditional['sequence_exact_under_gold_prefixes']!=(expected==prediction) or
                    conditional['first_target_mismatch']!=next((i for i,(x,y) in enumerate(zip(expected,prediction)) if x!=y),None) or
                    not math.isclose(conditional['mean_negative_log_probability'],-sum(logp)/len(logp),abs_tol=1e-12)):
                    raise ValueError('Conditional metric differs from raw readouts')
        if value['summary']!=summarize(value['cases']): raise ValueError('Summary differs from raw cases')
    if comparison:
        if report['source_comparison_sha256']!=digest(comparison.read_bytes()): raise ValueError('Source comparison fingerprint differs')
        raw=json.loads(comparison.read_bytes())
        first={'training-report.json':json.dumps(raw['first'])};second={'training-report.json':json.dumps(raw['second'])}
        for arm in MODELS:
            first[arm+'/trace.json']=json.dumps(raw['first_traces'][arm]);second[arm+'/trace.json']=json.dumps(raw['second_traces'][arm])
        if report['training_comparison']!=compare_training(first,second): raise ValueError('Training divergence differs from source traces')
    return {'valid':True,'models':2,'training_cases_per_model':96,'assistance_separate':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    print(json.dumps(audit(json.loads((args.directory/'report.json').read_bytes()),args.directory/'source-comparison.json')))
