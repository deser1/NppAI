"""Offline audit of stage roles, raw outputs, retention and training budget."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from staged_generation import stage_values
from model_backends import GenerationRequest
from scripts.all_stage_protection import ARMS,CODER,STAGES,objective_assignment,retained,summarize,frozen_artifacts_equal,artifact_retention
from scripts.stage_failure_diagnostic import digest,encoded
from scripts.train_staged_protocol import supervision
from scripts.evaluate_staged_generation import score_output
from scripts.audit_stage_failure import accepted


def audit_trace(trace,task,planner_id,coder_id):
    prior={};total=0
    if not 1<=len(trace['stages'])<=3:raise ValueError('Stage denominator differs')
    for i,call in enumerate(trace['stages']):
        stage=('specification','plan','code')[i];role='coder' if stage=='code' else 'planner'
        model_id=coder_id if role=='coder' else planner_id
        if (call['stage']!=stage or call['role']!=role or call['expected_model_id']!=model_id or
            call['request']!=GenerationRequest(**stage_values(stage,task,'',prior,128)).model_dump()):raise ValueError('Stage role/request identity differs')
        response=call['response']
        if response:
            total+=response['generated_tokens']
            reason=('Unexpected checkpoint for stage role' if response['model_id']!=model_id else
                'Stage token budget exceeded' if response['generated_tokens']>128 else 'Empty stage' if not response['output'].strip() else
                'Stage token limit' if response['finish_reason']!='eos' else '')
            if call['error']!=reason:raise ValueError('Stage acceptance differs')
            if not reason:prior[stage]=response['output']
        elif not call['error']:raise ValueError('Missing response without error')
        if call['error'] and i!=len(trace['stages'])-1:raise ValueError('Continued failed stage')
    expected=prior.get('code','') if trace['completed'] else ''
    if trace['code']!=expected or (trace['completed'] and len(prior)!=3) or trace['generated_tokens']!=total or trace['execution_status']!='not_run':raise ValueError('Trace completion/output differs')
    return expected


def audit_group(value,records,planner_id,coder_id,gold):
    if value['total']!=len(records) or len(value['cases'])!=len(records):raise ValueError('Group denominator differs')
    for case,r in zip(value['cases'],records):
        if case['id']!=r['id'] or case['record']!=r:raise ValueError('Case source differs')
        direct=case['direct']
        if direct['request']!=GenerationRequest(prompt=r['prompt'],max_new_tokens=128).model_dump() or direct['response']['model_id']!=coder_id:raise ValueError('Direct route differs')
        if direct['score']!=score_output(r,direct['response']['output']):raise ValueError('Direct score differs')
        text=audit_trace(case['autonomous']['trace'],r['prompt'],planner_id,coder_id)
        if case['autonomous']['score']!=score_output(r,text):raise ValueError('Autonomous score differs')
        if gold:
            call=case['gold_code']['call'];p,_=supervision(r,'code')
            if call['request']!=p.model_dump() or call['response']['model_id']!=coder_id:raise ValueError('Gold-prior provenance differs')
            if case['gold_code']['score']!=score_output(r,accepted(call)):raise ValueError('Gold-prior score differs')
        elif 'gold_code' in case:raise ValueError('Assistance inserted in regression group')
    if value['summary']!=summarize(value['cases'],gold):raise ValueError('Summary differs from raw records')
    return True


def audit(directory):
    def read(name):return json.loads((directory/name).read_bytes())
    p=json.loads((ROOT/'evaluation/all-stage-protection-v1.json').read_bytes());r=read('report.json')
    training=json.loads((ROOT/'evaluation/staged-protocol-v1/train.json').read_bytes());before=read('initial-training.json')
    if r['protocol']!=p or r['protocol_sha256']!=digest(encoded(p)) or set(r['models'])!=set(ARMS) or r['promotion'] is not False:raise ValueError('Protocol/model set differs')
    if r['train_sha256']!=digest((ROOT/'evaluation/staged-protocol-v1/train.json').read_bytes()) or not r['frozen_sources_unchanged'] or not r['evaluation_unchanged']:raise ValueError('Source/mutation guard differs')
    audit_group(before,training,'sha256:'+p['initial_checkpoint'],'sha256:'+p['initial_checkpoint'],True)
    trace=read('all_stage_replay/trace.json')
    if len(trace)!=512:raise ValueError('Training budget differs')
    for i,t in enumerate(trace):
        if t['step']!=i+1 or t['stages']!=objective_assignment(i) or len(set(t['indices']))!=4 or any(not 0<=k<96 for k in t['indices']) or not math.isfinite(t['loss']):raise ValueError('Objective/task/loss differs')
    schedule=[t['indices'] for t in trace]
    if digest(encoded(schedule))!=r['schedule_sha256']:raise ValueError('Schedule fingerprint differs')
    if digest((directory/'historical-coder.json').read_bytes())!=r['historical_coder_sha256']:raise ValueError('Historical source differs')
    old=read('historical-coder.json')
    if schedule!=[t['indices'] for t in old['trace']] or old['report']['initial_state_sha256']!=r['initial_state_sha256']:raise ValueError('Historical initialization/schedule mismatch')
    identities={'initial':('sha256:'+p['initial_checkpoint'],)*2,'all_stage_replay':('sha256:'+r['new_checkpoint_sha256'],)*2,
        'frozen_planner_coder':('sha256:'+p['initial_checkpoint'],'sha256:'+CODER)}
    baseline=read('initial/training.json')
    if baseline!=before:raise ValueError('Initial readouts differ')
    for name,(planner_id,coder_id) in identities.items():
        info=r['models'][name]
        if (info['planner_id'],info['coder_id'])!=(planner_id,coder_id) or info['loaded_parameters']!=(361088 if name=='frozen_planner_coder' else 180544):raise ValueError('Model role/parameter accounting differs')
        after=read(name+'/training.json');audit_group(after,training,planner_id,coder_id,True)
        if info['retention']!=retained(before,after) or info['artifact_retention']!=artifact_retention(before,after) or info['training_summary']!=after['summary']:raise ValueError('Retention differs')
        if name=='frozen_planner_coder' and (not info['all_training_planner_calls_equal_initial'] or not frozen_artifacts_equal(before,after)):raise ValueError('Frozen planner artifacts differ')
        for group,path in (('validation','evaluation/instruction-v2/validation.json'),('bilingual','evaluation/staged-protocol-v1/new-evaluation.json')):
            raw=(ROOT/path).read_bytes()
            if r['evaluation_sha256'][group]!=digest(raw):raise ValueError('Regression fingerprint differs')
            audit_group(read(name+'/'+group+'.json'),json.loads(raw),planner_id,coder_id,False)
        if r['execution_status']=='docker':
            f=read(name+'/functional.json');cases=read(name+'/bilingual.json')['cases']
            if set(f['modes'])!={'direct','autonomous'} or not re.fullmatch(r'sha256:[0-9a-f]{64}',f['image']):raise ValueError('Functional mode/image differs')
            for mode,value in f['modes'].items():
                if value['total']!=12 or len(value['cases'])!=12 or value['passed']!=sum(c['passed'] for c in value['cases']):raise ValueError('Functional denominator differs')
                for c,original in zip(value['cases'],cases):
                    if c['id']!=original['id'] or c['source']!=original[mode]['score']['output'] or c['passed']!=(c['exit_code']==0) or c['exit_code'] not in (None,0,1):raise ValueError('Functional source/status differs')
    if r['execution_status']=='docker':
        controls=r['validator_controls'];expected=json.loads((ROOT/'evaluation/staged-protocol-v1/new-evaluation.json').read_bytes())
        if [c['id'] for c in controls['references']]!=[c['id'] for c in expected] or any(c['exit_code']!=0 for c in controls['references']) or controls['incorrect_exit_code']!=1:raise ValueError('Validator control differs')
    parity=read('parity.json')
    if len(parity)!=4 or [v['stage'] for v in parity]!=list(STAGES) or r['four_exported_readouts_equal']!=all(v['equal'] for v in parity):raise ValueError('Parity denominator differs')
    for v in parity:
        if v['equal']!=(v['torch']==v['dense']) or v['torch']['model_id']!='sha256:'+r['new_checkpoint_sha256']:raise ValueError('Parity identity/readout differs')
    checkpoint=directory/'all_stage_replay/experimental.nppai'
    if checkpoint.exists() and digest(checkpoint.read_bytes())!=r['new_checkpoint_sha256']:raise ValueError('Checkpoint hash differs')
    return {'valid':True,'arms':3,'training_cases':96,'new_updates':512,'promotion':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args();print(json.dumps(audit(a.directory)))
