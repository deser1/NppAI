"""Offline raw-response, retention, fixed-budget and execution-evidence audit."""
import argparse
import json
import math
import re
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.staged_code_fit import ARMS,MODES,assignments,retention
from scripts.stage_failure_diagnostic import encoded,digest
from scripts.train_staged_protocol import supervision
from scripts.audit_stage_failure import audit_mode
from scripts.audit_staged_pilot import audit_report


def audit_training(value,records,model_id):
    if value['total']!=len(records) or len(value['cases'])!=len(records): raise ValueError('Training denominator mismatch')
    for case,record in zip(value['cases'],records):
        if case['id']!=record['id'] or case['record']!=record or set(case['modes'])!=set(MODES): raise ValueError('Training case mismatch')
        for mode in MODES: audit_mode(case['modes'][mode],record,mode,model_id)
        conditional=case['conditional_code'];payload,target=supervision(record,'code')
        targets=list(target.encode())+[0];pred=conditional['predicted_tokens_under_gold_prefixes'];logp=conditional['target_log_probabilities']
        if conditional['request']!=payload.model_dump() or conditional['target']!=target or conditional['target_tokens_including_eos']!=targets or len(pred)!=len(targets) or len(logp)!=len(targets):
            raise ValueError('Conditional targets/prefixes differ')
        if any(type(t) is not int or not 0<=t<256 for t in pred) or any(not math.isfinite(v) or v>0 for v in logp): raise ValueError('Invalid conditional readout')
        if (conditional['hits']!=sum(x==y for x,y in zip(targets,pred)) or conditional['total']!=len(targets) or
            conditional['sequence_exact_under_gold_prefixes']!=(targets==pred) or
            conditional['first_target_mismatch']!=next((i for i,(x,y) in enumerate(zip(targets,pred)) if x!=y),None) or
            not math.isclose(conditional['mean_negative_log_probability'],-sum(logp)/len(logp),abs_tol=1e-12)):
            raise ValueError('Conditional scores differ')
    summary={m:{k:sum(c['modes'][m]['score'][k] for c in value['cases']) for k in
           ('nonempty','syntax_any_function','requested_function','reference_ast','exact_response')} for m in MODES}
    if value['summary']!=summary or value['conditional_code_exact']!=sum(c['conditional_code']['sequence_exact_under_gold_prefixes'] for c in value['cases']):
        raise ValueError('Training summary differs from raw data')
    return True


def audit(directory):
    def read(name): return json.loads((directory/name).read_bytes())
    protocol=json.loads((ROOT/'evaluation/staged-code-fit-v1.json').read_bytes())
    records=json.loads((ROOT/'evaluation/staged-protocol-v1/train.json').read_bytes())
    report=read('report.json');before=read('before.json')
    if report['protocol']!=protocol or report['protocol_sha256']!=digest(encoded(protocol)) or set(report['models'])!=set(ARMS) or report['promotion'] is not False:
        raise ValueError('Protocol/model/promotion mismatch')
    if report['train_sha256']!=digest((ROOT/'evaluation/staged-protocol-v1/train.json').read_bytes()) or report['before_sha256']!=digest((directory/'before.json').read_bytes()):
        raise ValueError('Training or before fingerprint mismatch')
    if report['execution_status'] not in ('not_run','docker'): raise ValueError('Unknown execution status')
    if (directory/'initial.nppai').exists() and digest((directory/'initial.nppai').read_bytes())!=protocol['source_checkpoint']: raise ValueError('Initial checkpoint differs')
    audit_training(before,records,'sha256:'+protocol['source_checkpoint'])
    schedules=[]
    for arm in ARMS:
        info=report['models'][arm];after=read(arm+'/training.json')
        audit_training(after,records,'sha256:'+info['checkpoint_sha256'])
        if info['retention']!=retention(before,after) or info['initial_state_sha256']!=report['initial_state_sha256'] or not info['evaluation_unchanged']:
            raise ValueError('Retention/initialization/evaluation mismatch')
        trace=read(arm+'/trace.json')
        if len(trace)!=512 or digest((directory/arm/'trace.json').read_bytes())!=info['trace_sha256']: raise ValueError('Trace budget/fingerprint mismatch')
        for i,t in enumerate(trace):
            if t['step']!=i+1 or t['stages']!=assignments(arm,i) or len(set(t['indices']))!=4 or any(type(k) is not int or not 0<=k<96 for k in t['indices']) or not math.isfinite(t['loss']):
                raise ValueError('Trace task/objective/loss mismatch')
        if info['steps']!=512 or info['input_tokens']!=sum(t['input_tokens'] for t in trace) or info['target_tokens']!=sum(t['target_tokens'] for t in trace) or info['initial_loss']!=trace[0]['loss'] or info['final_loss']!=trace[-1]['loss']:
            raise ValueError('Training accounting mismatch')
        schedules.append([t['indices'] for t in trace])
        for group,path in (('exposed_validation','evaluation/instruction-v2/validation.json'),('bilingual_regression','evaluation/staged-protocol-v1/new-evaluation.json')):
            data=(ROOT/path).read_bytes()
            if report['evaluation_sha256'][path]!=digest(data): raise ValueError('Regression input mismatch')
            audit_report(read(arm+'/'+group+'.json'),json.loads(data))
        parity=read(arm+'/parity.json')
        if len(parity)!=4 or [p['stage'] for p in parity]!=['direct','specification','plan','code']: raise ValueError('Parity denominator differs')
        for p in parity:
            request,_=supervision(records[0],p['stage'])
            if p['request']!=request.model_dump() or p['equal']!=(p['torch']==p['dense']) or p['torch']['model_id']!='sha256:'+info['checkpoint_sha256'] or p['dense']['model_id']!='sha256:'+info['checkpoint_sha256']: raise ValueError('Parity record differs')
        if info['four_exported_readouts_equal']!=all(p['equal'] for p in parity): raise ValueError('Parity summary differs')
        checkpoint=directory/arm/'experimental.nppai'
        if checkpoint.exists() and digest(checkpoint.read_bytes())!=info['checkpoint_sha256']: raise ValueError('Checkpoint fingerprint mismatch')
        if report['execution_status']=='docker':
            f=read(arm+'/functional.json');raw=read(arm+'/bilingual_regression.json')
            if set(f['arms'])!={'staged','direct_code_budget','direct_total_budget'} or not re.fullmatch(r'sha256:[0-9a-f]{64}',f['image']): raise ValueError('Functional arm/image identity differs')
            for name,value in f['arms'].items():
                if value['total']!=12 or len(value['cases'])!=12 or value['passed']!=sum(c['status']=='passed' for c in value['cases']): raise ValueError('Functional denominator mismatch')
                for c,original in zip(value['cases'],raw['cases']):
                    if c['id']!=original['id'] or c['source']!=original['arms'][name]['score']['output']: raise ValueError('Functional source mismatch')
                    status='empty' if not c['source'].strip() else 'passed' if c['exit_code']==0 else 'candidate_failed' if c['exit_code']==1 else 'infrastructure_failure'
                    if c['status']!=status or status=='infrastructure_failure': raise ValueError('Functional status mismatch')
    if schedules[0]!=schedules[1] or digest(encoded(schedules[0]))!=report['schedule_sha256']: raise ValueError('Paired schedules differ')
    if report['execution_status']=='docker':
        controls=report['validator_controls']
        expected_ids=[r['id'] for r in json.loads((ROOT/'evaluation/staged-protocol-v1/new-evaluation.json').read_bytes())]
        if [c['id'] for c in controls['references']]!=expected_ids or len(controls['references'])!=12 or any(c['exit_code']!=0 for c in controls['references']) or controls['incorrect_exit_code']!=1:
            raise ValueError('Validator controls failed')
    return {'valid':True,'models':2,'steps_per_model':512,'training_records':96,'promotion':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    print(json.dumps(audit(a.directory)))
