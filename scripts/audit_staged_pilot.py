"""Offline reconstruction of retained stage-pilot reports; never execute source."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_staged_generation import score_output
from staged_generation import stage_values


def audit_report(report, records):
    if len(report['cases'])!=len(records) or report['total']!=len(records):
        raise ValueError('Case denominator mismatch')
    for case,record in zip(report['cases'],records):
        if (case['id'],case['prompt'],case['reference'])!=(record['id'],record['prompt'],record['answer']):
            raise ValueError('Evaluation task/reference changed')
        for arm,value in case['arms'].items():
            if arm=='staged':
                trace=value['trace']; prior={}; total=0
                for index,stage in enumerate(trace['stages']):
                    name=('specification','plan','code')[index]
                    if stage['stage']!=name or stage['request']!=stage_values(name,record['prompt'],'',prior,128):
                        raise ValueError('Stage request differs from original task/raw prior artifacts')
                    response=stage['response']
                    if response:
                        total+=response['generated_tokens']
                        if not stage['error']:
                            if response['model_id']!=report['model_id'] or response['finish_reason']!='eos' or not response['output'].strip() or response['generated_tokens']>128:
                                raise ValueError('Invalid accepted stage')
                            prior[name]=response['output']
                    if stage['error'] and index!=len(trace['stages'])-1:
                        raise ValueError('Pipeline continued after failed stage')
                expected=prior.get('code','') if trace['completed'] else ''
                if trace['code']!=expected or (trace['completed'] and len(prior)!=3) or trace['generated_tokens']!=total:
                    raise ValueError('Staged code/completion/token count mismatch')
                if trace['execution_status']!='not_run': raise ValueError('Generation trace claims execution')
                expected_error=next((s['error'] for s in trace['stages'] if s['error']),'')
                if value['error']!=expected_error: raise ValueError('Stage failure summary mismatch')
            else:
                budget=128 if arm=='direct_code_budget' else 384
                request=value['request']
                if request['prompt']!=record['prompt'] or request['context'] or request['assistant_prefix'] or request['max_new_tokens']!=budget:
                    raise ValueError('Direct control request changed')
                expected=value['response']['output'] if value['response'] and not value['error'] else ''
            if value['score']!=score_output(record,expected):
                raise ValueError('Raw output score mismatch')
    metrics=('nonempty','syntax_any_function','requested_function','reference_ast','exact_response')
    for arm,summary in report['summary'].items():
        expected={m:sum(c['arms'][arm]['score'][m] for c in report['cases']) for m in metrics}
        expected['errors']=sum(bool(c['arms'][arm]['error']) for c in report['cases'])
        if summary!=expected: raise ValueError('Summary differs from raw cases')
    return True


def audit_directory(directory):
    def read(name): return json.loads((directory/name).read_bytes())
    manifest=json.loads((ROOT/'evaluation/staged-protocol-v1/manifest.json').read_bytes())
    train=json.loads((ROOT/'evaluation/staged-protocol-v1/train.json').read_bytes())
    seen=set(); probe=[]
    for record in train:
        if record['family'] not in seen: probe.append(record);seen.add(record['family'])
    groups={'training_probe':probe,
        'new_synthetic':json.loads((ROOT/'evaluation/staged-protocol-v1/new-evaluation.json').read_bytes()),
        'exposed_v2':json.loads((ROOT/'evaluation/instruction-v2/test.json').read_bytes())}
    metadata=read('training-report.json')
    if metadata['manifest_sha256']!=hashlib.sha256((ROOT/'evaluation/staged-protocol-v1/manifest.json').read_bytes()).hexdigest():
        raise ValueError('Dataset manifest fingerprint differs')
    if metadata['protocol']!=manifest['protocol'] or metadata['promotion'] is not False:
        raise ValueError('Protocol/promotion mismatch')
    schedules=[]
    for arm in ('direct','staged_mix'):
        for group,records in groups.items(): audit_report(read(arm+'/'+group+'.json'),records)
        trace=read(arm+'/trace.json')
        if hashlib.sha256((directory/arm/'trace.json').read_bytes()).hexdigest()!=metadata['arms'][arm]['trace_sha256']:
            raise ValueError('Training trace fingerprint differs')
        if len(trace)!=1024 or any(v['step']!=i+1 or not math.isfinite(v['loss']) for i,v in enumerate(trace)):
            raise ValueError('Training step/loss mismatch')
        schedules.append([v['base_indices'] for v in trace])
        info=metadata['arms'][arm]
        if info['initial_state_sha256']!=metadata['initial_expanded_state_sha256'] or not info['evaluation_parameters_unchanged']:
            raise ValueError('Initial state or evaluation mutation differs')
        checkpoint=directory/arm/'experimental.nppai'
        if checkpoint.exists() and hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=info['checkpoint_sha256']:
            raise ValueError('Checkpoint hash differs')
        functional=directory/arm/'functional.json'
        if metadata['execution_status']=='docker' and not functional.exists():
            raise ValueError('Missing claimed functional execution evidence')
        if functional.exists():
            f=json.loads(functional.read_bytes())
            for name,result in f['arms'].items():
                generation=read(arm+'/new_synthetic.json')
                if result['total']!=12 or len(result['cases'])!=12 or result['passed']!=sum(c['status']=='passed' for c in result['cases']):
                    raise ValueError('Functional denominator/count differs')
                for c,r,raw in zip(result['cases'],groups['new_synthetic'],generation['cases']):
                    if c['id']!=r['id'] or c['source']!=raw['arms'][name]['score']['output']:
                        raise ValueError('Functional source differs from generated code')
    if schedules[0]!=schedules[1] or hashlib.sha256(json.dumps(schedules[0]).encode()).hexdigest()!=metadata['schedule_sha256']:
        raise ValueError('Paired task schedules differ')
    return {'valid':True,'models':2,'groups_per_model':3,'cases_per_model':56,'updates_per_model':1024}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit_directory(args.directory)))
