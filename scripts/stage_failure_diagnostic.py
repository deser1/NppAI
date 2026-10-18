"""Frozen training-only stage interventions; no optimization or source execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_backends import GenerationRequest
from staged_generation import stage_values,run_staged
from scripts.train_staged_protocol import supervision,format_prompt
from scripts.evaluate_staged_generation import score_output

SOURCES={
 12149182157:'3c0fdbacc37bb2826bca7051a041d9e0a092213dea8f729a97f03d5347d3d7b2',
 12149930502:'5ebea549959b7d8d7558f5e9bd2d80d3e2f9cd12bbc96e00f90b591a02e80dd2',
}
MODELS={'direct':'14053463569d60e828a7d9bddfa3bbeff9025f59c06c5e3aa927ef90e372d476',
        'staged_mix':'4859ad69e93ff2ca80d8951bca617476d118f8ef127abc4b313b8d2f0671c6fd'}
MODES=('direct','autonomous','gold_specification','gold_specification_and_plan')


def encoded(value): return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
def digest(raw): return hashlib.sha256(raw).hexdigest()


def source(archive,artifact_id):
    if digest(archive.read_bytes())!=SOURCES[artifact_id]: raise ValueError('Pinned archive mismatch')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise ValueError('Archive CRC failure')
        members={n:z.read(n) for n in z.namelist() if n.endswith(('.json','.nppai'))}
    if artifact_id==12149182157:
        for arm,expected in MODELS.items():
            if digest(members[arm+'/experimental.nppai'])!=expected: raise ValueError('Pinned checkpoint mismatch')
    return members


def compare_training(first,second):
    a,b=(json.loads(s['training-report.json']) for s in (first,second))
    result={'same_expanded_initial_state':a['initial_expanded_state_sha256']==b['initial_expanded_state_sha256'],
      'same_task_schedule_fingerprint':a['schedule_sha256']==b['schedule_sha256'],'arms':{},
      'limit':'Saved scalar losses and final checkpoints only; no per-update tensors/weights. Does not identify a causal operator or certify portability.'}
    for arm in MODELS:
        x,y=(json.loads(s[arm+'/trace.json']) for s in (first,second))
        if len(x)!=1024 or len(y)!=1024: raise ValueError('Training trace denominator mismatch')
        if any(p['base_indices']!=q['base_indices'] or p['stages']!=q['stages'] for p,q in zip(x,y)):
            raise ValueError('Base tasks/stage assignments differ between sources')
        mismatch=next(({'step':p['step'],'first_loss':p['loss'],'second_loss':q['loss']}
                       for p,q in zip(x,y) if p['loss']!=q['loss']),None)
        result['arms'][arm]={'first_saved_loss_difference':mismatch,
            'checkpoint_equal':a['arms'][arm]['checkpoint_sha256']==b['arms'][arm]['checkpoint_sha256']}
    return result


def accepted_generate(backend,payload):
    response=backend.generate(payload)
    error=''
    if response.generated_tokens>payload.max_new_tokens: error='token budget exceeded'
    elif not response.output.strip(): error='empty stage'
    elif response.finish_reason!='eos': error='stage token limit'
    return {'request':payload.model_dump(),'response':response.model_dump(),'error':error}, '' if error else response.output


def intervention(backend,record,mode):
    if mode not in MODES: raise ValueError('Unknown intervention')
    if mode=='autonomous':
        trace=run_staged(backend,record['prompt']).to_dict()
        return {'assistance':{},'trace':trace,'output':trace['code'],'score':score_output(record,trace['code'])}
    if mode=='direct':
        payload=GenerationRequest(prompt=record['prompt'],max_new_tokens=128)
        response=backend.generate(payload)
        return {'assistance':{},'calls':[{'request':payload.model_dump(),'response':response.model_dump(),'error':''}],
                'output':response.output,'score':score_output(record,response.output)}
    prior={'specification':record['specification']}
    assistance=dict(prior); calls=[]
    if mode=='gold_specification_and_plan':
        prior['plan']=record['plan']; assistance['plan']=record['plan']
    else:
        call,plan=accepted_generate(backend,GenerationRequest(**stage_values('plan',record['prompt'],'',prior,128)))
        calls.append(call)
        if call['error']:
            return {'assistance':assistance,'calls':calls,'output':'','score':score_output(record,'')}
        prior['plan']=plan
    call,output=accepted_generate(backend,GenerationRequest(**stage_values('code',record['prompt'],'',prior,128)))
    calls.append(call)
    return {'assistance':assistance,'calls':calls,'output':output,'score':score_output(record,output)}


def conditional_readout(model,payload,target):
    """Gold-prefix next-byte diagnostics, explicitly NOT autonomous generation."""
    import torch
    prefix=list(format_prompt(payload).encode('utf-8'))
    targets=list(target.encode('utf-8'))+[0]
    sequence=prefix+targets
    if len(sequence)>model.max_seq_len: raise ValueError('Full conditional sequence exceeds context')
    with torch.no_grad():
        logits=model(torch.tensor([sequence[:-1]]))[0,len(prefix)-1:]
        logp=torch.log_softmax(logits,dim=-1)
        prediction=logits.argmax(-1).tolist()
        probability=logp[torch.arange(len(targets)),torch.tensor(targets)].tolist()
    return {'request':payload.model_dump(),'target':target,'target_tokens_including_eos':targets,
        'predicted_tokens_under_gold_prefixes':prediction,'target_log_probabilities':probability,
        'hits':sum(x==y for x,y in zip(targets,prediction)),'total':len(targets),
        'sequence_exact_under_gold_prefixes':targets==prediction,
        'mean_negative_log_probability':-sum(probability)/len(probability),
        'first_target_mismatch':next((i for i,(x,y) in enumerate(zip(targets,prediction)) if x!=y),None)}


def code_mismatch_region(record,index):
    target=record['answer'].encode('utf-8')
    if index is None: return 'none'
    if index==len(target): return 'eos'
    if index<4: return 'header_keyword'
    if index<4+len(record['function'].encode('utf-8')): return 'function_name'
    if index<=target.index(b'\n'): return 'signature_after_name'
    return 'body'


def summarize(cases):
    metrics=('nonempty','syntax_any_function','requested_function','reference_ast','exact_response')
    regions={}
    for case in cases:
        region=code_mismatch_region(case['record'],case['conditional']['code']['first_target_mismatch'])
        regions[region]=regions.get(region,0)+1
    return {'total':len(cases),'code_first_conditional_mismatch_regions':regions,'generation':{mode:{m:sum(c['modes'][mode]['score'][m] for c in cases) for m in metrics} for mode in MODES},
       'conditional':{stage:{'hits':sum(c['conditional'][stage]['hits'] for c in cases),
                            'total':sum(c['conditional'][stage]['total'] for c in cases),
                            'exact_sequences':sum(c['conditional'][stage]['sequence_exact_under_gold_prefixes'] for c in cases)}
                      for stage in ('direct','specification','plan','code')}}


def run(archive,comparison,output):
    import torch
    from scripts.check_dense_runtime_parity import load_weights,torch_model
    from scripts.train_staged_protocol import TorchBackend
    from scripts.training_repro import state_digest
    first=source(archive,12149182157); second=source(comparison,12149930502)
    root=ROOT/'evaluation/staged-protocol-v1'
    manifest=json.loads((root/'manifest.json').read_bytes()); raw=(root/'train.json').read_bytes()
    if digest(raw)!=manifest['files']['train.json']['sha256']: raise ValueError('Frozen training records changed')
    records=json.loads(raw)
    if len(records)!=96 or any(not r['id'].startswith('train-') for r in records): raise ValueError('Diagnostic must use all 96 train records only')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True,exist_ok=True)
    report={'schema_version':1,'source_artifacts':SOURCES,'checkpoints':MODELS,'train_sha256':digest(raw),
        'runtime':{'torch':torch.__version__,'python':sys.version,'threads':2,'rmsnorm_epsilon':1e-5},
        'training_comparison':compare_training(first,second),'models':{},'optimization_performed':False,
        'generated_source_executed':False,'promotion':False,
        'limits':'All 96 cases are training diagnostics, not held-out quality. Gold specification/plan modes are assisted interventions. Conditional readouts provide correct target prefixes, including EOS; not autonomous success. Different call counts/inputs, no compute-matched causal accuracy claim.'}
    for arm,checkpoint_hash in MODELS.items():
        modelpath=output/(arm+'.nppai');modelpath.write_bytes(first[arm+'/experimental.nppai'])
        model=torch_model(load_weights(modelpath))
        for layer in model.modules():
            if isinstance(layer,torch.nn.RMSNorm): layer.eps=1e-5
        model.eval();before=state_digest(model); backend=TorchBackend(model,'sha256:'+checkpoint_hash)
        cases=[]
        for index,record in enumerate(records):
            modes={mode:intervention(backend,record,mode) for mode in MODES}
            conditional={stage:conditional_readout(model,*supervision(record,stage)) for stage in ('direct','specification','plan','code')}
            cases.append({'id':record['id'],'record':record,'modes':modes,'conditional':conditional})
            if index%16==0: print(f'{arm}: {index+1}/96',flush=True)
        retained=json.loads(first[arm+'/training_probe.json'])
        observed={c['id']:c for c in cases}; replay=[]
        for old in retained['cases']:
            now=observed[old['id']]['modes']; prior=old['arms']
            replay.append({'id':old['id'],'autonomous_output_equal':now['autonomous']['output']==prior['staged']['score']['output'],
                'direct_output_equal':now['direct']['output']==prior['direct_code_budget']['score']['output']})
        unchanged=state_digest(model)==before and digest(modelpath.read_bytes())==checkpoint_hash
        if not unchanged: raise ValueError('Frozen parameters/checkpoint changed')
        value={'cases':cases,'summary':summarize(cases),'retained_probe_replay':replay,
               'parameters_unchanged':unchanged,'state_sha256':before}
        report['models'][arm]=value
        (output/'report.json').write_bytes(encoded(report))
        print(json.dumps({'arm':arm,'summary':value['summary'],'all_retained_replies_replayed':all(p['autonomous_output_equal'] and p['direct_output_equal'] for p in replay)}),flush=True)
    (output/'source-comparison.json').write_bytes(encoded({'first':json.loads(first['training-report.json']),
        'second':json.loads(second['training-report.json']),
        'first_traces':{a:json.loads(first[a+'/trace.json']) for a in MODELS},
        'second_traces':{a:json.loads(second[a+'/trace.json']) for a in MODELS}}))
    report['source_comparison_sha256']=digest((output/'source-comparison.json').read_bytes())
    (output/'report.json').write_bytes(encoded(report))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    run(args.archive,args.comparison,args.output)
