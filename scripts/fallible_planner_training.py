"""Paired coder fitting with authored versus frozen model-produced priors.

Only existing training records can supply supervision. The planner stays frozen;
no generated specification/plan is promoted to a correct target.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_backends import GenerationRequest,format_prompt
from staged_generation import stage_values
from scripts.stage_failure_diagnostic import source,digest,encoded,MODELS
from scripts.train_staged_protocol import supervision,encoded_sample,TorchBackend
from scripts.all_stage_protection import measure,retained,artifact_retention,functional,frozen_artifacts_equal


def training_priors(records,cases,model_id):
    expected={r['id']:r for r in records}
    if len(expected)!=96 or any(not k.startswith('train-') for k in expected):
        raise ValueError('Exactly the 96 authored training records are required')
    if len(cases)!=96 or {c['id'] for c in cases}!=set(expected):
        raise ValueError('Training-only case set mismatch')
    rows=[]
    for c in cases:
        r=expected[c['id']]
        if c['record']!=r:raise ValueError('Retained training reference changed')
        stages=c['autonomous']['trace']['stages'];prior={}
        for name in ('specification','plan'):
            entry=next((s for s in stages if s['stage']==name),None)
            if not entry or entry['error'] or not entry['response']:break
            response=entry['response']
            if response['model_id']!=model_id:raise ValueError('Planner identity changed')
            if response['finish_reason']!='eos' or not response['output'].strip() or response['generated_tokens']>128:break
            prior[name]=response['output']
        if len(prior)==2:
            payload=GenerationRequest(**stage_values('code',r['prompt'],'',prior,128))
            code=next((s for s in stages if s['stage']=='code'),None)
            if not code or code['request']!=payload.model_dump():raise ValueError('Retained code request differs from prior artifacts')
            reason='observed_prior'
            if len(format_prompt(payload).encode())+len(r['answer'].encode())+1>1024:
                payload,_=supervision(r,'code');reason='authored_fallback_context'
        else:
            payload,_=supervision(r,'code');reason='authored_fallback_incomplete_planner'
        rows.append({'id':r['id'],'request':payload.model_dump(),'target':r['answer'],
                     'source':reason,'target_origin':'authored_training_answer'})
    return rows


def fit(initial,records,priors,schedule,arm,output):
    import torch
    import torch.nn.functional as F
    model=copy.deepcopy(initial).train();optimizer=torch.optim.AdamW(model.parameters(),lr=0.0003)
    observed={r['id']:r for r in priors};samples={}
    for i,r in enumerate(records):
        samples[i,'direct']=encoded_sample(*supervision(r,'direct'),1024)
        code=observed[r['id']]
        payload=GenerationRequest.model_validate(code['request']) if arm=='observed_priors' else supervision(r,'code')[0]
        samples[i,'code']=encoded_sample(payload,r['answer'],1024)
    trace=[]
    for step,indices in enumerate(schedule):
        stages=['code' if (step+j)%4<2 else 'direct' for j in range(4)]
        batch=[samples[i,s] for i,s in zip(indices,stages)]
        x=torch.nn.utils.rnn.pad_sequence([b[0] for b in batch],batch_first=True)
        y=torch.nn.utils.rnn.pad_sequence([b[1] for b in batch],batch_first=True,padding_value=-100)
        loss=F.cross_entropy(model(x).reshape(-1,256),y.reshape(-1),ignore_index=-100)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        trace.append({'step':step+1,'indices':indices,'stages':stages,'loss':float(loss.detach()),
                      'input_tokens':int((x!=0).sum()),'target_tokens':int((y!=-100).sum())})
        if step%128==0 or step==511:print(arm,step+1,'loss',trace[-1]['loss'],flush=True)
    output.mkdir(parents=True,exist_ok=True);(output/'trace.json').write_bytes(encoded(trace))
    return model.eval()


def run(archive,output,image=None):
    import torch
    from train_nppai import export_to_bin
    from scripts.check_dense_runtime_parity import load_weights,torch_model
    from scripts.training_repro import state_digest
    protocol=json.loads((ROOT/'evaluation/fallible-planner-v1.json').read_bytes())
    if (protocol['steps'],protocol['seed'],protocol['batch_size'],protocol['learning_rate'],protocol['context_bytes'])!=(512,43,4,0.0003,1024):raise ValueError('Frozen budget changed')
    source_files=source(archive,12149182157)
    evidence=ROOT/'docs/evidence/staged-protocol/all-stage-records.zip'
    manifest=json.loads((ROOT/'docs/evidence/staged-protocol/all-stage-source.json').read_bytes())
    if digest(evidence.read_bytes())!=manifest['records_sha256']:raise ValueError('Retained archive changed')
    with zipfile.ZipFile(evidence) as z:raw=z.read('initial/training.json')
    if digest(raw)!=manifest['members']['initial/training.json']:raise ValueError('Retained raw cases changed')
    training=ROOT/'evaluation/staged-protocol-v1/train.json';trainraw=training.read_bytes()
    frozen=json.loads((training.parent/'manifest.json').read_bytes())
    if digest(trainraw)!=frozen['files']['train.json']['sha256']:raise ValueError('Train split changed')
    records=json.loads(trainraw);priors=training_priors(records,json.loads(raw)['cases'],'sha256:'+MODELS['staged_mix'])
    output.mkdir(parents=True,exist_ok=True);(output/'training-priors.json').write_bytes(encoded(priors))
    checkpoint=output/'initial.nppai';checkpoint.write_bytes(source_files['staged_mix/experimental.nppai'])
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    initial=torch_model(load_weights(checkpoint))
    for layer in initial.modules():
        if isinstance(layer,torch.nn.RMSNorm):layer.eps=1e-5
    initial.eval();snapshot=state_digest(initial);planner=TorchBackend(initial,'sha256:'+MODELS['staged_mix'])
    rng=random.Random(43);schedule=[rng.sample(range(96),4) for _ in range(512)]
    paths={'validation':ROOT/'evaluation/instruction-v2/validation.json','bilingual':training.parent/'new-evaluation.json'}
    hashes={k:digest(p.read_bytes()) for k,p in paths.items()}
    report={'protocol':protocol,'train_sha256':digest(trainraw),'priors_sha256':digest(encoded(priors)),
        'source_counts':{s:sum(p['source']==s for p in priors) for s in sorted({p['source'] for p in priors})},
        'schedule_sha256':digest(encoded(schedule)),'initial_state_sha256':snapshot,
        'runtime':{'python':sys.version,'torch':torch.__version__},'arms':{},'promotion':False,
        'execution_status':'docker' if image else 'not_run'}
    before=measure(planner,planner,records,True);(output/'before.json').write_bytes(encoded(before))
    models={}
    for arm in ('authored_priors','observed_priors'):
        model=fit(initial,records,priors,schedule,arm,output/arm)
        export_to_bin(model,str(output/arm/'experimental.nppai'))
        models[arm]=TorchBackend(model,'sha256:'+digest((output/arm/'experimental.nppai').read_bytes()))
    if state_digest(initial)!=snapshot:raise ValueError('Frozen planner changed')
    if any(digest(p.read_bytes())!=hashes[k] for k,p in paths.items()):raise ValueError('Regression split changed')
    evals={k:json.loads(p.read_bytes()) for k,p in paths.items()}
    if image:
        import sandbox_validate
        controls=[]
        for r in evals['bilingual']:
            status,diagnostics=sandbox_validate.validate(r['answer'],'tests',r['cases'],image)
            if status!=0:raise RuntimeError('Reference validator failed: '+diagnostics)
            controls.append({'id':r['id'],'exit_code':status})
        status,_=sandbox_validate.validate('def suma(a,b):\n    return None\n','tests',evals['bilingual'][0]['cases'],image)
        if status!=1:raise RuntimeError('Incorrect control accepted')
        report['validator_controls']={'references':controls,'incorrect_exit_code':status}
    for arm,coder in models.items():
        model_hash=state_digest(coder.model)
        measured=measure(planner,coder,records,True);(output/arm/'training.json').write_bytes(encoded(measured))
        if not frozen_artifacts_equal(before,measured):raise ValueError('Planner calls changed')
        report['arms'][arm]={'training':measured['summary'],'retention':retained(before,measured),
                            'planner_retention':artifact_retention(before,measured),'checkpoint':coder.model_id}
        print(arm,json.dumps(report['arms'][arm]),flush=True)
        for group,rows in evals.items():
            data=measure(planner,coder,rows);(output/arm/(group+'.json')).write_bytes(encoded(data))
            report['arms'][arm][group]=data['summary']
            if group=='bilingual' and image:
                result=functional(data,rows,image);(output/arm/'functional.json').write_bytes(encoded(result))
                report['arms'][arm]['functional']={k:{'total':v['total'],'passed':v['passed']} for k,v in result['modes'].items()}
        if state_digest(coder.model)!=model_hash or state_digest(initial)!=snapshot:raise ValueError('Readout changed weights')
    report['evaluation_sha256']=hashes
    (output/'report.json').write_bytes(encoded(report));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--image');a=p.parse_args();run(a.archive,a.output,a.image)
