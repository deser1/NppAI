"""Fixed-budget staged-code fitting with an authored direct-response replay arm."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage_failure_diagnostic import source,encoded,digest,intervention,conditional_readout
from scripts.train_staged_protocol import supervision,encoded_sample,TorchBackend,functional

ARMS=('code_only','code_with_direct_replay')
MODES=('direct','autonomous','gold_specification_and_plan')


def assignments(arm,step):
    if arm not in ARMS: raise ValueError('Unknown training arm')
    return ['code']*4 if arm=='code_only' else ['code' if (step+j)%4<2 else 'direct' for j in range(4)]


def train(initial,records,schedule,arm,output):
    import torch
    import torch.nn.functional as F
    model=copy.deepcopy(initial).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0003)
    samples={(i,s):encoded_sample(*supervision(r,s),1024) for i,r in enumerate(records) for s in ('direct','code')}
    trace=[];start=time.monotonic()
    for step,indices in enumerate(schedule):
        stages=assignments(arm,step)
        batch=[samples[(i,s)] for i,s in zip(indices,stages)]
        x=torch.nn.utils.rnn.pad_sequence([v[0] for v in batch],batch_first=True)
        y=torch.nn.utils.rnn.pad_sequence([v[1] for v in batch],batch_first=True,padding_value=-100)
        loss=F.cross_entropy(model(x).reshape(-1,256),y.reshape(-1),ignore_index=-100)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        trace.append({'step':step+1,'indices':indices,'stages':stages,'loss':float(loss.detach()),
                      'input_tokens':int((x!=0).sum()),'target_tokens':int((y!=-100).sum())})
        if step%128==0 or step==len(schedule)-1: print(f'{arm} {step+1}/{len(schedule)} loss={trace[-1]["loss"]:.5f}',flush=True)
    output.mkdir(parents=True,exist_ok=True);(output/'trace.json').write_bytes(encoded(trace))
    return model.eval(),{'steps':len(schedule),'elapsed_seconds':time.monotonic()-start,
        'initial_loss':trace[0]['loss'],'final_loss':trace[-1]['loss'],
        'input_tokens':sum(t['input_tokens'] for t in trace),'target_tokens':sum(t['target_tokens'] for t in trace),
        'trace_sha256':digest((output/'trace.json').read_bytes())}


def training_readout(backend,records):
    cases=[]
    for i,r in enumerate(records):
        cases.append({'id':r['id'],'record':r,'modes':{m:intervention(backend,r,m) for m in MODES},
                      'conditional_code':conditional_readout(backend.model,*supervision(r,'code'))})
        if i%32==0: print('training readout',i+1,'/',len(records),flush=True)
    return {'total':len(cases),'cases':cases,'summary':{
        m:{k:sum(c['modes'][m]['score'][k] for c in cases) for k in
           ('nonempty','syntax_any_function','requested_function','reference_ast','exact_response')} for m in MODES},
        'conditional_code_exact':sum(c['conditional_code']['sequence_exact_under_gold_prefixes'] for c in cases)}


def retention(before,after):
    result={}
    for mode in MODES:
        a={c['id']:c['modes'][mode]['score']['exact_response'] for c in before['cases']}
        b={c['id']:c['modes'][mode]['score']['exact_response'] for c in after['cases']}
        if a.keys()!=b.keys(): raise ValueError('Retention case set mismatch')
        result[mode]={'initial_exact':sum(a.values()),'retained':sum(a[k] and b[k] for k in a),
            'lost_ids':[k for k in a if a[k] and not b[k]],'new_ids':[k for k in a if not a[k] and b[k]]}
    return result


def run(archive,output,image=None):
    import torch
    from train_nppai import export_to_bin
    from scripts.check_dense_runtime_parity import torch_model,load_weights
    from scripts.training_repro import state_digest
    from scripts.evaluate_task_plan import evaluate
    from model_backends import DenseLocalBackend
    p=ROOT/'evaluation/staged-code-fit-v1.json'; protocol=json.loads(p.read_bytes())
    if (protocol['steps_per_arm'],protocol['batch_size'],protocol['seed'],protocol['learning_rate'])!=(512,4,43,0.0003):
        raise ValueError('Frozen budget differs')
    members=source(archive,12149182157);raw=members['staged_mix/experimental.nppai']
    if digest(raw)!=protocol['source_checkpoint']: raise ValueError('Initialization mismatch')
    root=ROOT/'evaluation/staged-protocol-v1';trainraw=(root/'train.json').read_bytes()
    manifest=json.loads((root/'manifest.json').read_bytes())
    if digest(trainraw)!=manifest['files']['train.json']['sha256']: raise ValueError('Training source mismatch')
    records=json.loads(trainraw)
    if len(records)!=96 or any(not r['id'].startswith('train-') for r in records): raise ValueError('Training must contain exactly the 96 train records')
    output.mkdir(parents=True,exist_ok=True);initpath=output/'initial.nppai';initpath.write_bytes(raw)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    initial=torch_model(load_weights(initpath))
    for layer in initial.modules():
        if isinstance(layer,torch.nn.RMSNorm): layer.eps=1e-5
    initial.eval();initial_state=state_digest(initial)
    rng=random.Random(43);schedule=[rng.sample(range(96),4) for _ in range(512)]
    freshpath=root/'new-evaluation.json';devpath=ROOT/'evaluation/instruction-v2/validation.json'
    evalhashes={str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in (freshpath,devpath)}
    report={'protocol':protocol,'protocol_sha256':digest(encoded(protocol)),'train_sha256':digest(trainraw),
        'initial_state_sha256':initial_state,'schedule_sha256':digest(encoded(schedule)),
        'runtime':{'python':sys.version,'torch':torch.__version__,'threads':2,'rmsnorm_epsilon':1e-5},
        'source_sha256':{n:digest((ROOT/n).read_bytes()) for n in ('scripts/staged_code_fit.py','scripts/stage_failure_diagnostic.py','scripts/train_staged_protocol.py','staged_generation.py')},
        'parameter_count':sum(v.numel() for v in initial.parameters()),'models':{},'promotion':False}
    before=training_readout(TorchBackend(initial,'sha256:'+digest(raw)),records)
    (output/'before.json').write_bytes(encoded(before))
    report['before_sha256']=digest((output/'before.json').read_bytes())
    models={}
    # Complete both fixed optimizations before reading regression task contents.
    for arm in ARMS:
        if state_digest(initial)!=initial_state: raise ValueError('Shared initialization mutated')
        model,stats=train(initial,records,schedule,arm,output/arm)
        path=output/arm/'experimental.nppai';export_to_bin(model,str(path))
        models[arm]=TorchBackend(model,'sha256:'+digest(path.read_bytes()))
        report['models'][arm]={**stats,'initial_state_sha256':initial_state,
            'final_state_sha256':state_digest(model),'checkpoint_sha256':digest(path.read_bytes())}
        (output/'report.json').write_bytes(encoded(report))
    for path,expected in evalhashes.items():
        if digest((ROOT/path).read_bytes())!=expected: raise ValueError('Evaluation changed during optimization')
    fresh=json.loads(freshpath.read_bytes());dev=json.loads(devpath.read_bytes())
    if image:
        import sandbox_validate
        controls=[]
        for r in fresh:
            status,diagnostics=sandbox_validate.validate(r['answer'],'tests',r['cases'],image)
            if status!=0: raise RuntimeError('Reference control failed: '+diagnostics)
            controls.append({'id':r['id'],'exit_code':status})
        status,_=sandbox_validate.validate('def suma(a,b):\n    return None\n','tests',fresh[0]['cases'],image)
        if status!=1: raise RuntimeError('Incorrect candidate control did not fail')
        report['validator_controls']={'references':controls,'incorrect_exit_code':status}
    for arm,backend in models.items():
        after=training_readout(backend,records);(output/arm/'training.json').write_bytes(encoded(after))
        report['models'][arm]['retention']=retention(before,after)
        print(json.dumps({'arm':arm,'training':after['summary'],'retention':report['models'][arm]['retention']}),flush=True)
        for group,data in (('exposed_validation',dev),('bilingual_regression',fresh)):
            measured=evaluate(backend,data);(output/arm/(group+'.json')).write_bytes(encoded(measured))
            print(json.dumps({'arm':arm,'group':group,'summary':measured['summary']}),flush=True)
            if image and group=='bilingual_regression':
                result=functional(measured,data,image);(output/arm/'functional.json').write_bytes(encoded(result))
        dense=DenseLocalBackend(str(output/arm/'experimental.nppai'));parity=[]
        for stage in ('direct','specification','plan','code'):
            payload,_=supervision(records[0],stage);a=backend.generate(payload);b=dense.generate(payload)
            parity.append({'stage':stage,'request':payload.model_dump(),'torch':a.model_dump(),'dense':b.model_dump(),'equal':a.model_dump()==b.model_dump()})
        (output/arm/'parity.json').write_bytes(encoded(parity))
        report['models'][arm]['four_exported_readouts_equal']=all(v['equal'] for v in parity)
        report['models'][arm]['evaluation_unchanged']=state_digest(backend.model)==report['models'][arm]['final_state_sha256']
        if not report['models'][arm]['evaluation_unchanged']: raise RuntimeError('Evaluation changed parameters')
    report['execution_status']='docker' if image else 'not_run';report['evaluation_sha256']=evalhashes
    (output/'report.json').write_bytes(encoded(report))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--image');args=p.parse_args()
    run(args.archive,args.output,args.image)
