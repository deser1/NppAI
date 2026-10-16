"""Trace fixed both64 forward, saved autograd tensors and backward; no promotion."""
import argparse
from contextlib import ExitStack
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.first_step_ops import encoded, first_difference, gradients, snapshot, trace_mode
from scripts.arithmetic_variants import arithmetic_mode, complete_step, digest
from scripts.training_repro import PROFILES, configure, provenance, state_digest


def traced_step(weights,x,y,profile,path):
    import torch
    import torch.nn.functional as F
    from scripts.check_dense_runtime_parity import torch_model
    configure(torch,profile);model=torch_model(weights).train();initial=state_digest(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    recorder=trace_mode(path);mode=arithmetic_mode('both64');saved=[]
    def pack(tensor):
        # Recording must not add incidental detach/contiguous operators to the trace.
        with torch._C._DisableTorchDispatch():
            record=snapshot(tensor,recorder.store)
        saved.append({'index':len(saved),'phase':recorder.phase,'after_events':len(recorder.events),'value':record})
        return tensor
    def phase(value):recorder.phase=value;mode.phase=value
    # Recorder is outermost dispatch handler: it observes effective substituted outputs.
    with ExitStack() as stack:
        stack.enter_context(mode);stack.enter_context(recorder)
        stack.enter_context(torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor))
        phase('forward');logits=model(x)
        phase('loss');loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1),ignore_index=-100)
        optimizer.zero_grad(set_to_none=True);phase('backward');loss.backward()
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):raise ValueError('nonfinite forward/loss')
    if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):raise ValueError('nonfinite gradient')
    raw=gradients(model)
    with mode,recorder:
        phase('clip');torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    clipped=gradients(model)
    with mode,recorder:
        phase('optimizer');optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError('nonfinite weights')
    state={'initial_parameters':initial,'updated_parameters':state_digest(model),'loss_hex':float(loss.detach()).hex(),
        'raw_gradients':raw,'clipped_gradients':clipped,'logits':digest(logits.detach().contiguous().numpy().tobytes())}
    for name,value in [('events',recorder.events),('saved',saved),('ignored-non-aten',recorder.ignored)]:
        (path/(name+'.json')).write_bytes(encoded(value))
    return {'state':state,'overrides':mode.overrides}


def diagnose(source,dataset,output,profile):
    import torch
    from scripts import resampled_name_training as base
    from scripts.run_local_model_smoke import pinned_backend
    configure(torch,profile);output.mkdir(parents=True,exist_ok=True)
    backend=pinned_backend(source)
    if backend.model_id!=base.SOURCE_MODEL_ID:raise ValueError('pinned source differs')
    batch=next(base.schedule(base.training_data(dataset),'resampled'));samples=[base.sample(r) for _,r in batch]
    x=torch.nn.utils.rnn.pad_sequence([s[0] for s in samples],batch_first=True)
    y=torch.nn.utils.rnn.pad_sequence([s[1] for s in samples],batch_first=True,padding_value=-100)
    reference=complete_step(backend.weights,x,y,profile,'both64');traced=[];events=[];saved=[]
    for repeat in range(2):
        path=output/str(repeat);path.mkdir(parents=True,exist_ok=True)
        traced.append(traced_step(backend.weights,x,y,profile,path))
        events.append(json.loads((path/'events.json').read_bytes()));saved.append(json.loads((path/'saved.json').read_bytes()))
    result={'version':1,'variant':'both64','profile':profile,'source_model_id':backend.model_id,
        'batch_sha256':digest(base.encoded(batch)),'x_sha256':digest(x.numpy().tobytes()),'y_sha256':digest(y.numpy().tobytes()),
        'environment':provenance(torch,profile),'reference':reference,'traced':traced,
        'trace_preserves_untraced_step':all(t==reference for t in traced),
        'trace_repeat':first_difference(*events),'saved_repeat':saved[0]==saved[1],
        'saved_counts':[len(s) for s in saved],'steps':1,'promotion':False}
    (output/'report.json').write_bytes(encoded(result))
    if not result['trace_preserves_untraced_step'] or not result['trace_repeat']['identical'] or not result['saved_repeat'] or not all(result['saved_counts']):
        raise RuntimeError('unreliable both64 trace or saved tensors')
    print(json.dumps({'profile':profile,'trace_preserves_untraced_step':True,'saved_counts':result['saved_counts']}))


def saved_difference(left,right):
    for i,(a,b) in enumerate(zip(left,right)):
        if a!=b:return {'identical':False,'index':i,'left':a,'right':b}
    return {'identical':len(left)==len(right),'counts':[len(left),len(right)]}


def compare(root):
    results={}
    for profile in PROFILES:
        dirs=[root/f'both64-report-{profile}-{i}' for i in range(1,5)]
        reports=[json.loads((p/'report.json').read_bytes()) for p in dirs]
        events=[json.loads((p/'0/events.json').read_bytes()) for p in dirs]
        saved=[json.loads((p/'0/saved.json').read_bytes()) for p in dirs]
        for r in reports:
            if r['version']!=1 or r['variant']!='both64' or r['profile']!=profile or r['steps']!=1 or r['promotion'] is not False:raise ValueError('trace schema differs')
            if not r['trace_preserves_untraced_step'] or not r['trace_repeat']['identical'] or not r['saved_repeat'] or not all(r['saved_counts']) or any(t!=r['reference'] for t in r['traced']):raise ValueError('unreliable trace')
        for r in reports[1:]:
            if any(r[k]!=reports[0][k] for k in ('source_model_id','batch_sha256','x_sha256','y_sha256')) or r['reference']['state']['initial_parameters']!=reports[0]['reference']['state']['initial_parameters']:raise ValueError('source or exposure differs')
        pairs=[]
        for a,b in itertools.combinations(range(4),2):
            av=reports[a]['environment']['cpu'].get('vendor_id');bv=reports[b]['environment']['cpu'].get('vendor_id')
            pairs.append({'replicas':[a+1,b+1],'cross_vendor':bool(av and bv and av!=bv),
                'first_operator_difference':first_difference(events[a],events[b]),
                'first_saved_difference':saved_difference(saved[a],saved[b]),
                'step':{k:reports[a]['reference']['state'][k]==reports[b]['reference']['state'][k] for k in reports[a]['reference']['state']}})
        results[profile]={'pairs':pairs,'reports':reports,'coverage':{'cpu_models':[r['environment']['cpu'].get('model name') for r in reports],
            'vendors':[r['environment']['cpu'].get('vendor_id') for r in reports]},'promotion':False,
            'limits':'Effective ATen boundaries and saved tensors only; no internal instruction/accuracy attribution. One fixed step, actual CPU allocation only; no full-training portability claim.'}
    (root/'comparison.json').write_bytes(encoded(results))
    print(json.dumps({p:{'coverage':r['coverage'],'pairs':[{k:v for k,v in pair.items() if k not in ('first_operator_difference','first_saved_difference')} for pair in r['pairs']]} for p,r in results.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--compare',type=Path)
    for name in ('source','dataset','output'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--profile',choices=PROFILES);args=parser.parse_args()
    if args.compare:compare(args.compare)
    else:
        if not all((args.source,args.dataset,args.output,args.profile)):parser.error('source, dataset, output and profile required')
        diagnose(args.source,args.dataset,args.output,args.profile)
