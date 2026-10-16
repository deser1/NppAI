"""Locate first-step CPU numerical divergence at observed ATen operator boundaries."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.training_repro import PROFILES,configure,provenance,state_digest


def encoded(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False)+'\n').encode()


class TensorStore:
    def __init__(self,path,budget=128*1024*1024):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.budget=budget;self.bytes=0;self.seen=set()

    def put(self,raw,dtype,shape):
        header=encoded({'dtype':dtype,'shape':list(shape)})
        key=hashlib.sha256(header+raw).hexdigest()
        if key not in self.seen:
            if self.bytes+len(raw)>self.budget:raise ValueError('operator tensor snapshot budget exceeded')
            (self.path/(key+'.bin')).write_bytes(raw);self.bytes+=len(raw);self.seen.add(key)
        return {'tensor':key,'dtype':dtype,'shape':list(shape),'bytes':len(raw)}


def snapshot(value,store,read=True):
    import torch
    if isinstance(value,torch.Tensor):
        if value.device.type!='cpu' or value.layout!=torch.strided:raise ValueError('CPU dense tensors required')
        if not read:return {'dtype':str(value.dtype),'shape':list(value.shape),'stride':list(value.stride()),'not_read':'uninitialized allocation or write-only destination'}
        array=value.detach().contiguous().numpy()
        result=store.put(array.tobytes(),str(value.dtype),value.shape)
        result['stride']=list(value.stride())
        return result
    if isinstance(value,(list,tuple)):return {'sequence':type(value).__name__,'items':[snapshot(x,store,read) for x in value]}
    if isinstance(value,dict):return {str(k):snapshot(v,store,read) for k,v in sorted(value.items())}
    if isinstance(value,float):return {'float_hex':value.hex()}
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,(torch.dtype,torch.device,torch.layout,torch.memory_format)):return {'torch_value':str(value)}
    raise TypeError('unsupported operator argument '+str(type(value)))


def trace_mode(path):
    from torch.utils._python_dispatch import TorchDispatchMode
    class Recorder(TorchDispatchMode):
        def __init__(self):
            super().__init__();self.events=[];self.store=TensorStore(path/'tensors');self.phase='unset';self.ignored={}
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if not str(func).startswith('aten.'):
                self.ignored[str(func)]=self.ignored.get(str(func),0)+1
                return func(*args,**(kwargs or {}))
            write_only=str(func) in ('aten.zero_.default','aten.fill_.Scalar','aten.fill_.Tensor','aten.copy_.default')
            arguments={'sequence':'tuple','items':[snapshot(value,self.store,not (write_only and index==0)) for index,value in enumerate(args)]}
            keywords={key:snapshot(value,self.store,key!='out') for key,value in sorted((kwargs or {}).items())}
            event={'index':len(self.events),'phase':self.phase,'op':str(func),'schema':str(func._schema),
                   'args':arguments,'kwargs':keywords}
            output=func(*args,**(kwargs or {}))
            allocation=str(func).startswith(('aten.empty.','aten.empty_like.','aten.empty_strided.','aten.new_empty.','aten.new_empty_strided.'))
            event['output']=snapshot(output,self.store,not allocation);self.events.append(event)
            return output
    return Recorder()


def gradients(model):
    return {name:hashlib.sha256(p.grad.detach().contiguous().numpy().tobytes()).hexdigest()
            for name,p in model.named_parameters() if p.grad is not None}


def step(weights,x,y,profile,path=None):
    import torch
    import torch.nn.functional as F
    from scripts.check_dense_runtime_parity import torch_model
    configure(torch,profile);model=torch_model(weights).train();initial=state_digest(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    recorder=trace_mode(path) if path is not None else None
    with recorder if recorder else nullcontext():
        if recorder:recorder.phase='forward'
        logits=model(x)
        if recorder:recorder.phase='loss'
        loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1),ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        if recorder:recorder.phase='backward'
        loss.backward()
    raw_gradients=gradients(model)
    with recorder if recorder else nullcontext():
        if recorder:recorder.phase='clip'
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    clipped_gradients=gradients(model)
    with recorder if recorder else nullcontext():
        if recorder:recorder.phase='optimizer'
        optimizer.step()
    result={'initial_parameters':initial,'updated_parameters':state_digest(model),
            'loss_hex':float(loss.detach()).hex(),'raw_gradients':raw_gradients,'clipped_gradients':clipped_gradients,
            'logits':hashlib.sha256(logits.detach().contiguous().numpy().tobytes()).hexdigest()}
    if recorder:
        (path/'events.json').write_bytes(encoded(recorder.events))
        (path/'ignored-non-aten.json').write_bytes(encoded(recorder.ignored))
    return result


def first_difference(left,right):
    for index,(a,b) in enumerate(zip(left,right)):
        if a!=b:
            same_call=all(a.get(k)==b.get(k) for k in ('phase','op','schema'))
            same_inputs=all(a.get(k)==b.get(k) for k in ('args','kwargs'))
            return {'identical':False,'event':index,'same_call':same_call,'same_inputs':same_inputs,
                    'output_differs':a.get('output')!=b.get('output'),'left':a,'right':b,
                    'interpretation':'operator output differs for identical recorded inputs' if same_call and same_inputs else 'call/input divergence; not isolated kernel evidence'}
    return {'identical':len(left)==len(right),'event':None if len(left)==len(right) else min(len(left),len(right)),
            'counts':[len(left),len(right)]}


def diagnose(source,dataset,output,profile):
    import torch
    from scripts import resampled_name_training as base
    from scripts.run_local_model_smoke import pinned_backend
    output.mkdir(parents=True,exist_ok=True);configure(torch,profile)
    environment=provenance(torch,profile)
    backend=pinned_backend(source)
    if backend.model_id!=base.SOURCE_MODEL_ID:raise ValueError('pinned source differs')
    rows=base.training_data(dataset);selected=next(base.schedule(rows,'resampled'));samples=[base.sample(r) for _,r in selected]
    x=torch.nn.utils.rnn.pad_sequence([s[0] for s in samples],batch_first=True)
    y=torch.nn.utils.rnn.pad_sequence([s[1] for s in samples],batch_first=True,padding_value=-100)
    reference=step(backend.weights,x,y,profile);traced=[];events=[]
    for repeat in range(2):
        path=output/str(repeat);path.mkdir(parents=True,exist_ok=True)
        traced.append(step(backend.weights,x,y,profile,path));events.append(json.loads((path/'events.json').read_bytes()))
    result={'profile':profile,'environment':environment,'source_model_id':backend.model_id,
            'batch_sha256':base.digest(base.encoded(selected)),'x_sha256':base.digest(x.numpy().tobytes()),
            'y_sha256':base.digest(y.numpy().tobytes()),'reference':reference,'traced':traced,
            'trace_preserves_untraced_step':all(item==reference for item in traced),
            'trace_repeat':first_difference(*events),'steps':1,'promotion':False,
            'limits':'ATen boundaries observed by TorchDispatchMode, not all internal CPU instructions. Only first fixed training batch; no quality claim.'}
    (output/'report.json').write_bytes(encoded(result));print(json.dumps({k:v for k,v in result.items() if k not in ('environment','traced','reference')}))
    if not result['trace_preserves_untraced_step'] or not result['trace_repeat']['identical']:
        raise RuntimeError('tracing changed the step or same-host tracing differs')


def compare(root):
    results={}
    for profile in PROFILES:
        dirs=[root/f'first-step-{profile}-{i}' for i in range(1,5)]
        reports=[json.loads((p/'report.json').read_bytes()) for p in dirs]
        events=[json.loads((p/'0/events.json').read_bytes()) for p in dirs]
        if not all(r['trace_preserves_untraced_step'] and r['trace_repeat']['identical'] for r in reports):
            raise RuntimeError('unreliable operator trace')
        same_exposure=all(reports[0][key]==r[key] for r in reports[1:] for key in ('profile','source_model_id','batch_sha256','x_sha256','y_sha256'))
        if not same_exposure or any(reports[0]['reference']['initial_parameters']!=r['reference']['initial_parameters'] for r in reports[1:]):
            raise RuntimeError('source or batch differs')
        pairs=[{'replicas':[a+1,b+1],**first_difference(events[a],events[b])} for a in range(4) for b in range(a+1,4)]
        results[profile]={'pairs':pairs,'reports':reports,
            'coverage':{'cpu_models':[r['environment']['cpu'].get('model name') for r in reports],
                        'vendors':[r['environment']['cpu'].get('vendor_id') for r in reports]},
            'limits':'No divergence is also a valid diagnostic outcome; CPU allocation may not cover both vendors. No portability gate or promotion.'}
    (root/'comparison.json').write_bytes(encoded(results))
    print(json.dumps({profile:{k:v for k,v in result.items() if k!='reports'} for profile,result in results.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--compare',type=Path)
    for name in ('source','dataset','output'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--profile',choices=PROFILES);args=parser.parse_args()
    if args.compare:compare(args.compare)
    else:
        if not all((args.source,args.dataset,args.output,args.profile)):parser.error('source, dataset, output and profile required')
        diagnose(args.source,args.dataset,args.output,args.profile)
