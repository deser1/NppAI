"""Fixed mixed-precision operator probes; no baseline replacement or weight promotion."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.first_step_ops import encoded,gradients
from scripts.training_repro import PROFILES,configure,provenance,state_digest

VARIANTS=('native','softmax64','sqrt64','both64')
SPECS={
 'softmax':{'source':'softmax/0/tensors/7aca10a0ed8a06967526792b606d0f546d9a2f7c3c33725b601a2aa98884633d.bin',
   'shape':[4,177,177],'bytes':501264,'sha256':'e08a5eb647e691cdb550c3c06ac5e681ab4f70784f3b02ed9aa6c95764e2ea1a',
   'tensor_key':'7aca10a0ed8a06967526792b606d0f546d9a2f7c3c33725b601a2aa98884633d'},
 'sqrt':{'source':'sqrt/0/tensors/57db9820a623c32409b33c3dfdb7f364ad26aafeb8145b401ee42c220f890a62.bin',
   'shape':[256,64],'bytes':65536,'sha256':'c9a9a4c682f4567eee882365eaeecc3fcdd94720f754c29b2648c4e6e9d6c477',
   'tensor_key':'57db9820a623c32409b33c3dfdb7f364ad26aafeb8145b401ee42c220f890a62'}}


def digest(raw):return hashlib.sha256(raw).hexdigest()


def validate_input(raw,spec):
    if len(raw)!=spec['bytes'] or digest(raw)!=spec['sha256']:
        raise ValueError('frozen operator input size or digest differs')
    if digest(encoded({'dtype':'torch.float32','shape':spec['shape']})+raw)!=spec['tensor_key']:
        raise ValueError('frozen operator input tensor key differs')


def prepare(root,output):
    output.mkdir(parents=True,exist_ok=True)
    for name,spec in SPECS.items():
        source=root/spec['source']
        if source.stat().st_size!=spec['bytes']:raise ValueError('frozen input file size differs')
        raw=source.read_bytes();validate_input(raw,spec);(output/(name+'.bin')).write_bytes(raw)
    (output/'manifest.json').write_bytes(encoded(SPECS))


def load_inputs(root):
    import numpy as np
    import torch
    if json.loads((root/'manifest.json').read_bytes())!=SPECS:raise ValueError('operator manifest differs')
    result={}
    for name,spec in SPECS.items():
        path=root/(name+'.bin')
        if path.stat().st_size!=spec['bytes']:raise ValueError('operator input file size differs')
        raw=path.read_bytes();validate_input(raw,spec)
        result[name]=torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy().reshape(spec['shape']))
    return result


def arithmetic_mode(variant):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    if variant not in VARIANTS:raise ValueError('unknown arithmetic variant')
    class Mode(TorchDispatchMode):
        def __init__(self):super().__init__();self.phase='unset';self.overrides={}
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {};name=str(func)
            soft=variant in ('softmax64','both64');square=variant in ('sqrt64','both64')
            if soft and name=='aten._softmax.default' and self.phase=='forward' and args[0].dtype==torch.float32:
                x,dim,half=args
                if half or kwargs:raise ValueError('unsupported softmax call')
                self.overrides[name]=self.overrides.get(name,0)+1
                return func(x.to(torch.float64),dim,False).to(torch.float32)
            if soft and name=='aten._softmax_backward_data.default' and self.phase=='backward' and args[0].dtype==torch.float32:
                grad,output,dim,input_dtype=args
                if input_dtype!=torch.float32 or kwargs:raise ValueError('unsupported softmax backward call')
                self.overrides[name]=self.overrides.get(name,0)+1
                return func(grad.to(torch.float64),output.to(torch.float64),dim,torch.float64).to(torch.float32)
            if square and name=='aten.sqrt.default' and self.phase=='optimizer' and args[0].dtype==torch.float32:
                if len(args)!=1 or kwargs:raise ValueError('unsupported optimizer sqrt call')
                self.overrides[name]=self.overrides.get(name,0)+1
                return func(args[0].to(torch.float64)).to(torch.float32)
            return func(*args,**kwargs)
    return Mode()


def tensor_record(tensor,output):
    raw=tensor.detach().contiguous().numpy().tobytes()
    key=digest(encoded({'dtype':str(tensor.dtype),'shape':list(tensor.shape)})+raw)
    path=output/'arrays';path.mkdir(parents=True,exist_ok=True)
    (path/(key+'.bin')).write_bytes(raw)
    return {'tensor':key,'shape':list(tensor.shape),'dtype':str(tensor.dtype),'bytes':len(raw)}


def isolated(inputs,variant,output):
    import torch
    mode=arithmetic_mode(variant)
    x=inputs['softmax'];n=x.numel()
    # Exact binary fractions, no transcendental or random gradient generation.
    grad=((torch.arange(n,dtype=torch.int64)%17-8).to(torch.float32)/8).reshape(x.shape)
    with mode:
        mode.phase='forward';prob=torch.ops.aten._softmax.default(x,-1,False)
        mode.phase='backward';back=torch.ops.aten._softmax_backward_data.default(grad,prob,-1,torch.float32)
        mode.phase='optimizer';square=torch.ops.aten.sqrt.default(inputs['sqrt'])
    values={'softmax_forward':prob,'softmax_backward':back,'sqrt':square}
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):raise ValueError('nonfinite isolated result')
    return {'outputs':{name:tensor_record(value,output) for name,value in values.items()},'overrides':mode.overrides}


def complete_step(weights,x,y,profile,variant):
    import torch
    import torch.nn.functional as F
    from scripts.check_dense_runtime_parity import torch_model
    configure(torch,profile);model=torch_model(weights).train();initial=state_digest(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001);mode=arithmetic_mode(variant)
    with mode:
        mode.phase='forward';logits=model(x)
        mode.phase='loss';loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1),ignore_index=-100)
        optimizer.zero_grad(set_to_none=True);mode.phase='backward';loss.backward()
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):raise ValueError('nonfinite forward/loss')
    if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):raise ValueError('nonfinite gradient')
    raw_gradients=gradients(model)
    with mode:
        mode.phase='clip';torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    clipped=gradients(model)
    with mode:
        mode.phase='optimizer';optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError('nonfinite updated weights')
    state={'initial_parameters':initial,'updated_parameters':state_digest(model),'loss_hex':float(loss.detach()).hex(),
        'raw_gradients':raw_gradients,'clipped_gradients':clipped,'logits':digest(logits.detach().contiguous().numpy().tobytes())}
    return {'state':state,'overrides':mode.overrides}


def run(source,dataset,inputs,output,profile):
    import torch
    from scripts import resampled_name_training as base
    from scripts.first_step_ops import step
    from scripts.run_local_model_smoke import pinned_backend
    configure(torch,profile);output.mkdir(parents=True,exist_ok=True);frozen=load_inputs(inputs)
    backend=pinned_backend(source)
    if backend.model_id!=base.SOURCE_MODEL_ID:raise ValueError('pinned source differs')
    batch=next(base.schedule(base.training_data(dataset),'resampled'));samples=[base.sample(r) for _,r in batch]
    x=torch.nn.utils.rnn.pad_sequence([s[0] for s in samples],batch_first=True)
    y=torch.nn.utils.rnn.pad_sequence([s[1] for s in samples],batch_first=True,padding_value=-100)
    reference=step(backend.weights,x,y,profile);arms={}
    for variant in VARIANTS:
        repeats=[]
        for _ in range(2):
            repeats.append({'isolated':isolated(frozen,variant,output),'complete':complete_step(backend.weights,x,y,profile,variant)})
        if repeats[0]!=repeats[1]:raise RuntimeError('same-host arithmetic repeat differs')
        state=repeats[0]['complete']['state']
        if state['initial_parameters']!=reference['initial_parameters']:raise RuntimeError('initial weights differ')
        if variant=='native' and state!=reference:raise RuntimeError('native mode changed the reference update')
        calls=repeats[0]['complete']['overrides']
        if variant in ('softmax64','both64') and not all(calls.get(k,0)>0 for k in ('aten._softmax.default','aten._softmax_backward_data.default')):
            raise RuntimeError('missing full-step softmax replacement')
        if variant in ('sqrt64','both64') and not calls.get('aten.sqrt.default',0):raise RuntimeError('missing optimizer sqrt replacement')
        arms[variant]={'repeats':repeats,'same_host_identical':True,'updated_equals_native':state['updated_parameters']==reference['updated_parameters']}
    import numpy as np
    native=arms['native']['repeats'][0]['isolated']['outputs']
    for variant,arm in arms.items():
        deltas={}
        for kind,record in arm['repeats'][0]['isolated']['outputs'].items():
            a=np.frombuffer((output/'arrays'/(native[kind]['tensor']+'.bin')).read_bytes(),dtype='<f4')
            b=np.frombuffer((output/'arrays'/(record['tensor']+'.bin')).read_bytes(),dtype='<f4')
            deltas[kind]={'different_bits':int(np.count_nonzero(a.view('<u4')!=b.view('<u4'))),
                          'different_values':int(np.count_nonzero(a!=b)),
                          'max_abs_delta':float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))}
        arm['isolated_delta_from_native']=deltas
    report={'version':1,'profile':profile,'environment':provenance(torch,profile),'source_model_id':backend.model_id,
        'batch_sha256':base.digest(base.encoded(batch)),'x_sha256':digest(x.numpy().tobytes()),'y_sha256':digest(y.numpy().tobytes()),
        'operator_inputs':SPECS,'reference':reference,'arms':arms,'steps':1,'promotion':False,
        'limits':'Four predeclared arithmetic variants, first fixed batch only. FP64 intermediates return FP32. Backward promotes saved FP32 softmax output, not an unsaved FP64 result. No full-training, native compatibility, performance or model-quality claim.'}
    (output/'report.json').write_bytes(encoded(report));print(json.dumps({'profile':profile,'same_host_identical':True,'native_matches_reference':True,'steps':1}))


def agreement(a,b):
    kinds=('softmax_forward','softmax_backward','sqrt')
    isolated={kind:a['isolated']['outputs'][kind]==b['isolated']['outputs'][kind] for kind in kinds}
    left,right=a['complete']['state'],b['complete']['state']
    stages={key:left[key]==right[key] for key in ('initial_parameters','logits','loss_hex','raw_gradients','clipped_gradients','updated_parameters')}
    return {'isolated':isolated,'step':stages,'complete_step_identical':left==right}


def compare(root):
    result={}
    for profile in PROFILES:
        reports=[json.loads((root/f'arithmetic-{profile}-{i}'/'report.json').read_bytes()) for i in range(1,5)]
        for report in reports:
            if report['version']!=1 or report['profile']!=profile or report['steps']!=1 or report['promotion'] is not False or set(report['arms'])!=set(VARIANTS):raise ValueError('probe schema differs')
            for variant,arm in report['arms'].items():
                if not arm['same_host_identical'] or len(arm['repeats'])!=2 or arm['repeats'][0]!=arm['repeats'][1]:raise ValueError('invalid same-host repeat')
                if variant=='native' and arm['repeats'][0]['complete']['state']!=report['reference']:raise ValueError('native reference differs')
        for report in reports[1:]:
            if any(report[k]!=reports[0][k] for k in ('source_model_id','batch_sha256','x_sha256','y_sha256','operator_inputs')) or report['reference']['initial_parameters']!=reports[0]['reference']['initial_parameters']:
                raise ValueError('source or exposure differs')
        pairs=[]
        for a,b in itertools.combinations(range(4),2):
            pairs.append({'replicas':[a+1,b+1],
                'cross_vendor':reports[a]['environment']['cpu'].get('vendor_id')!=reports[b]['environment']['cpu'].get('vendor_id'),
                'variants':{variant:agreement(reports[a]['arms'][variant]['repeats'][0],reports[b]['arms'][variant]['repeats'][0]) for variant in VARIANTS}})
        result[profile]={'coverage':{'cpu_models':[r['environment']['cpu'].get('model name') for r in reports],
            'vendors':[r['environment']['cpu'].get('vendor_id') for r in reports]},'pairs':pairs,
            'summary':{variant:{'all_steps_identical':all(p['variants'][variant]['complete_step_identical'] for p in pairs),
                'all_isolated_identical':all(all(p['variants'][variant]['isolated'].values()) for p in pairs),
                'cross_vendor_pairs':sum(p['cross_vendor'] for p in pairs)} for variant in VARIANTS},
            'promotion':False,'limits':'Observed first-step agreement only; actual CPU coverage required for interpretation. No automatic recipe replacement.'}
    (root/'comparison.json').write_bytes(encoded(result));print(json.dumps({p:{k:v for k,v in r.items() if k!='pairs'} for p,r in result.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path);parser.add_argument('--compare',type=Path)
    for name in ('source','dataset','inputs','output'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--profile',choices=PROFILES);args=parser.parse_args()
    if args.compare:compare(args.compare)
    elif args.prepare:
        if not args.output:parser.error('output required')
        prepare(args.prepare,args.output)
    else:
        if not all((args.source,args.dataset,args.inputs,args.output,args.profile)):parser.error('source, dataset, inputs, output and profile required')
        run(args.source,args.dataset,args.inputs,args.output,args.profile)
