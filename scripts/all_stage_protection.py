"""Fixed own-model all-stage replay versus frozen planner/coder routing."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_backends import GenerationRequest
from scripts.stage_failure_diagnostic import source,encoded,digest,accepted_generate
from scripts.train_staged_protocol import supervision,encoded_sample,TorchBackend
from scripts.evaluate_staged_generation import score_output
from experimental_stage_router import run_routed

CODER_ARCHIVE='c017980508abc1f42cf3f5ea58b09ac21b32b1b9765605f33997afcc7715a7e7'
CODER='236cf897531431972120ce29a6272812dc593f187883ed06d34a9b7b99dfda76'
ARMS=('initial','all_stage_replay','frozen_planner_coder')
STAGES=('direct','specification','plan','code')


def objective_assignment(step):return [STAGES[(step+j)%4] for j in range(4)]


def train_all(initial,records,schedule,output):
    import torch
    import torch.nn.functional as F
    model=copy.deepcopy(initial).train();opt=torch.optim.AdamW(model.parameters(),lr=0.0003)
    samples={(i,s):encoded_sample(*supervision(r,s),1024) for i,r in enumerate(records) for s in STAGES}
    trace=[]
    for step,indices in enumerate(schedule):
        stages=objective_assignment(step);batch=[samples[i,s] for i,s in zip(indices,stages)]
        x=torch.nn.utils.rnn.pad_sequence([b[0] for b in batch],batch_first=True)
        y=torch.nn.utils.rnn.pad_sequence([b[1] for b in batch],batch_first=True,padding_value=-100)
        loss=F.cross_entropy(model(x).reshape(-1,256),y.reshape(-1),ignore_index=-100)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        trace.append({'step':step+1,'indices':indices,'stages':stages,'loss':float(loss.detach()),
                      'input_tokens':int((x!=0).sum()),'target_tokens':int((y!=-100).sum())})
        if step%128==0 or step==len(schedule)-1:print('all-stage',step+1,'loss',trace[-1]['loss'],flush=True)
    output.mkdir(parents=True,exist_ok=True);(output/'trace.json').write_bytes(encoded(trace));return model.eval()


def measure(planner,coder,records,gold=False):
    cases=[]
    for i,r in enumerate(records):
        payload=GenerationRequest(prompt=r['prompt'],max_new_tokens=128);response=coder.generate(payload)
        direct={'request':payload.model_dump(),'response':response.model_dump(),'score':score_output(r,response.output)}
        routed=run_routed(planner,coder,r['prompt'])
        value={'id':r['id'],'record':r,'direct':direct,'autonomous':{'trace':routed,'score':score_output(r,routed['code'])}}
        if gold:
            payload,_=supervision(r,'code');call,text=accepted_generate(coder,payload)
            value['gold_code']={'call':call,'score':score_output(r,text)}
        cases.append(value)
        if i%32==0:print('readout',i+1,'/',len(records),flush=True)
    return {'total':len(cases),'cases':cases,'summary':summarize(cases,gold)}


def summarize(cases,gold):
    modes=('direct','autonomous','gold_code') if gold else ('direct','autonomous')
    summary={m:{k:sum(c[m]['score'][k] for c in cases) for k in ('nonempty','syntax_any_function','requested_function','reference_ast','exact_response')} for m in modes}
    if gold:
        summary['artifact_exact']={s:sum(next((x['response']['output'] for x in c['autonomous']['trace']['stages'] if x['stage']==s and x['response']),None)==c['record'][s] for c in cases) for s in ('specification','plan')}
    return summary


def retained(before,after):
    result={}
    for mode in ('direct','autonomous','gold_code'):
        a={c['id']:c[mode]['score']['exact_response'] for c in before['cases']};b={c['id']:c[mode]['score']['exact_response'] for c in after['cases']}
        if a.keys()!=b.keys():raise ValueError('Retention case mismatch')
        result[mode]={'initial_exact':sum(a.values()),'retained':sum(a[k] and b[k] for k in a),
            'lost_ids':[k for k in a if a[k] and not b[k]],'new_ids':[k for k in a if not a[k] and b[k]]}
    return result


def artifact_retention(before,after):
    def hit(case,stage):
        value=next((s['response']['output'] for s in case['autonomous']['trace']['stages'] if s['stage']==stage and s['response']),None)
        return value==case['record'][stage]
    result={}
    for stage in ('specification','plan'):
        a={c['id']:hit(c,stage) for c in before['cases']};b={c['id']:hit(c,stage) for c in after['cases']}
        if a.keys()!=b.keys():raise ValueError('Artifact retention case mismatch')
        result[stage]={'initial_exact':sum(a.values()),'retained':sum(a[k] and b[k] for k in a),
            'lost_ids':[k for k in a if a[k] and not b[k]]}
    return result


def frozen_artifacts_equal(before,after):
    def prior(c):return [s for s in c['autonomous']['trace']['stages'] if s['stage']!='code']
    return all(a['id']==b['id'] and prior(a)==prior(b) for a,b in zip(before['cases'],after['cases'])) and len(before['cases'])==len(after['cases'])


def functional(measured,records,image):
    import sandbox_validate
    result={}
    for mode in ('direct','autonomous'):
        cases=[]
        for case,r in zip(measured['cases'],records):
            text=case[mode]['score']['output'];status=None;diagnostics=''
            if text.strip():
                status,diagnostics=sandbox_validate.validate(text,'tests',r['cases'],image)
                if status not in (0,1):raise RuntimeError('Validation infrastructure failed: '+diagnostics)
            cases.append({'id':r['id'],'source':text,'exit_code':status,'diagnostics':diagnostics,'passed':status==0})
        result[mode]={'total':len(records),'passed':sum(c['passed'] for c in cases),'cases':cases}
    return {'image':image,'modes':result}


def run(original,coder_archive,output,image=None):
    import torch
    from train_nppai import export_to_bin
    from scripts.check_dense_runtime_parity import load_weights,torch_model
    from scripts.training_repro import state_digest
    from model_backends import DenseLocalBackend
    protocol=json.loads((ROOT/'evaluation/all-stage-protection-v1.json').read_bytes())
    if (protocol['steps'],protocol['seed'],protocol['learning_rate'])!=(512,43,0.0003):raise ValueError('Frozen recipe changed')
    first=source(original,12149182157)
    if digest(coder_archive.read_bytes())!=CODER_ARCHIVE:raise ValueError('Pinned coder archive differs')
    with zipfile.ZipFile(coder_archive) as z:
        coder_bytes=z.read('code_with_direct_replay/experimental.nppai');coder_trace=json.loads(z.read('code_with_direct_replay/trace.json'));coder_report=json.loads(z.read('report.json'))
    if digest(coder_bytes)!=CODER:raise ValueError('Pinned coder checkpoint differs')
    output.mkdir(parents=True,exist_ok=True)
    def load(raw,name):
        p=output/(name+'.nppai');p.write_bytes(raw);m=torch_model(load_weights(p))
        for layer in m.modules():
            if isinstance(layer,torch.nn.RMSNorm):layer.eps=1e-5
        return m.eval(),TorchBackend(m,'sha256:'+digest(raw))
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    initial,base=load(first['staged_mix/experimental.nppai'],'initial');coder_model,coder=load(coder_bytes,'frozen-coder')
    initial_hash=state_digest(initial);coder_hash=state_digest(coder_model)
    root=ROOT/'evaluation/staged-protocol-v1';raw=(root/'train.json').read_bytes();manifest=json.loads((root/'manifest.json').read_bytes())
    if digest(raw)!=manifest['files']['train.json']['sha256']:raise ValueError('Training source changed')
    records=json.loads(raw)
    if len(records)!=96 or any(not r['id'].startswith('train-') for r in records):raise ValueError('Train-only supervision required')
    rng=random.Random(43);schedule=[rng.sample(range(96),4) for _ in range(512)]
    if schedule!=[t['indices'] for t in coder_trace] or coder_report['initial_state_sha256']!=initial_hash:raise ValueError('Historical coder initialization/schedule differs')
    evalpaths={'validation':ROOT/'evaluation/instruction-v2/validation.json','bilingual':root/'new-evaluation.json'}
    fingerprints={k:digest(p.read_bytes()) for k,p in evalpaths.items()}
    (output/'historical-coder.json').write_bytes(encoded({'report':coder_report,'trace':coder_trace}))
    report={'historical_coder_sha256':digest((output/'historical-coder.json').read_bytes()),'protocol':protocol,'protocol_sha256':digest(encoded(protocol)),'train_sha256':digest(raw),
      'initial_state_sha256':initial_hash,'frozen_coder_state_sha256':coder_hash,'schedule_sha256':digest(encoded(schedule)),
      'coder_archive_sha256':CODER_ARCHIVE,'source_archive_sha256':digest(original.read_bytes()),
      'runtime':{'python':sys.version,'torch':torch.__version__,'threads':2,'rmsnorm_epsilon':1e-5},
      'models':{},'promotion':False,'source_sha256':{n:digest((ROOT/n).read_bytes()) for n in ('scripts/all_stage_protection.py','experimental_stage_router.py')}}
    before=measure(base,base,records,True);(output/'initial-training.json').write_bytes(encoded(before))
    trained=train_all(initial,records,schedule,output/'all_stage_replay');export_to_bin(trained,str(output/'all_stage_replay/experimental.nppai'))
    trained_id=digest((output/'all_stage_replay/experimental.nppai').read_bytes());new=TorchBackend(trained,'sha256:'+trained_id)
    backends={'initial':(base,base),'all_stage_replay':(new,new),'frozen_planner_coder':(base,coder)}
    for name,(planner,decoder) in backends.items():
        folder=output/name;folder.mkdir(exist_ok=True)
        measured=before if name=='initial' else measure(planner,decoder,records,True)
        (folder/'training.json').write_bytes(encoded(measured))
        report['models'][name]={'planner_id':planner.model_id,'coder_id':decoder.model_id,'retention':retained(before,measured),
          'training_summary':measured['summary'],'artifact_retention':artifact_retention(before,measured),'loaded_parameters':sum(p.numel() for p in planner.model.parameters())+(sum(p.numel() for p in decoder.model.parameters()) if planner is not decoder else 0)}
        if name=='frozen_planner_coder':
            report['models'][name]['all_training_planner_calls_equal_initial']=frozen_artifacts_equal(before,measured)
            if not report['models'][name]['all_training_planner_calls_equal_initial']:raise ValueError('Frozen planner responses changed')
        print(json.dumps({'arm':name,'training':measured['summary'],'retention':report['models'][name]['retention']}),flush=True)
    if state_digest(initial)!=initial_hash or state_digest(coder_model)!=coder_hash:raise ValueError('Frozen source weights changed')
    if any(digest(p.read_bytes())!=fingerprints[k] for k,p in evalpaths.items()):raise ValueError('Regression inputs changed')
    evalrecords={k:json.loads(p.read_bytes()) for k,p in evalpaths.items()}
    if image:
        import sandbox_validate
        controls=[]
        for r in evalrecords['bilingual']:
            status,diagnostics=sandbox_validate.validate(r['answer'],'tests',r['cases'],image)
            if status!=0:raise RuntimeError('Reference validator control failed: '+diagnostics)
            controls.append({'id':r['id'],'exit_code':status})
        bad,_=sandbox_validate.validate('def suma(a,b):\n    return None\n','tests',evalrecords['bilingual'][0]['cases'],image)
        if bad!=1:raise RuntimeError('Incorrect candidate accepted')
        report['validator_controls']={'references':controls,'incorrect_exit_code':bad}
    snapshots={name:(state_digest(p.model),state_digest(c.model)) for name,(p,c) in backends.items()}
    for name,(planner,decoder) in backends.items():
        for group,rows in evalrecords.items():
            measured=measure(planner,decoder,rows);(output/name/(group+'.json')).write_bytes(encoded(measured))
            print(json.dumps({'arm':name,'group':group,'summary':measured['summary']}),flush=True)
            if group=='bilingual' and image:(output/name/'functional.json').write_bytes(encoded(functional(measured,rows,image)))
        if snapshots[name]!=(state_digest(planner.model),state_digest(decoder.model)):raise ValueError('Evaluation changed weights')
    parity=[];dense=DenseLocalBackend(str(output/'all_stage_replay/experimental.nppai'))
    for stage in STAGES:
        payload,_=supervision(records[0],stage);a=new.generate(payload);b=dense.generate(payload)
        parity.append({'stage':stage,'request':payload.model_dump(),'torch':a.model_dump(),'dense':b.model_dump(),'equal':a.model_dump()==b.model_dump()})
    (output/'parity.json').write_bytes(encoded(parity))
    report['new_checkpoint_sha256']=trained_id;report['four_exported_readouts_equal']=all(p['equal'] for p in parity)
    report['evaluation_unchanged']=True;report['frozen_sources_unchanged']=state_digest(initial)==initial_hash and state_digest(coder_model)==coder_hash
    report['evaluation_sha256']=fingerprints;report['execution_status']='docker' if image else 'not_run'
    (output/'report.json').write_bytes(encoded(report));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',type=Path,required=True);p.add_argument('--coder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--image');a=p.parse_args();run(a.original,a.coder,a.output,a.image)
