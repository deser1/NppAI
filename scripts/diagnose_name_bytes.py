"""Frozen conditional name-byte and scaffolded decoding diagnostic; no training."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import resampled_name_training as training
from scripts.run_local_model_smoke import pinned_backend

digest,encoded=training.digest,training.encoded
MODELS=('initial','fixed','resampled')
REPORT_HASHES={'fixed':'4eb529a6368ea2bd978b053a7a086dfb0445dd84a778bc6987b33c3ddff12d28',
               'resampled':'0fbbfda5766f67327083723cf46aec41a2e2014331d68184cc742155e15bf6b2'}
TRAINING_RUN=38804571444
MODE='gold_prefix_name_bytes_and_separate_def_scaffolded_decode'


def build(dataset):
    rows=training.training_data(dataset)
    groups={};names=set()
    for policy in training.POLICIES:
        selected=[];original=[]
        for batch in training.schedule(rows,policy):
            for pool,r in batch:
                if pool=='renamed':
                    names.add(r['function'])
                    if len(selected)<16:selected.append(r)
                elif policy=='fixed' and len(original)<16:original.append(r)
        groups['train/'+policy]=selected
        if policy=='fixed':groups['train/original']=original
    development=training.development_data(dataset)
    groups.update(development)
    for split in ('validation','test'):
        matched=[training.rename(r,'nc_'+r['function'][3:]) for r in development[split+'/renamed']]
        if names & {r['function'] for r in matched}:
            raise ValueError('matched-prefix development alias appeared in training')
        groups[split+'/matched_prefix']=matched
    return groups


def plan(record):
    if record['function'] not in record['prompt']:
        raise ValueError('name absent from prompt')
    # Only prompt and the fixed scaffold precede the first predicted byte.
    prefix=('[USER]: '+record['prompt']+'\n[AI]:\ndef ').encode()
    target=record['function'].encode()
    if not target or len(prefix)+max(32,len(target))>256 or b'\0' in prefix+target:
        raise ValueError('invalid name/context')
    return prefix,target


def metrics(target,predictions,probabilities,greedy):
    if len(predictions)!=len(target) or len(probabilities)!=len(target):
        raise ValueError('name-byte count differs')
    if any(type(v)!=int or not 0<=v<256 for v in predictions+greedy) or len(greedy)>32:
        raise ValueError('invalid byte prediction')
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in probabilities):
        raise ValueError('invalid probability')
    correct=[a==b for a,b in zip(target,predictions)]
    return {'name_bytes':len(target),'correct_bytes':sum(correct),
            'namespace_bytes':min(3,len(target)),'namespace_correct':sum(correct[:3]),
            'suffix_bytes':max(0,len(target)-3),'suffix_correct':sum(correct[3:]),
            'all_name_bytes_correct':int(all(correct)),
            'target_probability_sum':sum(probabilities),
            'nll_sum':sum(-math.log(max(p,1e-30)) for p in probabilities),
            'scaffolded_exact_name':int(greedy==list(target+b'('))}


def infer(model,prefix,target):
    import torch
    raw=prefix+target
    full=model(torch.tensor([list(raw[:-1])],dtype=torch.long))[0]
    start=len(prefix)-1
    logits=full[start:]
    predictions=logits.argmax(-1).tolist()
    probabilities=logits.softmax(-1).gather(1,torch.tensor(list(target))[:,None]).flatten().tolist()
    delta=0.0
    for i in range(len(target)):
        truncated=model(torch.tensor([list(raw[:len(prefix)+i])],dtype=torch.long))[0,-1]
        delta=max(delta,float((truncated-logits[i]).abs().max()))
        if not torch.allclose(truncated,logits[i],atol=5e-5,rtol=1e-4) or int(truncated.argmax())!=predictions[i]:
            raise RuntimeError('causal prefix check failed')
    generated=[]
    for _ in range(32):
        # No reference name bytes are supplied in this separate decode.
        token=int(model(torch.tensor([list(prefix)+generated],dtype=torch.long))[0,-1].argmax())
        generated.append(token)
        if token in (0,10,40):break
    return {'predictions':predictions,'target_probabilities':probabilities,'scaffolded_bytes':generated,
            'metrics':metrics(target,predictions,probabilities,generated),
            'prefix_check':{'count':len(target),'max_logit_delta':delta,'consistent':True}}


def summarize(cases):
    return {name:{'cases':len(cases),**{key:sum(c['models'][name]['metrics'][key] for c in cases)
                  for key in cases[0]['models'][name]['metrics']}} for name in MODELS}


def verify(roots,dataset):
    initial=pinned_backend(roots['initial'])
    paths={};reports={}
    for policy in training.POLICIES:
        root=roots[policy];raw=(root/'report.json').read_bytes()
        if digest(raw)!=REPORT_HASHES[policy]:raise ValueError('pinned training report differs')
        report=json.loads(raw);training.audit(report,dataset)
        path=root/'experimental.nppai'
        if digest(path.read_bytes())!=report['model_sha256']:raise ValueError('checkpoint differs')
        paths[policy]=path;reports[policy]=report
    if (reports['fixed']['before']!=reports['resampled']['before'] or
            reports['fixed']['schedule']['order_sha256']!=reports['resampled']['schedule']['order_sha256']):
        raise ValueError('training pair differs')
    return initial,paths,reports


def cached(reports,group,index,name):
    if group not in reports['fixed']['before']:return None
    report=reports['fixed' if name=='initial' else name]
    return report['before' if name=='initial' else 'after'][group][index]


def audit(report,dataset,reports):
    groups=build(dataset)
    hashes={'initial':training.SOURCE_MODEL_ID[7:],**{p:r['model_sha256'] for p,r in reports.items()}}
    if (report['mode']!=MODE or report['training_run_id']!=TRAINING_RUN or report['source_reports']!=REPORT_HASHES or
            report['model_sha256']!=hashes or report['groups'].keys()!=groups.keys()):
        raise ValueError('diagnostic provenance/groups differ')
    for group,rows in groups.items():
        cases=report['groups'][group]
        if len(cases)!=len(rows):raise ValueError('case count differs')
        for index,(r,c) in enumerate(zip(rows,cases)):
            prefix,target=plan(r)
            if c['index']!=index or c['id']!=r['id'] or c['prompt']!=r['prompt'] or c['function']!=r['function']:
                raise ValueError('diagnostic record differs')
            if set(c['models'])!=set(MODELS):raise ValueError('model set differs')
            for name,item in c['models'].items():
                if item['metrics']!=metrics(target,item['predictions'],item['target_probabilities'],item['scaffolded_bytes']):
                    raise ValueError('raw byte score differs')
                check=item['prefix_check']
                if (check['count']!=len(target) or check['consistent'] is not True or
                        not math.isfinite(check['max_logit_delta']) or not 0<=check['max_logit_delta']<=0.01):
                    raise ValueError('causal check differs')
                if item['cached_free']!=cached(reports,group,index,name):raise ValueError('cached full output differs')
        if report['summary'][group]!=summarize(cases):raise ValueError('summary differs')
    return {'valid':True,'cases':sum(map(len,groups.values())),
            'limits':'Gold previous name bytes and def scaffold are assistance, not autonomous code generation.'}


def run(roots,dataset,output):
    import torch
    from scripts.check_dense_runtime_parity import load_weights,torch_model
    initial,paths,reports=verify(roots,dataset)
    torch.set_num_threads(2)
    models={'initial':torch_model(initial.weights),**{p:torch_model(load_weights(path)) for p,path in paths.items()}}
    for model in models.values():model.eval()
    groups={}
    with torch.no_grad():
        for group,rows in build(dataset).items():
            cases=[]
            for index,r in enumerate(rows):
                prefix,target=plan(r)
                case={k:r[k] for k in ('id','prompt','function')};case.update(index=index,models={})
                for name,model in models.items():
                    item=infer(model,prefix,target)
                    item['cached_free']=cached(reports,group,index,name)
                    case['models'][name]=item
                cases.append(case)
            groups[group]=cases
            print(json.dumps({'group':group,'summary':summarize(cases)}),flush=True)
    report={'training_run_id':TRAINING_RUN,'source_reports':REPORT_HASHES,
            'model_sha256':{'initial':training.SOURCE_MODEL_ID[7:],**{p:r['model_sha256'] for p,r in reports.items()}},
            'mode':MODE,
            'runtime':{'torch':torch.__version__,'threads':2,'context':256,'scaffolded_budget':32},
            'groups':groups,'summary':{g:summarize(c) for g,c in groups.items()}}
    evidence=audit(report,dataset,reports)
    output.mkdir(parents=True,exist_ok=True)
    raw=encoded(report);(output/'report.json').write_bytes(raw)
    (output/'audit.json').write_bytes(encoded({**evidence,'report_sha256':digest(raw)}))
    print(json.dumps({'audit':evidence,'report_sha256':digest(raw)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('initial','fixed','resampled','dataset','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run({p:getattr(args,p) for p in MODELS},args.dataset,args.output)
