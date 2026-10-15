"""Matched full-response continuation with optional source-position attention loss."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import resampled_name_training as base
from scripts import diagnose_name_bytes as diagnostic
from scripts.run_local_model_smoke import pinned_backend

STEPS=1024
COEFFICIENT=0.1
ARMS=('control','aligned')
CONTROL_REPORT_SHA256=diagnostic.REPORT_HASHES['resampled']
CONTROL_MODEL_SHA256='fc059649eacf8840536fecd1619451bac702eb7ce858d94908527aed2344780a'


def alignment(record):
    prefix,raw=base.frame(record)
    name=record['function'];target=name.encode()
    matches=list(re.finditer(r'(?<!\w)'+re.escape(name)+r'(?=\s*\()',record['prompt']))
    if len(matches)!=1 or not record['answer'].startswith('def '+name+'('):
        raise ValueError('requires one explicit matching function signature')
    source=len(b'[USER]: ')+len(record['prompt'][:matches[0].start()].encode())
    query=len(prefix)-1+len(b'def ')
    pairs=[(query+i,source+i) for i in range(len(target))]
    if any(s>=q or raw[s]!=raw[q+1] for q,s in pairs):
        raise ValueError('source/target byte alignment differs')
    return pairs


@contextmanager
def capture_qk(model):
    """Training-only observation; forward API, architecture and export stay intact."""
    captured={};handles=[]
    try:
        for name in ('wq','wk'):
            handles.append(getattr(model.layers[-1],name).register_forward_hook(
                lambda module,args,output,key=name:captured.__setitem__(key,output)))
        yield captured
    finally:
        for handle in handles:handle.remove()


def attention_loss(captured,pairs,lengths):
    import torch
    import torch.nn.functional as F
    q,k=captured['wq'],captured['wk']
    scores=q@k.transpose(-2,-1)/(q.shape[-1]**0.5)
    mask=torch.tril(torch.ones(scores.shape[-2:],device=scores.device,dtype=torch.bool))
    scores=scores.masked_fill(~mask,float('-inf'))
    selected=[];labels=[]
    for batch,(mapping,length) in enumerate(zip(pairs,lengths)):
        for query,source in mapping:
            if not 0<=source<query<length:raise ValueError('invalid causal alignment index')
            selected.append(scores[batch,query]);labels.append(source)
    if not selected:raise ValueError('no aligned name bytes')
    return F.cross_entropy(torch.stack(selected),torch.tensor(labels,device=q.device))


def training_losses(model,x,y,records,lengths,aligned):
    import torch.nn.functional as F
    if aligned:
        with capture_qk(model) as captured:logits=model(x)
        auxiliary=attention_loss(captured,[alignment(r) for r in records],lengths)
    else:
        # Keep the control's forward/backward graph identical to the old recipe.
        logits=model(x)
        auxiliary=None
    ce=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1),ignore_index=-100)
    return ce,auxiliary


def development(dataset):
    groups=base.development_data(dataset)
    for split in ('validation','test'):
        groups[split+'/matched_prefix']=[base.rename(r,'nc_'+r['function'][3:]) for r in groups[split+'/renamed']]
    original=[];renamed=[];seen_names=set()
    for batch in base.schedule(base.training_data(dataset),'resampled'):
        for pool,r in batch:
            if pool=='renamed':
                seen_names.add(r['function'])
                if len(renamed)<16:renamed.append(r)
            elif len(original)<16:original.append(r)
    for split in ('validation','test'):
        if seen_names & {r['function'] for r in groups[split+'/matched_prefix']}:
            raise ValueError('development alias seen in training')
    groups.update({'train/original':original,'train/renamed':renamed})
    return groups


def evaluate(model,groups,with_bytes=False):
    from scripts.train_instruction_experiment import generate
    result={}
    for group,rows in groups.items():
        cases=[]
        for index,r in enumerate(rows):
            output=generate(model,r['prompt'],256)
            item={'index':index,'id':r['id'],'output':output,'metrics':base.score(r,output)}
            if with_bytes:item['name_bytes']=diagnostic.infer(model,*diagnostic.plan(r))
            cases.append(item)
        result[group]=cases
    return result


def summaries(groups):
    result={}
    for group,cases in groups.items():
        item=base.summary(cases)
        if 'name_bytes' in cases[0]:
            item['name_bytes']={k:sum(c['name_bytes']['metrics'][k] for c in cases) for k in
                                ('name_bytes','correct_bytes','all_name_bytes_correct','scaffolded_exact_name')}
        result[group]=item
    return result


def control_report():
    path=Path(__file__).resolve().parents[1]/'docs/evidence/resampled-name-training/resampled.json'
    raw=path.read_bytes()
    if base.digest(raw)!=CONTROL_REPORT_SHA256:raise ValueError('pinned control report differs')
    return json.loads(raw)


def check_reproduction(report,old):
    for stage in ('before','after'):
        if stage=='after' and report['arm']!='control':continue
        for group,cases in old[stage].items():
            if [c['output'] for c in report[stage][group]]!=[c['output'] for c in cases]:
                raise ValueError('cached control generations differ')
    if report['arm']=='control' and report['model_sha256']!=CONTROL_MODEL_SHA256:
        raise ValueError('control checkpoint does not reproduce')


def audit(report,dataset):
    groups=development(dataset);rows=base.training_data(dataset)
    if (report['arm'] not in ARMS or report['steps']!=STEPS or report['coefficient']!=(0.0 if report['arm']=='control' else COEFFICIENT)
            or report['source_run_id']!=base.SOURCE_RUN or report['source_report_sha256']!=base.REPORT_SHA256
            or report['initial_model_id']!=base.SOURCE_MODEL_ID or report['optimizer']!=base.OPTIMIZER
            or report['schedule']!=base.schedule_evidence(rows,'resampled')
            or report['development_sha256']!=base.digest(base.encoded(groups)) or not report['exported_weights_equal']):
        raise ValueError('experiment metadata differs')
    for stage in ('before','after'):
        if report[stage].keys()!=groups.keys():raise ValueError('evaluation groups differ')
        for group,refs in groups.items():
            cases=report[stage][group]
            if len(cases)!=len(refs):raise ValueError('case count differs')
            for index,(r,c) in enumerate(zip(refs,cases)):
                if c['index']!=index or c['id']!=r['id'] or c['metrics']!=base.score(r,c['output']):
                    raise ValueError('raw complete response differs')
                if stage=='after':
                    item=c['name_bytes'];_,target=diagnostic.plan(r)
                    if item['metrics']!=diagnostic.metrics(target,item['predictions'],item['target_probabilities'],item['scaffolded_bytes']):
                        raise ValueError('raw byte metrics differ')
                    check=item['prefix_check']
                    if (check['count']!=len(target) or check['consistent'] is not True or
                            not math.isfinite(check['max_logit_delta']) or not 0<=check['max_logit_delta']<=0.01):
                        raise ValueError('prefix check differs')
        if report['summary'][stage]!=summaries(report[stage]):raise ValueError('summary differs')
    if report['retention']!=base.retention(report['before'],report['after']):raise ValueError('retention differs')
    endpoint_keys={'response_ce'} if report['arm']=='control' else {'response_ce','alignment_ce'}
    if (len(report['loss_endpoints'])!=2 or any(set(point)!=endpoint_keys for point in report['loss_endpoints']) or
            any(not math.isfinite(v) for point in report['loss_endpoints'] for v in point.values())):
        raise ValueError('invalid loss endpoints')
    check_reproduction(report,control_report())
    return {'valid':True,'cases':sum(map(len,groups.values())),
            'limits':'Attention alignment and name assistance do not establish complete-code or functional competence.'}


def run(source,dataset,output,arm):
    import torch
    import torch.nn.functional as F
    from train_nppai import export_to_bin
    from scripts.check_dense_runtime_parity import load_weights,torch_model
    backend=pinned_backend(source)
    if backend.model_id!=base.SOURCE_MODEL_ID:raise ValueError('initial model differs')
    torch.manual_seed(42);torch.set_num_threads(2)
    model=torch_model(backend.weights);model.eval()
    rows=base.training_data(dataset);groups=development(dataset)
    with torch.no_grad():before=evaluate(model,groups)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    endpoints=[];coefficient=0.0 if arm=='control' else COEFFICIENT
    model.train()
    for step,selected in enumerate(base.schedule(rows,'resampled')):
        records=[r for _,r in selected];samples=[base.sample(r) for r in records]
        x=torch.nn.utils.rnn.pad_sequence([s[0] for s in samples],batch_first=True)
        y=torch.nn.utils.rnn.pad_sequence([s[1] for s in samples],batch_first=True,padding_value=-100)
        ce,auxiliary=training_losses(model,x,y,records,[len(s[0]) for s in samples],arm=='aligned')
        loss=ce if arm=='control' else ce+coefficient*auxiliary
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if step==0 or step==STEPS-1:
            point={'response_ce':float(ce.detach())}
            if auxiliary is not None:point['alignment_ce']=float(auxiliary.detach())
            endpoints.append(point)
        if step%128==0 or step==STEPS-1:print(f'{arm} step={step+1} response={float(ce.detach()):.6f}',flush=True)
    output.mkdir(parents=True,exist_ok=True);checkpoint=output/'experimental.nppai'
    model.eval();export_to_bin(model,str(checkpoint));loaded=torch_model(load_weights(checkpoint));loaded.eval()
    if any(not torch.equal(a,b) for a,b in zip(model.parameters(),loaded.parameters())):raise RuntimeError('exported tensors differ')
    with torch.no_grad():after=evaluate(loaded,groups,with_bytes=True)
    report={'arm':arm,'steps':STEPS,'coefficient':coefficient,'initial_model_id':backend.model_id,
            'source_run_id':base.SOURCE_RUN,'source_report_sha256':base.REPORT_SHA256,
            'model_sha256':base.digest(checkpoint.read_bytes()),'optimizer':base.OPTIMIZER,
            'schedule':base.schedule_evidence(rows,'resampled'),'development_sha256':base.digest(base.encoded(groups)),
            'before':before,'after':after,'summary':{'before':summaries(before),'after':summaries(after)},
            'retention':base.retention(before,after),'loss_endpoints':endpoints,'exported_weights_equal':True,
            'runtime':{'torch':torch.__version__,'threads':2,'context':256,'new_tokens':160},
            'limits':'One seed; exposed development data. Same native architecture and export, no new inference head. '
                     'Last-layer attention supervision is a training objective, not a guaranteed copying mechanism. '
                     'No generated code execution, smoke/fresh-probe access or automatic checkpoint promotion.'}
    evidence=audit(report,dataset)
    raw=base.encoded(report);(output/'report.json').write_bytes(raw)
    (output/'audit.json').write_bytes(base.encoded({**evidence,'report_sha256':base.digest(raw)}))
    print(json.dumps({'arm':arm,'summary':report['summary']['after'],'loss_endpoints':endpoints,
                      'model_sha256':report['model_sha256'],'audit':evidence}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','dataset','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    args=parser.parse_args();run(args.source,args.dataset,args.output,args.arm)
