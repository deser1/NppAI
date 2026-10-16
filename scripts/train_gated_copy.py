"""Paired experimental pointer/gate training and Docker-only functional probes."""
import argparse
import json
import math
from pathlib import Path
import platform
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import aligned_name_copy as aligned
from scripts import resampled_name_training as base
from scripts import diagnose_name_bytes as diagnostic
from scripts.run_local_model_smoke import pinned_backend
import sandbox_validate

ARMS=('control','copy')
HEAD_SEED=4242


def evaluate(model,groups,copy,with_bytes):
    from scripts.train_instruction_experiment import generate as plain_generate
    from scripts.gated_copy_model import generate,PromptAdapter
    results={}
    for group,rows in groups.items():
        cases=[]
        for index,r in enumerate(rows):
            output=generate(model,r['prompt']) if copy else plain_generate(model,r['prompt'],256)
            item={'index':index,'id':r['id'],'output':output,'metrics':base.score(r,output)}
            if with_bytes:
                adapter=PromptAdapter(model,r['prompt']) if copy else model
                item['name_bytes']=diagnostic.infer(adapter,*diagnostic.plan(r))
            cases.append(item)
        results[group]=cases
    return results


def historical_identity(report):
    old=aligned.control_report()
    for group,cases in old['before'].items():
        if report['arm']=='control' and [c['output'] for c in report['before'][group]]!=[c['output'] for c in cases]:
            raise ValueError('initial historical generations differ')
    if report['arm']=='control':
        for group,cases in old['after'].items():
            if [c['output'] for c in report['after'][group]]!=[c['output'] for c in cases]:
                raise ValueError('historical control generations differ')
        return report['checkpoint_sha256']==aligned.CONTROL_MODEL_SHA256
    return None


def audit(report,dataset):
    groups=aligned.development(dataset)
    expected={'arm','steps','head_seed','initial_model_id','schedule','development_sha256','optimizer',
              'checkpoint_sha256','before','after','summary','retention','loss_endpoints','runtime',
              'exported_parameters_equal','historical_checkpoint_equal','limits'}
    if set(report)!=expected or report['arm'] not in ARMS or report['steps']!=base.STEPS or report['head_seed']!=HEAD_SEED:
        raise ValueError('report schema or budget differs')
    if (report['initial_model_id']!=base.SOURCE_MODEL_ID or report['optimizer']!=base.OPTIMIZER or
            report['schedule']!=base.schedule_evidence(base.training_data(dataset),'resampled') or
            report['development_sha256']!=base.digest(base.encoded(groups)) or report['exported_parameters_equal'] is not True):
        raise ValueError('provenance or exposure differs')
    for stage in ('before','after'):
        if report[stage].keys()!=groups.keys():raise ValueError('group set differs')
        for group,rows in groups.items():
            cases=report[stage][group]
            if len(cases)!=len(rows):raise ValueError('case count differs')
            for index,(r,c) in enumerate(zip(rows,cases)):
                if c['index']!=index or c['id']!=r['id'] or c['metrics']!=base.score(r,c['output']):
                    raise ValueError('raw complete response score differs')
                if stage=='after':
                    item=c['name_bytes'];_,target=diagnostic.plan(r)
                    if item['metrics']!=diagnostic.metrics(target,item['predictions'],item['target_probabilities'],item['scaffolded_bytes']):
                        raise ValueError('raw assisted-byte score differs')
                    check=item['prefix_check']
                    if (check['count']!=len(target) or check['consistent'] is not True or
                            not math.isfinite(check['max_logit_delta']) or not 0<=check['max_logit_delta']<=0.01):
                        raise ValueError('causal diagnostic differs')
        if report['summary'][stage]!=aligned.summaries(report[stage]):raise ValueError('summary differs')
    if report['retention']!=base.retention(report['before'],report['after']):raise ValueError('retention differs')
    if report['historical_checkpoint_equal']!=historical_identity(report):raise ValueError('historical identity differs')
    if len(report['loss_endpoints'])!=2 or not all(math.isfinite(v) for v in report['loss_endpoints']):raise ValueError('nonfinite loss')
    return {'valid':True,'cases':sum(map(len,groups.values()))}


def run(source,dataset,output,arm):
    import torch
    import torch.nn.functional as F
    from train_nppai import export_to_bin
    from scripts.check_dense_runtime_parity import load_weights,torch_model
    from scripts.gated_copy_model import GatedCopyModel,source_mask,export_bundle,load_bundle
    backend=pinned_backend(source)
    if backend.model_id!=base.SOURCE_MODEL_ID:raise ValueError('pinned source differs')
    torch.manual_seed(42);torch.set_num_threads(2);torch.use_deterministic_algorithms(False)
    model=torch_model(backend.weights)
    copy=arm=='copy'
    if copy:
        torch.manual_seed(HEAD_SEED);model=GatedCopyModel(model)
    model.eval();groups=aligned.development(dataset);rows=base.training_data(dataset)
    with torch.no_grad():before=evaluate(model,groups,copy,False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001);losses=[];model.train()
    for step,selected in enumerate(base.schedule(rows,'resampled')):
        records=[r for _,r in selected];samples=[base.sample(r) for r in records]
        x=torch.nn.utils.rnn.pad_sequence([s[0] for s in samples],batch_first=True)
        y=torch.nn.utils.rnn.pad_sequence([s[1] for s in samples],batch_first=True,padding_value=-100)
        if copy:
            eligible=torch.stack([source_mask(x[i:i+1],r['prompt'])[0] for i,r in enumerate(records)])
            loss=F.nll_loss(model(x,eligible).reshape(-1,256),y.reshape(-1),ignore_index=-100)
        else:loss=F.cross_entropy(model(x).reshape(-1,256),y.reshape(-1),ignore_index=-100)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if step in (0,base.STEPS-1):losses.append(float(loss.detach()))
        if step%128==0 or step==base.STEPS-1:print(f'{arm} step={step+1} loss={float(loss.detach()):.6f}',flush=True)
    output.mkdir(parents=True,exist_ok=True);model.eval()
    checkpoint=output/('experimental.nppaicopy' if copy else 'control.nppai')
    if copy:
        export_bundle(model,checkpoint,base.SOURCE_MODEL_ID);loaded,manifest=load_bundle(checkpoint)
        if manifest['source_model_id']!=base.SOURCE_MODEL_ID:raise RuntimeError('bundle source differs')
    else:
        export_to_bin(model,str(checkpoint));loaded=torch_model(load_weights(checkpoint)).eval()
    if any(not torch.equal(a,b) for a,b in zip(model.parameters(),loaded.parameters())):raise RuntimeError('round-trip parameters differ')
    with torch.no_grad():after=evaluate(loaded,groups,copy,True)
    report={'arm':arm,'steps':base.STEPS,'head_seed':HEAD_SEED,'initial_model_id':base.SOURCE_MODEL_ID,
            'schedule':base.schedule_evidence(rows,'resampled'),'development_sha256':base.digest(base.encoded(groups)),
            'optimizer':base.OPTIMIZER,'checkpoint_sha256':base.digest(checkpoint.read_bytes()),
            'before':before,'after':after,'summary':{'before':aligned.summaries(before),'after':aligned.summaries(after)},
            'retention':base.retention(before,after),'loss_endpoints':losses,'exported_parameters_equal':True,
            'historical_checkpoint_equal':None,
            'runtime':{'torch':torch.__version__,'threads':2,'context':256,'new_tokens':160,
                       'platform':platform.platform(),'processor':platform.processor(),'python':platform.python_version()},
            'limits':'Experimental Python-only gated readout. One seed and exposed development views. '
                     'Full responses are separate from assisted names; no native compatibility or automatic promotion.'}
    report['historical_checkpoint_equal']=historical_identity(report)
    evidence=audit(report,dataset);raw=base.encoded(report)
    (output/'report.json').write_bytes(raw)
    print(json.dumps({'arm':arm,'summary':report['summary']['after'],'audit':evidence}),flush=True)
    return report,loaded


def functional_probes(models,tasks_path,image):
    from scripts.gated_copy_model import generate
    from scripts.train_instruction_experiment import generate as plain_generate
    raw=tasks_path.read_bytes();tasks=json.loads(raw);prepared=[];seen=set()
    for task in tasks:
        if set(task)!={'task_id','prompt','cases_file'} or task['task_id'] in seen:raise ValueError('invalid independent task')
        seen.add(task['task_id']);cases,suite=sandbox_validate.load_cases(Path(task['cases_file']))
        prepared.append((task,cases,suite))
    if not prepared:raise ValueError('independent tasks required')
    results={}
    for arm,model in models.items():
        items=[]
        for task,cases,suite in prepared:
            source=generate(model,task['prompt']) if arm=='copy' else plain_generate(model,task['prompt'],256)
            category='empty';detail='Empty/whitespace output'
            if source.strip():
                status,detail=sandbox_validate.validate(source,'syntax',cases,image);category='syntax_failure'
                if status==0:
                    status,detail=sandbox_validate.validate(source,'tests',cases,image)
                    category='passed' if status==0 else 'functional_failure'
                if status not in (0,1):raise RuntimeError('validation infrastructure: '+detail[:512])
            items.append({**task,'source':source,'suite_sha256':suite,'category':category,'diagnostics':detail[:4096]})
        results[arm]={'tasks':len(items),'passed':sum(c['category']=='passed' for c in items),'cases':items}
    return {'manifest_sha256':base.digest(raw),'image':image,'models':results,
            'limits':'Six separately authored, optimization-independent probes, correlated with simple-function families; not broad independent task accuracy.'}


def run_pair(source,dataset,output,tasks,image):
    control,control_model=run(source,dataset,output/'control','control')
    repeat,_=run(source,dataset,output/'control-repeat','control')
    if base.encoded(control)!=base.encoded(repeat):raise RuntimeError('same-host control repeat differs')
    copy,copy_model=run(source,dataset,output/'copy','copy')
    if control['schedule']!=copy['schedule'] or control['runtime']!=copy['runtime']:raise RuntimeError('paired exposure/runtime differs')
    pair={'control_repeat_identical':True,'same_host':True,'schedule_identical':True,
          'historical_control_checkpoint_equal':control['historical_checkpoint_equal'],
          'report_sha256':{arm:base.digest(base.encoded(r)) for arm,r in [('control',control),('copy',copy)]}}
    probes=functional_probes({'control':control_model,'copy':copy_model},tasks,image)
    (output/'pair-audit.json').write_bytes(base.encoded(pair));(output/'functional.json').write_bytes(base.encoded(probes))
    print(json.dumps({'paired_audit':pair,'functional':{arm:{k:r[k] for k in ('tasks','passed')} for arm,r in probes['models'].items()}}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','dataset','output','tasks'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--image',required=True)
    args=parser.parse_args();run_pair(args.source,args.dataset,args.output,args.tasks,args.image)
