"""Matched response-loss versus body-only-loss continuation, with identical data/draws."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.body_argument_aliases import (build, verify_body, evaluate, paired, audit as audit_body,
    OPTIMIZER, STEPS, SOURCE_RUN, digest, encoded, ARMS)
from scripts.header_original_replay import draw_batch
from scripts.evaluate_staged_generation import valid_header

POLICIES = ('full_response', 'body_only')
LIMITS = ('Matched6000-update continuation from the same20k body, varied argument pools and50% original examples. '
          'Same inputs, examples, order, optimizer and budget; only supervised header targets differ. '
          'Body-only masking retains teacher-forced headers as input/context and trains all weights. '
          'Cross-entropy averages active targets, so masking changes loss normalization and gradient allocation too. '
          'Assisted signatures are prompt-provided, not learned copying. One seed and1100 correlated exposed development views. '
          'No generated-source execution, fresh-probe use, canonical restoration, native conversion or promotion. '
          'AST/exact are structural reference metrics, not semantic correctness. Fixed seeds do not guarantee identical retraining.')


def boundaries(record):
    answer = record['answer']
    header = answer.splitlines(keepends=True)[0] if answer else ''
    if not valid_header(header) or not header.startswith('def '+record['function']+'(') or len(answer) <= len(header):
        raise ValueError('requires full reference with LF header and body')
    start = len(('[USER]: '+record['prompt']+'\n[AI]:\n').encode()) - 1
    return start, len(header.encode()), len(answer.encode()) - len(header.encode()) + 1


def mask_spec(record, policy):
    if policy not in POLICIES:
        raise ValueError('unknown loss policy')
    start, header, body = boundaries(record)
    return {'prompt_targets_ignored':start, 'header_targets':header if policy=='full_response' else 0,
            'header_targets_ignored':header if policy=='body_only' else 0, 'body_and_eos_targets':body}


def sample(record, context, policy):
    from scripts.train_broader_instruction_experiment import sample as full_sample
    spec = mask_spec(record, policy)
    x,y = full_sample(record,context)
    if policy=='body_only':
        start=spec['prompt_targets_ignored']
        y[start:start+spec['header_targets_ignored']]=-100
    return x,y


def empty_counts():
    return {pool:{'header':0,'body_and_eos':0} for pool in ('original','renamed')}


def scheduled_evidence(original, renamed, policy):
    rng=random.Random(42);hasher=hashlib.sha256();counts=empty_counts()
    specs={r['id']:mask_spec(r,policy) for r in original+renamed}
    for _ in range(STEPS):
        selected=draw_batch(original,renamed,'half_original',rng)
        hasher.update(encoded([[pool,r['id']] for pool,r in selected]))
        for pool,r in selected:
            spec=specs[r['id']]
            counts[pool]['header']+=spec['header_targets']
            counts[pool]['body_and_eos']+=spec['body_and_eos_targets']
    return {'sha256':hasher.hexdigest(),'supervised_targets':counts,
            'draws':{'original':STEPS*2,'renamed':STEPS*2}}


def audit(report,dataset):
    evidence=audit_body(report,dataset)
    original,renamed,_,_=build(dataset,'varied_arguments')
    policy=report['loss_policy']
    expected=scheduled_evidence(original,renamed,policy)
    if (report['argument_policy']!='varied_arguments' or report['training_schedule']!=expected or
        report['observed_training_draws']!=expected['draws'] or report['masking']!={
            'policy':policy,'prompt':'ignored','header':'ignored' if policy=='body_only' else 'supervised',
            'body_and_eos':'supervised','header_context':'retained teacher forcing',
            'reduction':'mean over active response targets'}):
        raise ValueError('loss mask, sample order or actual supervised target counts differ')
    return {**evidence,'loss_policy':policy,'training_schedule':expected,'limits':LIMITS}


def train(original,renamed,development,initial_model,checkpoint,policy):
    import torch
    import torch.nn.functional as F
    from train_nppai import NppAIModel,export_to_bin
    from scripts.train_identifier_headers import initialize_header_weights
    from scripts.train_instruction_experiment import generate
    from scripts.check_dense_runtime_parity import load_weights,torch_model
    torch.manual_seed(42);torch.set_num_threads(2);rng=random.Random(42)
    model=NppAIModel(256,64,128,2,256)
    initialization=initialize_header_weights(model,initial_model)
    model.eval()
    with torch.no_grad():before=evaluate(model,development,generate)
    samples={};target_counts={}
    for record in original+renamed:
        x,y=sample(record,256,policy);samples[record['id']]=(x,y)
        start,header,_=boundaries(record)
        target_counts[record['id']]={'header':int((y[start:start+header]!=-100).sum()),
                                    'body_and_eos':int((y[start+header:]!=-100).sum())}
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    counts=empty_counts();draws={'original':0,'renamed':0};hasher=hashlib.sha256()
    model.train()
    for step in range(STEPS):
        selected=draw_batch(original,renamed,'half_original',rng)
        hasher.update(encoded([[pool,r['id']] for pool,r in selected]))
        for pool,record in selected:
            draws[pool]+=1
            for key,value in target_counts[record['id']].items():counts[pool][key]+=value
        batch=[samples[r['id']] for _,r in selected]
        x=torch.nn.utils.rnn.pad_sequence([b[0] for b in batch],batch_first=True)
        y=torch.nn.utils.rnn.pad_sequence([b[1] for b in batch],batch_first=True,padding_value=-100)
        loss=F.cross_entropy(model(x).reshape(-1,256),y.reshape(-1),ignore_index=-100)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if step%100==0 or step==STEPS-1:print(f'step={step+1} loss={float(loss.detach()):.6f}',flush=True)
    model.eval();checkpoint.parent.mkdir(parents=True,exist_ok=True);export_to_bin(model,str(checkpoint))
    reloaded=torch_model(load_weights(checkpoint))
    with torch.no_grad():
        after=evaluate(reloaded,development,generate)
        for split,rows in development.items():
            for record,cached in zip(rows,after[split]['cases']):
                for arm in ARMS:
                    prefix=cached['assisted_header'] if arm=='task_signature_assisted' else ''
                    if cached[arm]['output']!=generate(model,record['prompt'],256,assistant_prefix=prefix):
                        raise RuntimeError('export/reload changed complete body outputs')
    return {'initialization':initialization,'observed_training_draws':draws,
            'training_schedule':{'sha256':hasher.hexdigest(),'supervised_targets':counts,'draws':draws},
            'runtime':{'python':sys.version.split()[0],'torch':torch.__version__,'threads':2},
            'reload_outputs_identical':True,'evaluation':{'before':before,'after':after}}


def run(source,initial,dataset,output,policy):
    original,renamed,development,metadata=build(dataset,'varied_arguments')
    initial_model,provenance=verify_body(source,initial,dataset)
    destination=Path(output);checkpoint=destination/'NppAI-body-loss-mask.nppai'
    measured=train(original,renamed,development,initial_model,checkpoint,policy)
    report={**measured,'source_run_id':SOURCE_RUN,'source':provenance,'argument_policy':'varied_arguments',
            'dataset':metadata,'loss_policy':policy,'additional_steps':STEPS,'total_body_updates':26000,
            'optimizer':OPTIMIZER,'context':256,'new_tokens_per_call':160,
            'model_sha256':digest(checkpoint.read_bytes()),'limits':LIMITS,
            'masking':{'policy':policy,'prompt':'ignored','header':'ignored' if policy=='body_only' else 'supervised',
                'body_and_eos':'supervised','header_context':'retained teacher forcing','reduction':'mean over active response targets'}}
    report['paired_before_after']=paired(report['evaluation'])
    evidence=audit(report,dataset);raw=encoded(report)
    (destination/'loss-report.json').write_bytes(raw);evidence['source_report_sha256']=digest(raw)
    (destination/'loss-audit.json').write_bytes(encoded(evidence))
    for name,value in (('original-training',original),('renamed-training',renamed),('development',development),('manifest',metadata)):
        (destination/(name+'.json')).write_bytes(encoded(value))
    for folder in ('body-source','header-source','initial-header-source','initial-body-source'):
        shutil.copytree(Path(source)/folder,destination/folder,dirs_exist_ok=True)
    shutil.copy2(Path(source)/'staged-report.json',destination/'source-continuation-report.json')
    print(json.dumps({'loss_policy':policy,'audit':evidence,
        'summary':{stage:{split:g['summary'] for split,g in groups.items()} for stage,groups in report['evaluation'].items()},
        'paired_before_after':report['paired_before_after']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','initial','dataset','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--policy',choices=POLICIES,required=True)
    args=parser.parse_args();run(args.source,args.initial,args.dataset,args.output,args.policy)
