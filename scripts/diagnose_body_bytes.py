"""Frozen body-byte teacher forcing; gold prefixes are diagnostic assistance, not generation."""
import argparse
import ast
import io
import json
import keyword
import math
from pathlib import Path
import shutil
import sys
import tokenize

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.body_loss_masking import audit as audit_loss, build, digest, encoded, boundaries
from scripts.diagnose_header_replay import score

SOURCE_RUN=38702282716
HASHES={'full_response':'bdac050d83b2efd9b3ad1bd12d6b991142120d4e0e2ffdc2994fb78b4bba4296',
        'body_only':'4a9becc992979d49a4e1918e5ad9678a7857943dc50af3821f773084c473a645'}
MODELS=('source20k','full_response','body_only')
CATEGORIES=('argument_first_byte','argument_suffix_byte','operation_name','operator_punctuation','other_body','eos')
LIMITS=('Inference-only teacher-forcing diagnostic on1100 correlated exposed development views. '
        'Every previous reference byte is supplied, including operators/callees and prior argument bytes. '
        'Top1 accuracy/target probabilities do not measure autonomous copying, operation choice or semantic correctness. '
        'Reference token categories are descriptive; non-argument NAME and OP tokens are not functional correctness metrics. '
        'Cached free/assisted outputs are copied from pinned reports, not newly generated. '
        'No training, output repair, canonicalization, generated-source execution, fresh-probe use or promotion. '
        'Results do not establish a unique causal mechanism.')


def plan(record,context=256):
    start,header,_=boundaries(record)
    answer=record['answer'];raw=answer.encode();prefix=('[USER]: '+record['prompt']+'\n[AI]:\n').encode()
    sequence=prefix+raw+b'\0'
    if b'\0' in prefix+raw or len(sequence)>context:
        raise ValueError('NUL or full reference context overflow')
    tree=ast.parse(answer)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError('requires single reference function')
    fn=tree.body[0];args=fn.args
    if args.posonlyargs or args.defaults or args.vararg or args.kwonlyargs or args.kwarg:
        raise ValueError('requires plain positional reference arguments')
    names=[a.arg for a in args.args]
    lines=answer.splitlines(keepends=True);offsets=[];offset=0
    for line in lines:offsets.append(offset);offset+=len(line.encode())
    def token_offset(position):
        row,col=position
        return offsets[row-1]+len(lines[row-1][:col].encode())
    refs={(offsets[n.lineno-1]+n.col_offset,offsets[n.end_lineno-1]+n.end_col_offset):n.id
          for n in ast.walk(fn) if isinstance(n,ast.Name) and n.id in names and n.lineno>1}
    expected=list(raw[header:])+[0];categories=['other_body']*(len(expected)-1)+['eos'];spans=[]
    for token in tokenize.generate_tokens(io.StringIO(answer).readline):
        if token.start[0]<=1 or token.type not in (tokenize.NAME,tokenize.OP):continue
        left,right=token_offset(token.start),token_offset(token.end)
        a,b=left-header,right-header
        if (left,right) in refs:
            name=refs[(left,right)]
            if bytes(expected[a:b])!=name.encode():raise ValueError('argument token byte span differs')
            spans.append({'name':name,'argument_index':names.index(name),'start':a,'end':b})
            categories[a:b]=['argument_first_byte']+['argument_suffix_byte']*(b-a-1)
        elif token.type==tokenize.NAME and not keyword.iskeyword(token.string):categories[a:b]=['operation_name']*(b-a)
        elif token.type==tokenize.OP:categories[a:b]=['operator_punctuation']*(b-a)
    return {'sequence':sequence,'start':start+header,'expected':expected,'categories':categories,'argument_spans':spans}


def visible_plan(p):
    return {'body_target_start':p['start'],'categories':p['categories'],'argument_spans':p['argument_spans']}


def metrics(p,predictions,probabilities):
    expected=p['expected']
    if len(predictions)!=len(expected) or len(probabilities)!=len(expected):raise ValueError('body prediction length differs')
    if any(type(v)!=int or not 0<=v<256 for v in predictions):raise ValueError('invalid predicted byte')
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in probabilities):
        raise ValueError('invalid target probability')
    correct=[a==b for a,b in zip(expected,predictions)]
    by_category={}
    for cat in CATEGORIES:
        indices=[i for i,c in enumerate(p['categories']) if c==cat]
        by_category[cat]={'total_bytes':len(indices),'correct_bytes':sum(correct[i] for i in indices),
                          'target_probability_sum':sum(probabilities[i] for i in indices),
                          'nll_sum':sum(-math.log(max(probabilities[i],1e-30)) for i in indices)}
    spans=p['argument_spans'];suffix=[s for s in spans if s['end']-s['start']>1]
    occurrences={'total':len(spans),'all_bytes_correct':sum(all(correct[s['start']:s['end']]) for s in spans),
                 'initial_byte_correct':sum(correct[s['start']] for s in spans),
                 'with_suffix':len(suffix),'suffix_all_correct':sum(all(correct[s['start']+1:s['end']]) for s in suffix)}
    return {'by_category':by_category,'argument_occurrences':occurrences,
            'cases_with_arguments':int(bool(spans)),
            'cases_all_argument_bytes_correct':int(bool(spans) and all(all(correct[s['start']:s['end']]) for s in spans))}


def infer(model,p):
    import torch
    sequence=p['sequence'];x=torch.tensor([list(sequence[:-1])],dtype=torch.long)
    full=model(x)[0];logits=full[p['start']:]
    if len(logits)!=len(p['expected']):raise ValueError('shifted logits do not cover body/EOS')
    targets=torch.tensor(p['expected'],dtype=torch.long)
    probabilities=logits.softmax(dim=-1).gather(1,targets[:,None]).flatten().tolist()
    predictions=logits.argmax(dim=-1).tolist();max_delta=0.0
    for span in p['argument_spans']:
        index=p['start']+span['start']
        # This truncated input contains only actual preceding gold bytes.
        prefix=model(x[:,:index+1])[0,-1]
        max_delta=max(max_delta,float((prefix-full[index]).abs().max()))
        if not torch.allclose(prefix,full[index],atol=5e-5,rtol=1e-4) or int(prefix.argmax())!=predictions[span['start']]:
            raise RuntimeError('reference prefill/truncated-prefix causal logits differ')
    return {'predictions':predictions,'target_probabilities':probabilities,
            'metrics':metrics(p,predictions,probabilities),
            'prefix_check':{'count':len(p['argument_spans']),'consistent':True,'max_logit_delta':max_delta}}


def verify_sources(roots,dataset):
    reports={};models={}
    for policy in HASHES:
        root=Path(roots[policy]);raw=(root/'loss-report.json').read_bytes()
        if digest(raw)!=HASHES[policy]:raise ValueError('pinned loss report differs')
        r=json.loads(raw);audit_loss(r,dataset)
        if r['loss_policy']!=policy:raise ValueError('loss policy source differs')
        model=root/'NppAI-body-loss-mask.nppai'
        if digest(model.read_bytes())!=r['model_sha256']:raise ValueError('loss model hash differs')
        for path,key in [('body-source/NppAI-broader-instruction.nppai','model_sha256'),
                         ('body-source/broader-report.json','report_sha256'),
                         ('initial-body-source/NppAI-broader-instruction.nppai','initial_body_sha256')]:
            if digest((root/path).read_bytes())!=r['source'][key]:raise ValueError('preserved body source hash differs')
        reports[policy]=r;models[policy]=model
    a,b=reports.values()
    if (a['source']!=b['source'] or a['dataset']!=b['dataset'] or a['evaluation']['before']!=b['evaluation']['before'] or
        a['training_schedule']['sha256']!=b['training_schedule']['sha256']):raise ValueError('matched loss sources differ')
    models['source20k']=Path(roots['full_response'])/'body-source/NppAI-broader-instruction.nppai'
    return reports,models


def caches(sources,split):
    return {name:{c['id']:c for c in sources['full_response' if name=='source20k' else name]['evaluation'][
            'before' if name=='source20k' else 'after'][split]['cases']} for name in MODELS}


def summarize(cases):
    result={'cases':len(cases)}
    for name in MODELS:
        items=[c['models'][name] for c in cases]
        result[name]={'by_category':{cat:{key:sum(i['metrics']['by_category'][cat][key] for i in items)
                       for key in ('total_bytes','correct_bytes','target_probability_sum','nll_sum')} for cat in CATEGORIES},
                     'argument_occurrences':{key:sum(i['metrics']['argument_occurrences'][key] for i in items)
                       for key in ('total','all_bytes_correct','initial_byte_correct','with_suffix','suffix_all_correct')},
                     'cases_with_arguments':sum(i['metrics']['cases_with_arguments'] for i in items),
                     'cases_all_argument_bytes_correct':sum(i['metrics']['cases_all_argument_bytes_correct'] for i in items),
                     'cached_assisted_exact':sum(i['cached_assisted']['exact_response'] for i in items),
                     'cached_free_exact':sum(i['cached_free']['exact_response'] for i in items),
                     'prefix_checks':sum(i['prefix_check']['count'] for i in items)}
    return result


def audit(report,dataset,sources):
    development=build(dataset,'varied_arguments')[2]
    model_hashes={'source20k':sources['full_response']['source']['model_sha256'],
                  **{p:sources[p]['model_sha256'] for p in HASHES}}
    if (report['source_run_id']!=SOURCE_RUN or report['source_report_sha256']!=HASHES or
        report['model_sha256']!=model_hashes or report['mode']!='teacher_forced_reference_prefixes' or
        report['groups'].keys()!=development.keys()):
        raise ValueError('byte diagnostic source/groups differ')
    for split,rows in development.items():
        cases=report['groups'][split];expected={r['id']:r for r in rows};cache=caches(sources,split)
        if len(cases)!=len(rows) or len({c['id'] for c in cases})!=len(rows):raise ValueError('missing/duplicate byte cases')
        for case in cases:
            r=expected.get(case['id'])
            if r is None or any(case.get(k)!=v for k,v in r.items()):raise ValueError('byte task/reference differs')
            p=plan(r)
            if case['layout']!=visible_plan(p) or case['models'].keys()!=set(MODELS):raise ValueError('byte spans/models differ')
            for name in MODELS:
                item=case['models'][name];old=cache[name][r['id']]
                if item['metrics']!=metrics(p,item['predictions'],item['target_probabilities']):raise ValueError('teacher-forced raw metrics differ')
                check=item['prefix_check']
                if (check['count']!=len(p['argument_spans']) or check['consistent'] is not True or
                    not math.isfinite(check['max_logit_delta']) or not 0<=check['max_logit_delta']<=0.01):raise ValueError('causal prefix check differs')
                if item['cached_assisted']!=old['task_signature_assisted'] or item['cached_free']!=old['single_model']:
                    raise ValueError('cached generation copied incorrectly')
        if report['summary'][split]!=summarize(cases):raise ValueError('byte summary differs')
    return {'valid':True,'cases':sum(map(len,development.values())),'limits':LIMITS}


def run(roots,dataset,output):
    sources,paths=verify_sources(roots,dataset);development=build(dataset,'varied_arguments')[2]
    import torch
    from scripts.check_dense_runtime_parity import load_weights,torch_model
    from scripts.identifier_curriculum import validate_initial_weights
    torch.set_num_threads(2);models={}
    for name in MODELS:
        weights=load_weights(paths[name]);validate_initial_weights(weights);models[name]=torch_model(weights)
    groups={}
    with torch.no_grad():
        for split,rows in development.items():
            cache=caches(sources,split);cases=[]
            for r in rows:
                p=plan(r);case={**r,'layout':visible_plan(p),'models':{}}
                for name in MODELS:
                    item=infer(models[name],p);old=cache[name][r['id']]
                    item.update(cached_assisted=old['task_signature_assisted'],cached_free=old['single_model'])
                    case['models'][name]=item
                cases.append(case)
            groups[split]=cases
            print(json.dumps({'split':split,**summarize(cases)}),flush=True)
    report={'source_run_id':SOURCE_RUN,'source_report_sha256':HASHES,
            'model_sha256':{n:digest(p.read_bytes()) for n,p in paths.items()},
            'runtime':{'torch':torch.__version__,'python':sys.version.split()[0],'threads':2},
            'mode':'teacher_forced_reference_prefixes','limits':LIMITS,
            'groups':groups,'summary':{s:summarize(c) for s,c in groups.items()}}
    evidence=audit(report,dataset,sources);destination=Path(output);destination.mkdir(parents=True,exist_ok=True)
    raw=encoded(report);(destination/'body-byte-report.json').write_bytes(raw);evidence['source_report_sha256']=digest(raw)
    (destination/'body-byte-audit.json').write_bytes(encoded(evidence))
    for name,root in roots.items():shutil.copytree(root,destination/name,dirs_exist_ok=True)
    print(json.dumps({'audit':evidence,'summary':report['summary']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('full-response','body-only','dataset','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();run({'full_response':args.full_response,'body_only':args.body_only},args.dataset,args.output)
