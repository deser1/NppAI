"""Frozen prompt-only argument normalization diagnostic; never execute generated code."""
import argparse
import ast
import builtins
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tokenize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.body_argument_aliases import audit as audit_body, build, digest, encoded
from scripts.evaluate_task_signature_wording import task_header
from scripts.diagnose_header_replay import score, METRICS

SOURCE_RUN = 38697039023
REPORT_SHA = 'e11680be1dc9da3ff056339cbe893ee544d897f8061d4c162dae065e05108a28'
POLICIES = ('x', 'values', 'text')
ARMS = ('direct_assisted', *POLICIES)
LIMITS = ('Inference-only assisted diagnostic on1100 correlated exposed development views. '
          'Signature parsing, prompt normalization and restoration are deterministic assistance, not learned copying. '
          'Each policy is fixed globally; no per-case best/oracle selection. Two-argument policies share the same a,b output; unsupported arities pass through unchanged. '
          'No training, generated-source execution, reference repair, fresh-probe use, native conversion or promotion. '
          'Exact/AST matches are structural reference metrics, not semantic correctness.')


def normalize(task, policy):
    if policy not in POLICIES:
        raise ValueError('unknown normalization policy')
    original_header = task_header(task)
    node = ast.parse(original_header + '    pass\n').body[0]
    names = [a.arg for a in node.args.args]
    if not names or set(names) & set(dir(builtins)) or node.name in names:
        raise ValueError('requires nonempty non-builtin distinct arguments')
    canonical = [policy] if len(names) == 1 else ['a', 'b'] if len(names) == 2 else names
    mapping = dict(zip(names, canonical))
    pattern = r'\b(?:' + '|'.join(map(re.escape, names)) + r')\b'
    normalized = re.sub(pattern, lambda m: mapping[m[0]], task)
    header = task_header(normalized)
    if ast.parse(header+'    pass\n').body[0].name != node.name:
        raise ValueError('normalization changed function name')
    return {'prompt': normalized, 'header': header, 'original_header': original_header,
            'restore_map': dict(zip(canonical, names)),
            'mode': 'canonical' if len(names) in (1, 2) else 'passthrough_arity'}


def restore(text, transform):
    """Rename only AST variable/argument token spans in a single return function."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {'status': 'invalid_syntax', 'output': ''}
    if not text.startswith(transform['header']) or len(tree.body) != 1:
        return {'status': 'rejected_shape', 'output': ''}
    function = tree.body[0]
    forbidden = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                 ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await)
    expected = ast.parse(transform['header']+'    pass\n').body[0]
    if (not isinstance(function, ast.FunctionDef) or function.decorator_list or
        function.name != expected.name or ast.dump(function.args) != ast.dump(expected.args) or
        len(function.body) != 1 or not isinstance(function.body[0], ast.Return) or
        any(isinstance(n, forbidden) for n in ast.walk(function))):
        return {'status': 'rejected_shape', 'output': ''}
    mapping = transform['restore_map']
    # Reject capture of a pre-existing free name by a restored parameter.
    targets = set(mapping.values()) - set(mapping)
    if any(isinstance(n, ast.Name) and n.id in targets for n in ast.walk(function)):
        return {'status': 'rejected_capture', 'output': ''}
    # AST columns are UTF-8 byte offsets; tokenizer columns count characters.
    lines = text.splitlines(keepends=True)
    def position(node):
        line = lines[node.lineno-1]
        column = len(line.encode()[:node.col_offset].decode())
        return (node.lineno, column)
    replacements = {position(n): mapping[n.id] for n in ast.walk(function)
                    if isinstance(n, ast.Name) and n.id in mapping}
    replacements.update({position(n): mapping[n.arg] for n in function.args.args if n.arg in mapping})
    tokens = tokenize.generate_tokens(io.StringIO(text).readline)
    output = tokenize.untokenize([t._replace(string=replacements[t.start])
              if t.type == tokenize.NAME and t.start in replacements else t for t in tokens])
    if not output.startswith(transform['original_header']):
        raise ValueError('restoration did not preserve requested declaration')
    return {'status': 'ok', 'output': output}


def verify_source(source, dataset):
    root = Path(source)
    raw = (root/'body-argument-report.json').read_bytes()
    if digest(raw) != REPORT_SHA:
        raise ValueError('pinned full-response report differs')
    report = json.loads(raw)
    audit_body(report, dataset)
    model = root/'NppAI-body-argument-aliases.nppai'
    if report['argument_policy'] != 'fixed_arguments' or digest(model.read_bytes()) != report['model_sha256']:
        raise ValueError('frozen control checkpoint differs')
    checks = [('body-source/NppAI-broader-instruction.nppai', report['source']['model_sha256']),
              ('body-source/broader-report.json', report['source']['report_sha256']),
              ('initial-body-source/NppAI-broader-instruction.nppai', report['source']['initial_body_sha256'])]
    for path, expected in checks:
        if digest((root/path).read_bytes()) != expected:
            raise ValueError('preserved body provenance differs')
    return model, report


def summarize(cases):
    def counts(rows):
        return {'total':len(rows), **{arm:{metric:sum(c[arm]['score'][metric] for c in rows)
                for metric in METRICS} for arm in ARMS},
                'restoration_accepted':{p:sum(c[p]['restoration']['status']=='ok' for c in rows) for p in POLICIES}}
    return {'overall':counts(cases), 'name_length':{str(n):counts([c for c in cases if c['name_length']==n])
            for n in sorted({c['name_length'] for c in cases})}}


def paired(groups):
    return {split:{p:{metric:{label:sum(bool(c['direct_assisted']['score'][metric])==b and bool(c[p]['score'][metric])==a for c in cases)
            for label,b,a in (('both',True,True),('lost',True,False),('gained',False,True),('neither',False,False))}
            for metric in ('reference_ast','exact_response')} for p in POLICIES} for split,cases in groups.items()}


def audit(report, dataset, source):
    development = build(dataset,'fixed_arguments')[2]
    if (report['source_run_id'] != SOURCE_RUN or report['source_report_sha256'] != REPORT_SHA or
        report['model_sha256'] != source['model_sha256'] or report['groups'].keys() != development.keys()):
        raise ValueError('normalization source/groups differ')
    for split,rows in development.items():
        cases=report['groups'][split]
        expected={r['id']:r for r in rows}
        cached={r['id']:r for r in source['evaluation']['after'][split]['cases']}
        if len(cases)!=len(rows) or len({c['id'] for c in cases})!=len(rows):
            raise ValueError('missing/duplicate normalization cases')
        for case in cases:
            record=expected.get(case['id'])
            if record is None or any(case.get(k)!=v for k,v in record.items()):
                raise ValueError('normalization task/reference differs')
            direct=case['direct_assisted']
            if direct['score']!=score(record,direct['score']['output']) or direct['score']!=cached[case['id']]['task_signature_assisted']:
                raise ValueError('direct frozen output differs')
            for policy in POLICIES:
                item=case[policy]; transform=normalize(record['prompt'],policy)
                if item['transform']!=transform or not item['normalized_output'].startswith(transform['header']):
                    raise ValueError('prompt normalization/prefix differs')
                restored=restore(item['normalized_output'],transform)
                if item['restoration']!=restored or item['score']!=score(record,restored['output']):
                    raise ValueError('restoration/raw score differs')
                if transform['prompt']==record['prompt'] and item['normalized_output']!=direct['score']['output']:
                    raise ValueError('identity normalization changed frozen output')
            if len(ast.parse(task_header(record['prompt'])+'    pass\n').body[0].args.args)==2 and len({case[p]['normalized_output'] for p in POLICIES})!=1:
                raise ValueError('shared two-argument normalization differs')
        if report['summary'][split]!=summarize(cases):
            raise ValueError('normalization summary differs')
    if report['paired_to_direct']!=paired(report['groups']):
        raise ValueError('normalization pairs differ')
    return {'valid':True,'cases':sum(map(len,development.values())),'limits':LIMITS}


def run(source,dataset,output):
    path, cached = verify_source(source,dataset)
    development=build(dataset,'fixed_arguments')[2]
    import torch
    from scripts.check_dense_runtime_parity import load_weights,torch_model
    from scripts.identifier_curriculum import validate_initial_weights
    from scripts.train_instruction_experiment import generate
    torch.set_num_threads(2)
    weights=load_weights(path); validate_initial_weights(weights); model=torch_model(weights)
    groups={}
    with torch.no_grad():
        for split,rows in development.items():
            cases=[]
            for record in rows:
                task=record['prompt']; case=dict(record)
                direct=generate(model,task,256,assistant_prefix=task_header(task))
                case['direct_assisted']={'score':score(record,direct)}
                outputs={}
                for policy in POLICIES:
                    transform=normalize(task,policy)
                    key=(transform['prompt'],transform['header'])
                    if key not in outputs:
                        outputs[key]=generate(model,transform['prompt'],256,assistant_prefix=transform['header'])
                    text=outputs[key]; restored=restore(text,transform)
                    case[policy]={'transform':transform,'normalized_output':text,'restoration':restored,'score':score(record,restored['output'])}
                cases.append(case)
            groups[split]=cases
            print(json.dumps({'split':split,**summarize(cases)['overall']}),flush=True)
    report={'source_run_id':SOURCE_RUN,'source_report_sha256':REPORT_SHA,'model_sha256':cached['model_sha256'],
            'runtime':{'torch':torch.__version__,'threads':2,'python':sys.version.split()[0]},
            'generation':{'context':256,'new_tokens_per_call':160,'greedy':True},
            'limits':LIMITS,'groups':groups,'summary':{s:summarize(c) for s,c in groups.items()},'paired_to_direct':paired(groups)}
    evidence=audit(report,dataset,cached)
    destination=Path(output);destination.mkdir(parents=True,exist_ok=True)
    raw=encoded(report);(destination/'binding-report.json').write_bytes(raw)
    evidence['source_report_sha256']=digest(raw)
    (destination/'binding-audit.json').write_bytes(encoded(evidence))
    shutil.copytree(source,destination/'source',dirs_exist_ok=True)
    print(json.dumps({'audit':evidence,'summary':report['summary'],'paired_to_direct':report['paired_to_direct']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','dataset','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();run(args.source,args.dataset,args.output)
