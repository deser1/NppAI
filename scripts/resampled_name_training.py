"""Matched fixed versus per-draw function aliases; no generated-code execution."""
import argparse
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import sys
import tokenize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prompt_robustness import augment_training_records
from scripts.run_local_model_smoke import pinned_backend, SOURCE_RUN, REPORT_SHA256

STEPS = 1024
POLICIES = ('fixed', 'resampled')
SOURCE_MODEL_ID = 'sha256:161d6303c7e77f005cbabba5a1790b9d6a6dfc1812c121bfbb3f1cb6820e5b1a'
OPTIMIZER = {'name':'AdamW','lr':0.001,'seed':42,'batch_size':4,'clip':1.0,'reset':True}
METRICS = ('syntax_function', 'requested_name', 'requested_signature', 'reference_ast', 'exact_response')
SPLIT_HASHES = {
    'train': '7d729ccac90a001853663a661398cee21e66f7c2a57e749b6a855db3d673f3c6',
    'validation': 'c4963e993c9ebcd1c34f89597bb3a4b06ea4ed9baa62354198e3042fe991b445',
    'test': 'b174bf1f980c8a11415e8174f9d6eaac91faf8abd38ceee0610d910e278fa56d'}


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def alias(key, prefix='nc_'):
    return prefix + digest(key.encode())[:8]


def rename(record, name):
    old = record['function']
    tokens = list(tokenize.generate_tokens(io.StringIO(record['answer']).readline))
    answer = tokenize.untokenize([t._replace(string=name) if t.type == tokenize.NAME and t.string == old else t for t in tokens])
    prompt = re.sub(r'\b' + re.escape(old) + r'\b', name, record['prompt'])
    if prompt == record['prompt'] or not answer.startswith('def ' + name + '('):
        raise ValueError('function transformation failed')
    result = {**record, 'function': name, 'prompt': prompt, 'answer': answer}
    frame(result)  # never truncate reference bytes
    return result


def frame(record):
    prefix = ('[USER]: ' + record['prompt'] + '\n[AI]:\n').encode()
    raw = prefix + record['answer'].encode() + b'\0'
    if b'\0' in raw[:-1] or len(raw) > 256:
        raise ValueError('invalid or overflowing reference frame')
    return prefix, raw


def read_split(dataset, split):
    raw = (dataset / (split + '.json')).read_bytes()
    if digest(raw) != SPLIT_HASHES[split]:
        raise ValueError('frozen split differs')
    return json.loads(raw)


def training_data(dataset):
    # This path never opens validation/test references.
    rows = augment_training_records(read_split(dataset, 'train'), 'four_forms')
    for row in rows:
        frame(row)
    return rows


def development_data(dataset):
    result = {}
    for split in ('validation', 'test'):
        rows = read_split(dataset, split)
        result[split + '/original'] = rows
        result[split + '/renamed'] = [rename(r, alias(split + '/' + r['function'], 'ev_')) for r in rows]
    return result


def schedule(rows, policy, steps=STEPS):
    if policy not in POLICIES:
        raise ValueError('unknown policy')
    rng = random.Random(42)
    for step in range(steps):
        selected = [('original', r) for r in rng.sample(rows, 2)]
        for slot, record in enumerate(rng.sample(rows, 2)):
            key = record['function'] if policy == 'fixed' else f'{step}/{slot}'
            selected.append(('renamed', rename(record, alias(policy + '/' + key))))
        rng.shuffle(selected)
        yield selected


def schedule_evidence(rows, policy, steps=STEPS):
    order, content = hashlib.sha256(), hashlib.sha256()
    names, draws = set(), {'original': 0, 'renamed': 0}
    targets = 0
    for batch in schedule(rows, policy, steps):
        order.update(encoded([(pool, r['id']) for pool, r in batch]))
        content.update(encoded(batch))
        for pool, record in batch:
            draws[pool] += 1
            targets += len(record['answer'].encode()) + 1
            if pool == 'renamed':
                names.add(record['function'])
    if policy == 'resampled' and len(names) != steps * 2:
        raise ValueError('per-draw alias collision')
    return {'order_sha256': order.hexdigest(), 'content_sha256': content.hexdigest(),
            'draws': draws, 'unique_aliases': len(names), 'response_targets': targets}


def score(record, source):
    result = dict.fromkeys(METRICS, False)
    result['exact_response'] = source == record['answer']
    reference = ast.parse(record['answer'])
    try:
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            return result
        fn, expected = tree.body[0], reference.body[0]
        result.update(syntax_function=True, requested_name=fn.name == expected.name,
                      requested_signature=(fn.name == expected.name and ast.dump(fn.args) == ast.dump(expected.args)),
                      reference_ast=ast.dump(tree) == ast.dump(reference))
    except (SyntaxError, ValueError):
        pass
    return result


def summary(cases):
    return {'total': len(cases), **{k: sum(c['metrics'][k] for c in cases) for k in METRICS}}


def evaluate(model, groups):
    from scripts.train_instruction_experiment import generate
    return {group: [{'id': r['id'], 'output': output, 'metrics': score(r, output)}
                    for r in rows for output in [generate(model, r['prompt'], 256)]]
            for group, rows in groups.items()}


def retention(before, after):
    return {group: {metric: {'retained': sum(a['metrics'][metric] and b['metrics'][metric] for a,b in zip(prior, after[group])),
                            'lost': sum(a['metrics'][metric] and not b['metrics'][metric] for a,b in zip(prior, after[group])),
                            'gained': sum(not a['metrics'][metric] and b['metrics'][metric] for a,b in zip(prior, after[group]))}
                    for metric in ('requested_signature', 'exact_response')}
            for group, prior in before.items()}


def audit(report, dataset):
    rows = training_data(dataset)
    groups = development_data(dataset)
    if report['steps'] != STEPS or report['source_run_id'] != SOURCE_RUN or report['source_report_sha256'] != REPORT_SHA256:
        raise ValueError('training provenance differs')
    if report['schedule'] != schedule_evidence(rows, report['policy']):
        raise ValueError('training schedule differs')
    if report['development_sha256'] != digest(encoded(groups)):
        raise ValueError('development records differ')
    if (report['training_sha256'] != digest(encoded(rows)) or
            report['initial_model_id'] != SOURCE_MODEL_ID or
            report['optimizer'] != OPTIMIZER or not report['exported_weights_equal']):
        raise ValueError('initialization, training or export metadata differs')
    for stage in ('before', 'after'):
        if set(report[stage]) != set(groups):
            raise ValueError('development groups differ')
        for group, references in groups.items():
            cases = report[stage][group]
            if [c['id'] for c in cases] != [r['id'] for r in references]:
                raise ValueError('missing, duplicate or reordered cases')
            for r,c in zip(references,cases):
                if c['metrics'] != score(r,c['output']):
                    raise ValueError('raw output score differs')
            if report['summary'][stage][group] != summary(cases):
                raise ValueError('summary differs')
    if report['retention'] != retention(report['before'], report['after']):
        raise ValueError('retention differs')
    if len(report['loss_endpoints']) != 2 or not all(math.isfinite(v) for v in report['loss_endpoints']):
        raise ValueError('nonfinite loss')
    return {'valid': True, 'development_sha256': report['development_sha256']}


def sample(record):
    import torch
    prefix, raw = frame(record)
    x = torch.tensor(list(raw[:-1]), dtype=torch.long)
    y = torch.tensor(list(raw[1:]), dtype=torch.long)
    y[:len(prefix)-1] = -100
    return x,y


def run(source, dataset, output, policy):
    import torch
    import torch.nn.functional as F
    from train_nppai import export_to_bin
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    backend = pinned_backend(source)
    if backend.model_id != SOURCE_MODEL_ID:
        raise ValueError('unexpected initialization model')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = torch_model(backend.weights)
    rows, groups = training_data(dataset), development_data(dataset)
    train_names = {r['function'] for r in rows}
    dev_names = {r['function'] for rs in groups.values() for r in rs}
    if train_names & dev_names or any(n.startswith('nc_') for n in dev_names):
        raise ValueError('training/development name overlap')
    model.eval()
    before = evaluate(model, groups)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    losses = []
    model.train()
    for step, selected in enumerate(schedule(rows, policy)):
        batch = [sample(r) for _,r in selected]
        x = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True, padding_value=-100)
        loss = F.cross_entropy(model(x).reshape(-1,256), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or step == STEPS-1:
            losses.append(float(loss.detach()))
        if step % 128 == 0 or step == STEPS-1:
            print(f'{policy} step={step+1} loss={float(loss.detach()):.6f}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'experimental.nppai'
    model.eval()
    export_to_bin(model, str(checkpoint))
    loaded = torch_model(load_weights(checkpoint))
    if any(not torch.equal(a,b) for a,b in zip(model.parameters(),loaded.parameters())):
        raise RuntimeError('exported weights differ')
    loaded.eval()
    after = evaluate(loaded, groups)
    report = {'policy': policy, 'steps': STEPS, 'source_run_id': SOURCE_RUN,
              'source_report_sha256': REPORT_SHA256, 'initial_model_id': backend.model_id,
              'model_sha256': digest(checkpoint.read_bytes()), 'schedule': schedule_evidence(rows,policy),
              'training_sha256': digest(encoded(rows)), 'development_sha256': digest(encoded(groups)),
              'before': before, 'after': after, 'loss_endpoints': losses,
              'summary': {stage: {g: summary(c) for g,c in data.items()} for stage,data in [('before',before),('after',after)]},
              'retention': retention(before,after), 'exported_weights_equal': True,
              'optimizer': OPTIMIZER,
              'runtime': {'torch':torch.__version__,'threads':2,'context':256,'new_tokens':160},
              'limits': 'One seed, 1024-update diagnostic on exposed synthetic development tasks. '
                        'Original and renamed examples are 50/50 by draw, not by bytes. '
                        'AST/exact equality is not functional correctness. No prefix assistance, '
                        'fresh-probe access, smoke-probe optimization, generated-source execution or promotion.'}
    audit(report,dataset)
    (output/'report.json').write_bytes(encoded(report))
    print(json.dumps({'policy':policy,'model_sha256':report['model_sha256'],
                      'schedule':report['schedule'],'summary':report['summary'],'retention':report['retention']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=Path('evaluation/instruction-v3'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy',choices=POLICIES,required=True)
    args=parser.parse_args()
    run(args.source,args.dataset,args.output,args.policy)
