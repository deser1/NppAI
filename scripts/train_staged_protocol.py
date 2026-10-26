"""Fixed-budget paired NppAI stage-protocol pilot; own dense model only."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_backends import GenerationRequest, GenerationResponse, format_prompt
from staged_generation import stage_values


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(root):
    manifest = json.loads((root/'manifest.json').read_bytes())
    train = root/'train.json'
    if digest(train) != manifest['files']['train.json']['sha256']:
        raise ValueError('Training dataset hash mismatch')
    rows = json.loads(train.read_bytes())
    if len(rows) != manifest['files']['train.json']['count']:
        raise ValueError('Training count mismatch')
    return manifest, rows


def supervision(record, stage):
    if stage not in ('direct', 'specification', 'plan', 'code'):
        raise ValueError('Unknown stage')
    if stage == 'direct':
        return GenerationRequest(prompt=record['prompt']), record['answer']
    prior = {}
    for name in ('specification', 'plan', 'code'):
        if name == stage:
            payload = GenerationRequest(**stage_values(name, record['prompt'], '', prior, 128))
            return payload, record['answer'] if name == 'code' else record[name]
        prior[name] = record[name]
    raise ValueError('Unknown stage')


def encoded_sample(payload, target, context):
    import torch
    prefix = list(format_prompt(payload).encode('utf-8'))
    seq = prefix + list(target.encode('utf-8')) + [0]
    if len(seq) > context:
        raise ValueError('Full supervision exceeds context; no truncation allowed')
    x, y = torch.tensor(seq[:-1]), torch.tensor(seq[1:])
    y[:len(prefix)-1] = -100
    return x, y


def expanded_initial(source, context=1024):
    import torch
    from train_nppai import NppAIModel
    from scripts.check_dense_runtime_parity import load_weights, torch_model
    weights = load_weights(source)
    old = torch_model(weights)
    if context < weights['context']:
        raise ValueError('Cannot shrink initialization context')
    torch.manual_seed(42)
    model = NppAIModel(256, weights['dim'], weights['hidden'], len(weights['layers']), context)
    state = old.state_dict()
    state['pos_embeddings.weight'] = torch.cat((old.pos_embeddings.weight.detach(),
        model.pos_embeddings.weight.detach()[weights['context']:]), dim=0)
    model.load_state_dict(state)
    # Match the production NumPy/native dense runtime's RMSNorm epsilon.
    for layer in model.modules():
        if isinstance(layer, torch.nn.RMSNorm):
            layer.eps = 1e-5
    return model


class TorchBackend:
    """FP32 greedy readout with the same formatting, EOS and budget as dense local."""
    def __init__(self, model, model_id):
        self.model, self.model_id = model.eval(), model_id
        self.prompt_bytes = 0

    def generate(self, payload):
        import torch
        tokens = list((format_prompt(payload, self.prompt_bytes) + payload.assistant_prefix).encode('utf-8'))
        if len(tokens) > self.model.max_seq_len:
            raise ValueError("Formatted prompt exceeds this model's byte context")
        output = bytearray(payload.assistant_prefix.encode('utf-8'))
        reason = 'length'
        with torch.no_grad():
            for count in range(1, payload.max_new_tokens + 1):
                logits = self.model(torch.tensor([tokens[-self.model.max_seq_len:]]))[0,-1]
                if not torch.isfinite(logits).all():
                    raise RuntimeError('Non-finite logits')
                token = int(logits.argmax())
                if token == 0:
                    reason = 'eos'
                    break
                tokens.append(token)
                output.append(token)
        return GenerationResponse(model_id=self.model_id, output=output.decode('utf-8',errors='replace'),
            generated_tokens=count, finish_reason=reason)


def train_arm(initial, records, schedule, arm, context, output):
    import torch
    import torch.nn.functional as F
    model = copy.deepcopy(initial).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    stages = ('direct', 'specification', 'plan', 'code')
    samples = {(i, stage):encoded_sample(*supervision(r,stage), context)
               for i,r in enumerate(records) for stage in stages}
    trace = []
    start = time.monotonic()
    for step, indices in enumerate(schedule):
        assignments = ['direct'] * 4 if arm == 'direct' else [stages[(step+j)%4] for j in range(4)]
        batch = [samples[(i,s)] for i,s in zip(indices, assignments)]
        x = torch.nn.utils.rnn.pad_sequence([v[0] for v in batch],batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence([v[1] for v in batch],batch_first=True,padding_value=-100)
        loss = F.cross_entropy(model(x).reshape(-1,256),y.reshape(-1),ignore_index=-100)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        trace.append({'step':step+1,'base_indices':indices,'stages':assignments,'loss':float(loss.detach()),
                      'input_tokens':int((x!=0).sum()),'target_tokens':int((y!=-100).sum())})
        if step%128==0 or step==len(schedule)-1:
            print(f'{arm} step={step+1}/{len(schedule)} loss={trace[-1]["loss"]:.5f}',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    (output/'trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    return model.eval(), {'steps':len(schedule),'initial_loss':trace[0]['loss'],'final_loss':trace[-1]['loss'],
        'elapsed_seconds':time.monotonic()-start,'input_tokens':sum(v['input_tokens'] for v in trace),
        'target_tokens':sum(v['target_tokens'] for v in trace),'trace_sha256':digest(output/'trace.json')}


def functional(report, records, image):
    import sandbox_validate
    by_id = {r['id']:r for r in records}
    results = {}
    for arm in ('staged','direct_code_budget','direct_total_budget'):
        cases = []
        for row in report['cases']:
            source = row['arms'][arm]['score']['output']
            item = {'id':row['id'],'source':source,'status':'empty','diagnostics':'','exit_code':None}
            if source.strip():
                code, diagnostics = sandbox_validate.validate(source,'tests',by_id[row['id']]['cases'],image)
                item.update(status='passed' if code==0 else 'candidate_failed' if code==1 else 'infrastructure_failure',
                            exit_code=code,diagnostics=diagnostics)
                if code not in (0,1):
                    raise RuntimeError('Docker validation infrastructure failed: '+diagnostics)
            cases.append(item)
        results[arm] = {'total':len(cases),'passed':sum(c['status']=='passed' for c in cases),'cases':cases}
    return {'image':image,'arms':results}


def run(source, dataset, output, *, image=None):
    import torch
    from train_nppai import export_to_bin
    from scripts.evaluate_task_plan import evaluate
    from scripts.training_repro import state_digest
    manifest, training = load_data(dataset)
    protocol = manifest['protocol']
    if (protocol['steps_per_arm'],protocol['batch_size'],protocol['context_bytes'],protocol['seed']) != (1024,4,1024,42):
        raise ValueError('Frozen pilot budget differs')
    if digest(source) != protocol['initial_model_sha256']:
        raise ValueError('Pinned initialization differs')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    initial = expanded_initial(source, protocol['context_bytes'])
    initial_id = state_digest(initial)
    rng = random.Random(protocol['seed'])
    schedule = [rng.sample(range(len(training)),4) for _ in range(protocol['steps_per_arm'])]
    output.mkdir(parents=True,exist_ok=True)
    # Fix and hash evaluation files before optimization; do not read their content yet.
    eval_path = dataset/'new-evaluation.json'
    if digest(eval_path) != manifest['files']['new-evaluation.json']['sha256']:
        raise ValueError('Frozen new evaluation bytes changed')
    old_path = ROOT/'evaluation/instruction-v2/test.json'
    old_hash = digest(old_path)
    metadata = {'protocol':protocol,'manifest_sha256':digest(dataset/'manifest.json'),
        'initial_expanded_state_sha256':initial_id,'schedule_sha256':hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
        'runtime':{'torch':torch.__version__,'python':sys.version,'threads':torch.get_num_threads(),
                   'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'rmsnorm_epsilon':1e-5},
        'original_context':256,'new_context':1024,'arms':{},'promotion':False,
        'limits':'One seed, correlated authored templates and previously exposed operation families. Gold prior artifacts during training; autonomous generated artifacts during evaluation. Equal updates/batch/task schedule, unequal stage objectives/token counts and compute. No benchmark-wide reasoning claim.'}
    metadata['source_sha256']={p:digest(ROOT/p) for p in ('scripts/train_staged_protocol.py',
        'scripts/build_stage_dataset.py','staged_generation.py','scripts/evaluate_task_plan.py','train_nppai.py')}
    models = {}
    for arm in protocol['arms']:
        if state_digest(initial) != initial_id:
            raise ValueError('Shared initialization changed')
        model, summary = train_arm(initial,training,schedule,arm,1024,output/arm)
        path = output/arm/'experimental.nppai'
        export_to_bin(model,str(path))
        models[arm] = TorchBackend(model,'sha256:'+digest(path))
        metadata['arms'][arm] = {**summary,'checkpoint_sha256':digest(path),
            'initial_state_sha256':initial_id,'final_state_sha256':state_digest(model)}
        (output/'training-report.json').write_text(json.dumps(metadata,indent=2)+'\n')
    # Evaluation content is first read only after BOTH optimizations finish.
    if digest(eval_path) != manifest['files']['new-evaluation.json']['sha256'] or digest(old_path)!=old_hash:
        raise ValueError('Evaluation changed during training')
    fresh = json.loads(eval_path.read_bytes())
    exposed = json.loads(old_path.read_bytes())
    # One first training record per family; disclose these as memorization probes.
    seen = set(); probe=[]
    for r in training:
        if r['family'] not in seen:
            probe.append(r); seen.add(r['family'])
    initial_backend=TorchBackend(initial,'state:'+initial_id)
    metadata['initial_training_probe_direct']=[]
    from scripts.evaluate_staged_generation import score_output
    for record in probe:
        response=initial_backend.generate(GenerationRequest(prompt=record['prompt'],max_new_tokens=128))
        metadata['initial_training_probe_direct'].append({'id':record['id'],'score':score_output(record,response.output)})
    metadata['initial_training_probe_exact']=sum(c['score']['exact_response'] for c in metadata['initial_training_probe_direct'])
    if image:
        import sandbox_validate
        controls=[]
        for record in fresh:
            code,diagnostics=sandbox_validate.validate(record['answer'],'tests',record['cases'],image)
            if code!=0:
                raise RuntimeError('Authored reference validator control failed: '+diagnostics)
            controls.append({'id':record['id'],'exit_code':code})
        code,diagnostics=sandbox_validate.validate('def suma(a, b):\n    return None\n','tests',fresh[0]['cases'],image)
        if code!=1:
            raise RuntimeError('Incorrect candidate validator control did not fail')
        metadata['validator_controls']={'authored_references':controls,'incorrect_candidate_exit_code':code}
    for arm,backend in models.items():
        from model_backends import DenseLocalBackend
        dense=DenseLocalBackend(str(output/arm/'experimental.nppai'))
        parity=[]
        for stage in ('direct','specification','plan','code'):
            payload,_=supervision(training[0],stage)
            a,b=backend.generate(payload),dense.generate(payload)
            parity.append({'stage':stage,'request':payload.model_dump(),'torch':a.model_dump(),
                'dense':b.model_dump(),'equal':a.model_dump()==b.model_dump()})
        (output/arm/'runtime-parity.json').write_text(json.dumps(parity,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        metadata['arms'][arm]['exported_readout_parity_4_training_requests']=all(p['equal'] for p in parity)
        for group,records in (('training_probe',probe),('exposed_v2',exposed),('new_synthetic',fresh)):
            report = evaluate(backend,records,tokens_per_stage=128)
            (output/arm/(group+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(json.dumps({'arm':arm,'group':group,'summary':report['summary']}),flush=True)
            if group=='training_probe':
                before={c['id']:c['score']['exact_response'] for c in metadata['initial_training_probe_direct']}
                after={c['id']:c['arms']['direct_code_budget']['score']['exact_response'] for c in report['cases']}
                metadata['arms'][arm]['training_probe_retention']={
                    'initial_exact':sum(before.values()),'retained_exact':sum(before[k] and after[k] for k in before),
                    'lost_ids':[k for k in before if before[k] and not after[k]]}
        if image:
            report=json.loads((output/arm/'new_synthetic.json').read_bytes())
            result=functional(report,fresh,image)
            (output/arm/'functional.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        metadata['arms'][arm]['evaluation_parameters_unchanged'] = state_digest(backend.model)==metadata['arms'][arm]['final_state_sha256']
        if not metadata['arms'][arm]['evaluation_parameters_unchanged']:
            raise RuntimeError('Evaluation changed model parameters')
    metadata['execution_status']='docker' if image else 'not_run'
    metadata['evaluation_sha256']={'new_synthetic':digest(eval_path),'exposed_v2':old_hash}
    (output/'training-report.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-model',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=ROOT/'evaluation/staged-protocol-v1')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--image')
    args=parser.parse_args()
    run(args.initial_model,args.dataset,args.output,image=args.image)
