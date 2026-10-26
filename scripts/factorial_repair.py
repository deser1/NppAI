"""Reproducible bounded repair of the released own model, with raw evidence."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_backends import GenerationRequest, DenseLocalBackend, format_prompt
from scripts.train_staged_protocol import TorchBackend

SOURCE='4859ad69e93ff2ca80d8951bca617476d118f8ef127abc4b313b8d2f0671c6fd'
WIDTH=256

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def answer(name,arg):
    return f'def {name}({arg}):\n    result = 1\n    for i in range(2, {arg} + 1):\n        result *= i\n    return result\n'

def record(name,arg,description,mode,ident):
    signature=f'{name}({arg})'
    prompt=f'Implement {signature}. '+description
    return dict(id=ident,request=dict(prompt=prompt,assistant_prefix=f'def {signature}:\n    ' if mode=='signature' else '',max_new_tokens=256),answer=answer(name,arg),function=name,
                cases=[dict(function=name,args=[n],kwargs={},expected=math.factorial(n)) for n in (0,1,2,5,10,15,20)])

def dataset():
    descriptions=['Return the factorial of a nonnegative integer.', 'Compute the product of integers from 1 to the input. Return 1 for zero.', 'Calculate factorial using a loop.', 'Zwróć silnię nieujemnej liczby całkowitej.', 'Oblicz iloczyn liczb od 1 do argumentu; dla zera zwróć 1.', 'Zwróć wynik mnożenia kolejnych liczb naturalnych do podanej wartości.', 'Compute n factorial; zero factorial equals one.', 'Return a factorial, without recursion.', 'Napisz funkcję obliczającą silnię.', 'Calculate the factorial and return its integer value.', 'Compute the factorial of the given non-negative integer.']
    descriptions += ['Return the product of integers from two through the input, using one for zero.', 'Multiply all positive integers not exceeding the argument. Start with one and return the product.', 'Calculate the factorial of the input integer. For input zero return one. Otherwise multiply successive positive integers.', 'Given a nonnegative integer argument, return its factorial as an integer result.', 'Return one times two times every next integer up to the supplied argument; the empty product is one.', 'Implement the nonnegative factorial operation with a running product and a for loop.', 'Zaimplementuj silnię: dla zera wynikiem jest jeden, dla dodatniej liczby pomnóż kolejne liczby aż do argumentu.', 'Compute factorial and return the product. Include the input itself in the multiplication.', 'Produce a Python function for factorial. Use an accumulator initialized to one and iterate from two up to the argument.', 'Calculate the factorial of a whole number greater than or equal to zero. Return an integer, without printing.', 'Return the product from one through the provided non-negative value, with zero factorial defined as one.']
    training=[record(name,arg,text,mode,f'train-{i}-{j}-{k}-{mode}') for i,name in enumerate(('silnia','factorial','oblicz_silnie','faktorial')) for j,arg in enumerate(('n','k','liczba')) for k,text in enumerate(descriptions) for mode in ('signature','plain')]
    for i,prompt in enumerate(('wygeneruj kod na silnię','Wygeneruj kod na silnię.','Napisz kod obliczający silnię.','Oblicz silnię w Pythonie.')):
        row=record('silnia','n','', 'plain',f'plain-{i}');row['request']['prompt']=prompt;training.append(row)
    evaluation=[record(name,arg,'Return the product of every integer from two through the argument, or one when it is zero.',mode,f'wording-{name}-{arg}-{mode}') for name in ('silnia','factorial') for arg in ('n','k','liczba') for mode in ('signature','plain')]
    evaluation += [record(name,'n','Compute factorial for a nonnegative integer.',mode,f'name-{name}-{mode}') for name in ('factorial_value','policz_silnie') for mode in ('signature','plain')]
    evaluation += [record('silnia','n','Wygeneruj kod na silnię dla nieujemnej liczby całkowitej n. Zwróć wynik.',mode,f'reproduction-{mode}') for mode in ('signature','plain')]
    return training,evaluation

def encode(row,context):
    import torch
    payload=GenerationRequest(**row['request'])
    prefix=(format_prompt(payload,WIDTH)+payload.assistant_prefix).encode()
    target=row['answer'].encode()
    pre=payload.assistant_prefix.encode()
    if not target.startswith(pre):raise ValueError('Incompatible supervision')
    tokens=list(prefix)+list(target[len(pre):])+[0]
    if len(tokens)>context:raise ValueError('No training truncation permitted')
    x,y=torch.tensor(tokens[:-1]),torch.tensor(tokens[1:])
    y[:len(prefix)-1]=-100
    return x,y

def run(source,output,image=None):
    import torch
    import torch.nn.functional as F
    from scripts.check_dense_runtime_parity import load_weights,torch_model
    from scripts.training_repro import state_digest
    from train_nppai import export_to_bin
    if sha(source)!=SOURCE:raise ValueError('Wrong released initialization')
    torch.set_num_threads(2);torch.manual_seed(47);torch.use_deterministic_algorithms(True)
    training,evaluation=dataset()
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(seed=47,updates=4096,learning_rate=.0003,batch_size=8,old_replay=4,prompt_bytes=WIDTH,source_sha256=SOURCE)
    (output/'protocol.json').write_text(json.dumps(dict(protocol=protocol,training=training,evaluation=evaluation),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    old=json.loads((ROOT/'evaluation/staged-protocol-v1/train.json').read_bytes())
    initial=torch_model(load_weights(source))
    for layer in initial.modules():
        if isinstance(layer,torch.nn.RMSNorm):layer.eps=1e-5
    model=copy.deepcopy(initial).train()
    old_samples=[]
    for row in old:
        p=list(format_prompt(GenerationRequest(prompt=row['prompt'])).encode());seq=p+list(row['answer'].encode())+[0]
        x,y=torch.tensor(seq[:-1]),torch.tensor(seq[1:]);y[:len(p)-1]=-100;old_samples.append((x,y))
    new_samples=[encode(r,model.max_seq_len) for r in training]
    rng=random.Random(47);optimizer=torch.optim.AdamW(model.parameters(),lr=.0003);trace=[]
    for step in range(4096):
        oi=rng.sample(range(len(old_samples)),4);ni=rng.sample(range(len(new_samples)),4)
        samples=[old_samples[i] for i in oi]+[new_samples[i] for i in ni]
        x=torch.nn.utils.rnn.pad_sequence([s[0] for s in samples],batch_first=True)
        y=torch.nn.utils.rnn.pad_sequence([s[1] for s in samples],batch_first=True,padding_value=-100)
        losses=F.cross_entropy(model(x).transpose(1,2),y,ignore_index=-100,reduction='none')
        loss=(losses.sum(1)/(y!=-100).sum(1)).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        trace.append(dict(step=step+1,old_indices=oi,new_indices=ni,loss=float(loss.detach())))
        if step%128==0:print('step',step+1,'loss',trace[-1]['loss'],flush=True)
    model.eval();path=output/'NppAI-factorial.nppai';export_to_bin(model,str(path))
    (output/'trace.json').write_text(json.dumps(trace)+'\n')
    weights_before=state_digest(model)
    report=dict(protocol=protocol,model_sha256=sha(path),trace_sha256=sha(output/'trace.json'),protocol_sha256=sha(output/'protocol.json'),old_replay_sha256=sha(ROOT/'evaluation/staged-protocol-v1/train.json'),torch=torch.__version__,readouts={},promotion=False,limitations='Authored and previously exposed factorial probes; one seed; no broad code-generation claim. Plain Polish prompts are training memorization probes. New names measured separately.')
    for arm,backend in [('initial',TorchBackend(initial,'source')),('initial_profile',DenseLocalBackend(str(source),prompt_bytes=WIDTH,python_functions=True)),('repair_raw',TorchBackend(model,'repair')),('repair',DenseLocalBackend(str(path),prompt_bytes=WIDTH,python_functions=True))]:
        rows=[]
        for row in evaluation+[r for r in training if r['id'].startswith('plain-')]:
            payload=GenerationRequest(**row['request'])
            if arm=='repair_raw':
                # Same fixed boundary, greedy readout; no decoder intervention.
                backend.prompt_bytes=WIDTH
            generated=backend.generate(payload).output
            item=dict(id=row['id'],request=row['request'],output=generated,exact=generated==row['answer'],cases=row['cases'],source_sha256=hashlib.sha256(generated.encode()).hexdigest())
            if image:
                import sandbox_validate
                code,diagnostics=(sandbox_validate.validate(generated,'tests',row['cases'],image) if generated.strip() else (1,'Empty candidate; no execution'))
                if code not in (0,1):raise RuntimeError('Docker unavailable: '+diagnostics)
                item.update(functional_pass=code==0,diagnostics=diagnostics)
            rows.append(item)
        report['readouts'][arm]=rows
        print(arm,'exact',sum(r['exact'] for r in rows),'/',len(rows),flush=True)
    baseline=TorchBackend(initial,'source');repaired=TorchBackend(model,'repair')
    retained=[]
    for row in old:
        p=GenerationRequest(prompt=row['prompt'],max_new_tokens=128)
        a=baseline.generate(p).output;b=repaired.generate(p).output
        retained.append(dict(id=row['id'],before=a,after=b,answer=row['answer'],before_exact=a==row['answer'],after_exact=b==row['answer']))
    report['retention']=retained
    # Actual runtime/Torch greedy parity on the user reproduction, independent of beam.
    numpy_raw=DenseLocalBackend(str(path),prompt_bytes=WIDTH)
    aligned=TorchBackend(model,'repair');aligned.prompt_bytes=WIDTH
    report['parity']=[dict(id=r['id'],passed=numpy_raw.generate(GenerationRequest(**r['request'])).output==aligned.generate(GenerationRequest(**r['request'])).output) for r in evaluation[-2:]+[training[-4]]]
    report['readout_weights_unchanged']=state_digest(model)==weights_before
    if image:
        import sandbox_validate
        report['validator_image']=image
        report['positive_controls']=[sandbox_validate.validate(r['answer'],'tests',r['cases'],image)[0] for r in evaluation]
        report['negative_control']=sandbox_validate.validate('def silnia(n):\n    return 1\n','tests',evaluation[-1]['cases'],image)[0]
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('report',output/'report.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--image');a=p.parse_args();run(a.source,a.output,a.image)
