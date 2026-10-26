"""Optional bounded neural beam decoding with structural Python checks.

No algorithm templates, reference answers or test cases are available here.
Syntax is a search constraint, never a claim of functional correctness.
"""
import ast
import builtins
import re
import symtable


def declaration(payload):
    if payload.assistant_prefix:
        header=payload.assistant_prefix
    else:
        signatures=re.findall(r'\b([A-Za-z_]\w*\([A-Za-z_][A-Za-z_0-9]*(?:,\s*[A-Za-z_][A-Za-z_0-9]*)*\))',payload.prompt)
        if len(set(signatures))!=1:return '',None
        header='def '+signatures[0]+':\n    '
    try:
        tree=ast.parse(header+'pass\n')
        f=tree.body[0]
        if len(tree.body)!=1 or not isinstance(f,ast.FunctionDef):return payload.assistant_prefix,None
        return header,(f.name,ast.dump(f.args))
    except SyntaxError:return payload.assistant_prefix,None


def acceptable(code,signature):
    try:
        tree=ast.parse(code)
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        if len(functions)!=1:return False
        f=functions[0]
        if signature and (f.name,ast.dump(f.args))!=signature:return False
        if any(not isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom)) for n in tree.body):return False
        table=symtable.symtable(code,'candidate.py','exec')
        allowed=set(dir(builtins))|set(table.get_identifiers())
        def bound(t):
            return all(not s.is_referenced() or not s.is_global() or s.get_name() in allowed for s in t.get_symbols()) and all(bound(c) for c in t.get_children())
        return bound(table)
    except (SyntaxError,ValueError,RecursionError):return False


class CachedLogits:
    """Request-local immutable causal KV branches; full recompute on sliding."""
    def __init__(self,w):self.w=w;self.previous={};self.current={}
    def __call__(self,tokens):
        import numpy as np
        w=self.w;key=tuple(tokens);parent=self.previous.get(key[:-1])
        def norm(x,weight):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)*weight
        if parent is None or len(tokens)>w['context']:
            tokens=tokens[-w['context']:];x=w['embedding'][tokens]+w['position'][:len(tokens)]
        else:x=w['embedding'][tokens[-1:]]+w['position'][len(tokens)-1:len(tokens)]
        cache=[]
        for index,layer in enumerate(w['layers']):
            h=norm(x,layer['norm1']);q,k,v=(h@layer[a] for a in ('q','k','v'))
            if parent is not None and len(key)<=w['context']:
                k=np.concatenate((parent[index][0],k));v=np.concatenate((parent[index][1],v))
            scores=q@k.T/np.sqrt(np.float32(w['dim']))
            if len(x)>1:scores[np.triu_indices(len(x),1)]=-np.inf
            a=np.exp(scores-scores.max(axis=-1,keepdims=True));a/=a.sum(axis=-1,keepdims=True)
            x=x+a@v@layer['o'];h=norm(x,layer['norm2']);gate=h@layer['gate']
            x=x+(gate/(1+np.exp(-gate))*(h@layer['up']))@layer['down'];cache.append((k,v))
        self.current[key]=cache
        return (norm(x[-1:],w['norm'])@w['output']).reshape(-1)
    def next_depth(self):self.previous,self.current=self.current,{}


def generate(backend,payload):
    import numpy as np
    from model_backends import GenerationResponse,format_prompt
    header,signature=declaration(payload)
    prefix=list((format_prompt(payload,backend.prompt_bytes)+header).encode())
    if len(prefix)>backend.weights['context']:raise ValueError('Formatted prompt exceeds model context')
    logits=CachedLogits(backend.weights)
    # Each tuple contains model log probability, emitted bytes and token count.
    beam=[(0.,b'',0)];finished=[];fallback=None
    for step in range(payload.max_new_tokens):
        expanded=[]
        for score,body,count in beam:
            values=logits(prefix+list(body))
            if not np.isfinite(values).all():raise RuntimeError('Non-finite model logits')
            shifted=values-values.max();logp=shifted-np.log(np.exp(shifted).sum())
            for token in np.argsort(-logp,kind='stable')[:8]:
                candidate=(score+float(logp[token]),body+(bytes([int(token)]) if token else b''),count+1)
                if token:
                    expanded.append(candidate)
                else:
                    code=header+body.decode('utf-8',errors='replace')
                    if fallback is None or candidate[0]>fallback[0]:fallback=candidate
                    if acceptable(code,signature):finished.append(candidate)
        logits.next_depth()
        beam=sorted(expanded,key=lambda c:c[0],reverse=True)[:8]
        if finished and (not beam or max(c[0] for c in finished)>=beam[0][0]):break
        if not beam:break
    choices=finished or ([fallback] if fallback else beam)
    winner=max(choices,key=lambda c:c[0]);reason='eos' if finished or fallback and not beam else 'length'
    if not finished and beam:
        winner=max(([fallback] if fallback else [])+beam,key=lambda c:c[0]);reason='eos' if winner is fallback else 'length'
    return GenerationResponse(model_id=backend.model_id,output=header+winner[1].decode('utf-8',errors='replace'),generated_tokens=winner[2],finish_reason=reason)
