"""Experimental Python-only pointer/gate readout; never a legacy .nppai model."""
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.resampled_name_training import digest


class GatedCopyModel(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base=base
        self.copy_q=nn.Linear(base.dim,base.dim,bias=False)
        self.copy_k=nn.Linear(base.dim,base.dim,bias=False)
        self.gate=nn.Linear(base.dim,1)
        nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-4.0)

    def components(self,tokens,source_mask):
        batch,length=tokens.shape
        if source_mask.shape!=tokens.shape or source_mask.dtype!=torch.bool:
            raise ValueError('source mask shape/type differs')
        if length>self.base.max_seq_len:raise ValueError('context overflow')
        positions=torch.arange(length,device=tokens.device)
        x=self.base.tok_embeddings(tokens)+self.base.pos_embeddings(positions)
        causal=torch.tril(torch.ones(length,length,device=tokens.device,dtype=torch.bool))
        for layer in self.base.layers:x=layer(x,causal[None,:,:])
        hidden=self.base.norm(x);logits=self.base.output(hidden)
        scores=self.copy_q(hidden)@self.copy_k(hidden).transpose(-2,-1)/(self.base.dim**0.5)
        eligible=source_mask[:,None,:]&causal[None,:,:]&(tokens[:,None,:]!=0)
        available=eligible.any(-1,keepdim=True)
        # Empty-source queries use a zero-mass dummy softmax entry; no NaNs.
        safe=eligible|((~available)&(positions[None,None,:]==0))
        attention=scores.masked_fill(~safe,float('-inf')).softmax(-1)*eligible
        pointer=aggregate(attention,tokens)
        gate=torch.where(available,self.gate(hidden).sigmoid().clamp(1e-5,1-1e-5),0.0)
        vocab=logits.softmax(-1)
        mixture=(1-gate)*vocab+gate*pointer
        return logits,pointer,gate,attention,mixture

    def forward(self,tokens,source_mask):
        mixture=self.components(tokens,source_mask)[-1]
        return mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()


def aggregate(attention,tokens):
    result=torch.zeros((*attention.shape[:2],256),device=attention.device,dtype=attention.dtype)
    return result.scatter_add(-1,tokens[:,None,:].expand_as(attention),attention)


def source_bounds(prompt):
    if '\0' in prompt:raise ValueError('NUL in prompt')
    return len(b'[USER]: '),len(b'[USER]: ')+len(prompt.encode())


def source_mask(tokens,prompt):
    left,right=source_bounds(prompt)
    positions=torch.arange(tokens.shape[1],device=tokens.device)
    return ((positions>=left)&(positions<right))[None,:].expand_as(tokens)


def generate(model,prompt,prefix='',budget=160):
    tokens=list(('[USER]: '+prompt+'\n[AI]:\n'+prefix).encode())
    if len(tokens)>model.base.max_seq_len:raise ValueError('prompt exceeds context')
    left,right=source_bounds(prompt)
    mask=[left<=i<right for i in range(len(tokens))]
    output=bytearray(prefix.encode())
    with torch.no_grad():
        for _ in range(budget):
            x=torch.tensor([tokens[-model.base.max_seq_len:]],dtype=torch.long)
            eligible=torch.tensor([mask[-model.base.max_seq_len:]],dtype=torch.bool)
            token=int(model(x,eligible)[0,-1].argmax())
            if token==0:break
            output.append(token);tokens.append(token);mask.append(False)
    return output.decode('utf-8',errors='replace')


class PromptAdapter(nn.Module):
    """Diagnostic prefixes start at position zero and never roll the context."""
    def __init__(self,model,prompt):
        super().__init__();self.model=model;self.prompt=prompt
    def forward(self,tokens):return self.model(tokens,source_mask(tokens,self.prompt))


def zip_bytes(entries):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in sorted(entries.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,raw)
    return stream.getvalue()


def export_bundle(model,path,source_model_id):
    from train_nppai import export_to_bin
    with tempfile.TemporaryDirectory() as temp:
        basepath=Path(temp)/'base.nppai';export_to_bin(model.base,str(basepath));base=basepath.read_bytes()
    arrays={'copy_q':model.copy_q.weight,'copy_k':model.copy_k.weight,
            'gate_weight':model.gate.weight,'gate_bias':model.gate.bias}
    entries={}
    for name,tensor in arrays.items():
        stream=io.BytesIO();np.save(stream,tensor.detach().cpu().numpy(),allow_pickle=False)
        entries[name+'.npy']=stream.getvalue()
    head=zip_bytes(entries)
    manifest={'format':'nppai-gated-copy','version':1,'native_compatible':False,
              'base_sha256':digest(base),'head_sha256':digest(head),'source_model_id':source_model_id}
    Path(path).write_bytes(zip_bytes({'base.nppai':base,'head.npz':head,
                                    'manifest.json':json.dumps(manifest,sort_keys=True).encode()}))


def bounded_zip(raw,expected):
    if len(raw)>8*1024*1024:raise ValueError('bundle exceeds byte limit')
    archive=zipfile.ZipFile(io.BytesIO(raw))
    names=archive.namelist()
    if len(names)!=len(expected) or set(names)!=set(expected):raise ValueError('unexpected/duplicate bundle members')
    if any(info.file_size>8*1024*1024 for info in archive.infolist()):raise ValueError('oversized bundle member')
    return {name:archive.read(name) for name in names}


def load_bundle(path):
    from scripts.check_dense_runtime_parity import load_weights,torch_model
    entries=bounded_zip(Path(path).read_bytes(),('base.nppai','head.npz','manifest.json'))
    manifest=json.loads(entries['manifest.json'])
    if (set(manifest)!={'format','version','native_compatible','base_sha256','head_sha256','source_model_id'} or
            manifest['format']!='nppai-gated-copy' or type(manifest['version'])!=int or manifest['version']!=1 or
            manifest['native_compatible'] is not False or not isinstance(manifest['source_model_id'],str)):
        raise ValueError('unsupported experimental bundle')
    if digest(entries['base.nppai'])!=manifest['base_sha256'] or digest(entries['head.npz'])!=manifest['head_sha256']:
        raise ValueError('bundle digest mismatch')
    weights=load_weights(None,raw=entries['base.nppai'])
    if not all(np.isfinite(a).all() for a in [weights['embedding'],weights['position'],weights['norm'],weights['output'],
                    *(a for layer in weights['layers'] for a in layer.values())]):raise ValueError('nonfinite base weights')
    model=GatedCopyModel(torch_model(weights))
    arrays=bounded_zip(entries['head.npz'],('copy_q.npy','copy_k.npy','gate_weight.npy','gate_bias.npy'))
    for name,parameter in [('copy_q',model.copy_q.weight),('copy_k',model.copy_k.weight),
                           ('gate_weight',model.gate.weight),('gate_bias',model.gate.bias)]:
        array=np.load(io.BytesIO(arrays[name+'.npy']),allow_pickle=False)
        if array.dtype!=np.dtype('float32') or tuple(array.shape)!=tuple(parameter.shape) or not np.isfinite(array).all():
            raise ValueError('invalid head tensor')
        with torch.no_grad():parameter.copy_(torch.from_numpy(array))
    return model.eval(),manifest
