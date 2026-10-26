"""Fail-closed publication gate; report failed families without hiding them."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.factorial_repair import SOURCE,dataset,sha


def qualify(root):
    r=json.loads((root/'report.json').read_bytes())
    if r['model_sha256']!=sha(root/'NppAI-factorial.nppai'):raise ValueError('Checkpoint mismatch')
    if r['protocol_sha256']!=sha(root/'protocol.json') or r['trace_sha256']!=sha(root/'trace.json'):raise ValueError('Evidence changed')
    if r['protocol']['source_sha256']!=SOURCE or r['protocol']['updates']!=4096:raise ValueError('Protocol mismatch')
    if not r.get('readout_weights_unchanged') or not all(v['passed'] for v in r['parity']) or len(r['parity'])!=3:raise ValueError('Runtime parity/identity missing')
    if not r.get('validator_image','').startswith('sha256:') or r.get('positive_controls')!=[0]*18 or r.get('negative_control')!=1:raise ValueError('Real Docker validator controls missing')
    train,ev=dataset();expected=ev+[v for v in train if v['id'].startswith('plain-')]
    totals={}
    for arm,rows in r['readouts'].items():
        if [v['id'] for v in rows]!=[v['id'] for v in expected]:raise ValueError('Incomplete readout')
        for row,source in zip(rows,expected):
            if row['request']!=source['request'] or row['cases']!=source['cases'] or row['source_sha256']!=hashlib.sha256(row['output'].encode()).hexdigest():raise ValueError('Functional evidence not bound to output')
        totals[arm]={prefix:dict(total=sum(v['id'].startswith(prefix) for v in rows),passed=sum(v.get('functional_pass') is True for v in rows if v['id'].startswith(prefix))) for prefix in ('wording-','name-','reproduction-','plain-')}
    repair=totals['repair']
    if repair['reproduction-']['passed']!=2 or repair['plain-']['passed']!=4 or repair['wording-']['passed']<8:raise ValueError('Bounded functional repair gate failed')
    retained=r['retention']
    if len(retained)!=96 or any(v['before']==v['answer'] and v['after']!=v['answer'] for v in retained):raise ValueError('Original direct hit lost')
    old=json.loads((ROOT/'evaluation/staged-protocol-v1/train.json').read_bytes())
    if r['old_replay_sha256']!=sha(ROOT/'evaluation/staged-protocol-v1/train.json') or [(v['id'],v['answer']) for v in retained]!=[(v['id'],v['answer']) for v in old]:raise ValueError('Original retention set changed')
    before=sum(v['before']==v['answer'] for v in retained);after=sum(v['after']==v['answer'] for v in retained)
    if before!=88:raise ValueError('Pinned initialization no longer reproduces88 original hits')
    summary=dict(model_sha256=r['model_sha256'],functional=totals,original_direct=dict(before=before,after=after,total=96),gate='passed',scope='Bounded factorial repair; no stable/general-coding promotion',windows_docker_apply_undo='not_verified')
    (root/'qualified-report.json').write_text(json.dumps(summary,indent=2)+'\n')
    card=f'''# NppAI factorial repair — experimental

Own dense v2 model, initialized from rc.2 `{SOURCE}`; 4096 updates, seed47,
batch8 (four original direct examples and four factorial examples), AdamW0.0003.
532 authored training records; per-example completion loss; fixed256-byte
prompt boundary. Optional width8 neural beam search uses generic AST/signature
and name-binding checks. An unambiguous requested function declaration may be
prefilled mechanically; all algorithm/body bytes come from model logits.
No test answers or algorithm templates are used by the decoder.

Model SHA-256: `{r['model_sha256']}`.

Docker-tested result counts (raw outputs/cases/controls in report.json):

```json
{json.dumps(totals,indent=2)}
```

Original direct training probes: {before}/96 before, {after}/96 after, no original
exact hit lost. This measures memorization retention, not unseen accuracy.
Plain Polish requests are also training probes. Wording and new-name rows are
authored/exposed development probes; one seed, one narrow operation family.
The generic search decoder can still produce syntactically valid wrong code.
Configured task tests remain mandatory before Apply. The optional fixed-boundary
profile rejects tasks plus context longer than256 UTF-8 bytes, without truncation;
long repair feedback may therefore be rejected. These results replace no
previous public0/12 benchmark result and do not establish general code quality.

## Windows quick start

Download NppAI-factorial.nppai and NppAI-factorial-service.zip from this release.
Use the x64 plugin from v0.5.0-rc.2. Extract the service ZIP, open PowerShell there:

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-inference.txt
.\\.venv\\Scripts\\python.exe scripts/start_nppai.py --model C:\\NppAI\\NppAI-factorial.nppai --prompt-bytes 256 --python-functions --notepad 'C:\\Notepad++\\notepad++.exe'
```

Open an empty .py document and ask `wygeneruj kod na silnię`.
The launcher selects the own loopback service because the C++ runtime does not
implement this optional decoding profile. Legacy/default models stay greedy.
Use the included examples/silnia-validation-cases.json with a local Docker
Desktop Linux image; see docs/validated-generation.md. With no validator,
Apply remains disabled. The release GUI recording demonstrates the actual model
and review; it is not evidence of Docker Desktop or validated GUI Apply/Undo.
Those Windows end-to-end checks and upgrade remain unverified.
'''
    (root/'MODEL_CARD.md').write_text(card,encoding='utf-8');return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);print(json.dumps(qualify(p.parse_args().root),indent=2))
