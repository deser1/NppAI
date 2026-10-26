"""Verify measured identities and publish an experimental model, never overwrite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.qualify_factorial_repair import qualify
from scripts.publish_release_candidate import gh,wait_for_ci
from scripts.factorial_repair import sha


def verify():
    summary=qualify(Path('measured'))
    report=json.loads(Path('plugin-evidence/report.json').read_bytes())
    raw=json.loads(Path('measured/report.json').read_bytes())
    if raw.get('source_commit')!=os.environ.get('GITHUB_SHA'):raise ValueError('Measured source commit mismatch')
    if any(sha(ROOT/name)!=digest for name,digest in raw.get('source_files',{}).items()) or not raw.get('source_files'):raise ValueError('Measured source files changed')
    expected=next(r for r in raw['readouts']['repair'] if r['id']=='plain-0')
    demo=report['live_generation_demo']
    if not expected.get('functional_pass') or demo['proposal']['code']!=expected['output']:raise ValueError('Actual plugin differs from Docker-tested model output')
    if report['model_sha256']!=summary['model_sha256'] or report['health']['model_id']!='sha256:'+summary['model_sha256']:raise ValueError('GUI model mismatch')
    if report['health'].get('prompt_bytes')!=256 or report['health'].get('decode_profile')!='python_function_beam8':raise ValueError('GUI recipe mismatch')
    if not all(demo.get(k) is True for k in ('completed','model_http_response_200','apply_disabled','review_uses_crlf')) or not demo['proposal']['syntax']['passed']:raise ValueError('Missing actual readable model review')
    if report['service_archive_sha256']!=sha('NppAI-factorial-service.zip') or demo['recording_sha256']!=sha('plugin-evidence/notepad-live-demo.gif'):raise ValueError('GUI asset mismatch')
    if report['package_sha256']!='faac23fa483cf9a80da4406ebc8a21a7ec55d27f7da8972e9980b3b1a2d09848':raise ValueError('Wrong published plugin')
    return summary


def run():
    summary=verify()
    if os.environ.get('GITHUB_EVENT_NAME')!='push' or os.environ.get('GITHUB_REF')!='refs/heads/main':raise ValueError('Publication is main-push only')
    repo,commit=os.environ['GITHUB_REPOSITORY'],os.environ['GITHUB_SHA']
    ci=wait_for_ci(repo,commit);tag='model-factorial-v1'
    if gh('release','view',tag,'--repo',repo,check=False).returncode==0:raise RuntimeError('Release exists; do not overwrite')
    if gh('api',f'repos/{repo}/git/ref/tags/{tag}',check=False).returncode==0:raise RuntimeError('Tag exists; do not repoint')
    out=Path('model-assets');out.mkdir()
    for name in ('NppAI-factorial.nppai','MODEL_CARD.md','qualified-report.json'):
        shutil.copyfile(Path('measured')/name,out/name)
    shutil.copyfile('NppAI-factorial-service.zip',out/'NppAI-factorial-service.zip')
    shutil.copyfile('plugin-evidence/notepad-live-demo.gif',out/'notepad-live-demo.gif')
    shutil.copyfile('plugin-evidence/report.json',out/'installation-report.json')
    with zipfile.ZipFile(out/'model-evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('measured','plugin-evidence'):
            for p in sorted(Path(folder).rglob('*')):
                if p.is_file():z.write(p,p.as_posix())
    checksum=out/'SHA256SUMS-model.txt'
    checksum.write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file()),encoding='utf-8')
    body=Path('model-release-notes.md')
    body.write_text((out/'MODEL_CARD.md').read_text(encoding='utf-8')+f'\nSource: `{commit}`\n\nExact main CI: {ci}\n\nActual Windows request/review recorded; output equals Docker-tested Linux candidate. Windows Docker Desktop + validated GUI Apply/Undo and installation upgrade remain unverified.\n',encoding='utf-8')
    gh('release','create',tag,'--repo',repo,'--target',commit,'--prerelease','--latest=false','--title','Experimental bounded factorial model repair','--notes-file',str(body),*[str(p) for p in sorted(out.iterdir())])
    print('Published',tag,json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:print(json.dumps(verify(),indent=2))
    else:run()
