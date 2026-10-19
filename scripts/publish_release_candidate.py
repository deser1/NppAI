"""Publish the checked prerelease on main only, after exact-commit CI succeeds."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def gh(*args, check=True):
    return subprocess.run(['gh', *args], check=check, capture_output=True, text=True)


def wait_for_ci(repository, sha, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = json.loads(gh('api', f'repos/{repository}/actions/runs?head_sha={sha}&event=push&per_page=100').stdout)
        runs = [r for r in result['workflow_runs'] if r['path'] == '.github/workflows/CI_build.yml'
                and r['head_sha'] == sha and r['head_branch'] == 'main']
        if runs:
            latest = max(runs, key=lambda r: r['id'])
            if latest['status'] == 'completed':
                if latest['conclusion'] != 'success':
                    raise RuntimeError('Exact main commit CI_build did not pass')
                return latest['html_url']
        time.sleep(15)
    raise TimeoutError('No successful exact-commit main CI_build within 15 minutes')


def verify_installation(assets):
    report = json.loads((assets/'installation-report.json').read_bytes())
    for field in ('service_from_extracted_package', 'polish_prompt_round_trip', 'apply_disabled_without_validation'):
        if report.get(field) is not True:
            raise ValueError('Missing installation evidence: '+field)
    for field, name in (('package_sha256', 'NppAI-x64.zip'),
                        ('service_archive_sha256', 'NppAI-service.zip'),
                        ('model_sha256', 'NppAI-research-staged-mix.nppai')):
        if report.get(field) != hashlib.sha256((assets/name).read_bytes()).hexdigest():
            raise ValueError('Installation report does not match release asset: '+name)
    if report['health'].get('model_id') != 'sha256:'+report['model_sha256'] or report['health'].get('status') != 'ready':
        raise ValueError('Readiness does not match the supplied checkpoint')


def publish():
    if os.environ.get('GITHUB_EVENT_NAME') != 'push' or os.environ.get('GITHUB_REF') != 'refs/heads/main':
        raise RuntimeError('Candidate publication requires a main push')
    repository, sha = os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_SHA']
    candidate = json.loads(Path('release/candidate.json').read_bytes())
    tag = candidate['tag']
    if not re.fullmatch(r'v\d+\.\d+\.\d+-rc\.\d+', tag):
        raise ValueError('Only explicit release-candidate tags are accepted')
    notes = Path(candidate['notes'])
    if notes.parent != Path('docs/releases') or not notes.is_file():
        raise ValueError('Invalid release notes path')
    ci = wait_for_ci(repository, sha)
    if gh('release', 'view', tag, '--repo', repository, check=False).returncode == 0:
        raise RuntimeError('Candidate already exists; never replace published assets')
    ref = gh('api', f'repos/{repository}/git/ref/tags/{tag}', check=False)
    if ref.returncode == 0:
        target = json.loads(ref.stdout)['object']
        if target.get('type') != 'commit' or target.get('sha') != sha:
            raise RuntimeError('An existing tag is not the exact checked commit')
    assets = sorted(p for p in Path('candidate-assets').iterdir() if p.is_file())
    expected = {'NppAI-x64.zip', 'NppAI-x86.zip', 'NppAI-arm64.zip', 'NppAI-service.zip',
                'NppAI-research-staged-mix.nppai', 'MODEL_CARD.md', 'installation-report.json'}
    if {p.name for p in assets} != expected:
        raise ValueError('Candidate asset set differs from checked inputs')
    verify_installation(Path('candidate-assets'))
    model = Path('candidate-assets/NppAI-research-staged-mix.nppai')
    if hashlib.sha256(model.read_bytes()).hexdigest() != '4859ad69e93ff2ca80d8951bca617476d118f8ef127abc4b313b8d2f0671c6fd':
        raise ValueError('Research checkpoint changed')
    checksum = Path('candidate-assets/SHA256SUMS.txt')
    checksum.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in assets), encoding='utf-8')
    body = Path('candidate-notes.md')
    body.write_text(notes.read_text(encoding='utf-8') + f'\nSource commit: `{sha}`\n\nVerified main CI: {ci}\n', encoding='utf-8')
    gh('release', 'create', tag, '--repo', repository, '--target', sha,
       '--prerelease', '--latest=false', '--title', candidate['name'], '--notes-file', str(body),
       *(str(p) for p in assets), str(checksum))
    print(f'Published experimental candidate {tag} from {sha}')


if __name__ == '__main__':
    publish()
