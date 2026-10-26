"""Create an inference-only service ZIP; never include keys, data or training weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'python_function_decoding.py', 'server_backend.py', 'model_backends.py', 'requirements-inference.txt',
    'scripts/start_nppai.py', 'scripts/check_dense_runtime_parity.py',
    'docs/notepad-setup-pl.md', 'docs/model-format.md', 'docs/validated-generation.md',
    'docs/automatic-plugin-mode.md', 'docs/research-model-card.md',
    'examples/silnia-validation-cases.json', 'examples/python-validation-cases.json',
    'license.txt',
)


def package(destination, readme=None):
    payload = {name: (ROOT / name).read_bytes() for name in FILES}
    notes = json.loads((ROOT/'release/candidate.json').read_bytes())['notes']
    if Path(notes).parent != Path('docs/releases'):
        raise ValueError('Invalid packaged release notes path')
    payload[notes] = (ROOT/notes).read_bytes()
    for name, raw in payload.items():
        if name.endswith('.py'):
            compile(raw, name, 'exec')
    payload['README.md'] = Path(readme).read_bytes() if readme else payload[notes]
    payload['SOURCE_MANIFEST.json'] = (json.dumps({
        'purpose': 'Inference only; training endpoints have no trainer bundled.',
        'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in payload.items()}
    }, indent=2) + '\n').encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(payload.items()):
            archive.writestr(name, raw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--readme', type=Path)
    args = parser.parse_args()
    package(args.destination, args.readme)
