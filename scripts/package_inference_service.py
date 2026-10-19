"""Create an inference-only service ZIP; never include keys, data or training weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'server_backend.py', 'model_backends.py', 'requirements-inference.txt',
    'scripts/start_nppai.py', 'scripts/check_dense_runtime_parity.py',
    'docs/notepad-setup-pl.md', 'docs/model-format.md', 'docs/validated-generation.md',
    'docs/automatic-plugin-mode.md', 'docs/research-model-card.md',
    'examples/silnia-validation-cases.json', 'examples/python-validation-cases.json',
    'docs/releases/v0.5.0-rc.1.md', 'license.txt',
)


def package(destination):
    payload = {name: (ROOT / name).read_bytes() for name in FILES}
    for name, raw in payload.items():
        if name.endswith('.py'):
            compile(raw, name, 'exec')
    payload['README.md'] = payload['docs/releases/v0.5.0-rc.1.md']
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
    package(parser.parse_args().destination)
