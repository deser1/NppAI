"""Authored process-I/O fixture; never executes candidate code."""
import argparse
from pathlib import Path
import sys
import time
p = argparse.ArgumentParser()
for name in ('phase', 'source', 'cases', 'cancel-file', 'image'):
    p.add_argument('--' + name, required=True)
a = p.parse_args()
text = Path(a.source).read_text()
if text == 'wait':
    while not Path(a.cancel_file).exists():
        time.sleep(.02)
    print('cancelled', flush=True)
    sys.exit(3)
if text == 'flood':
    print('x' * 20000, flush=True)
else:
    print('fixture ' + a.phase + ': ' + text, flush=True)
sys.exit(1 if text == 'fail' else 0)
