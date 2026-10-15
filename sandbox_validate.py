"""Local Docker validation driver. Never imports/executes candidate code on host.

Exit codes: 0 validated, 1 candidate failed, 2 unavailable/configuration error,
3 cancelled. stdout is bounded public feedback, not a machine success receipt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time

# The candidate runs in a child, while the container supervisor keeps expected
# answers and the host receipt out of that child's input and environment.
WORKER = r'''
import json, sys
p = json.load(sys.stdin)
namespace = {"__name__": "__nppai_candidate__"}
exec(compile(p["code"], "candidate.py", "exec"), namespace)
f = namespace[p["function"]]
actual = f(*p["args"], **p["kwargs"])
print(json.dumps({"actual": actual}, allow_nan=False, sort_keys=True))
'''
SUPERVISOR = r'''
import json, subprocess, sys, traceback
p = json.load(sys.stdin)
try:
    compile(p["code"], "candidate.py", "exec")
    if p["phase"] == "tests":
        for index, case in enumerate(p["cases"]):
            value = {"code": p["code"], "function": case["function"], "args": case["args"], "kwargs": case["kwargs"]}
            child = subprocess.run([sys.executable, "-I", "-B", "-c", p["worker"]],
                input=json.dumps(value), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
            if child.returncode:
                raise AssertionError("case %d: candidate failed: %s" % (index + 1, child.stderr[-2048:]))
            actual = json.loads(child.stdout)["actual"]
            if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(case["expected"], sort_keys=True, allow_nan=False):
                raise AssertionError("case %d: expected %r, got %r" % (index + 1, case["expected"], actual))
    print("NPPAI:" + p["receipt"] + ":" + p["phase"] + ":" + str(len(p["cases"])))
except BaseException:
    traceback.print_exc(limit=3)
    sys.exit(1)
'''


def load_cases(path):
    raw = Path(path).read_bytes()
    if len(raw) > 65536:
        raise ValueError('Test cases exceed 64 KiB')
    def invalid(value):
        raise ValueError('Non-finite JSON values are not supported')
    cases = json.loads(raw, parse_constant=invalid)
    if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
        raise ValueError('Configure 1..100 authored test cases')
    for case in cases:
        if not isinstance(case, dict) or set(case) != {'function', 'args', 'kwargs', 'expected'}:
            raise ValueError('Each case needs function, args, kwargs, expected')
        if not isinstance(case['function'], str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', case['function']):
            raise ValueError('Test function must be a plain identifier')
        if not isinstance(case['args'], list) or not isinstance(case['kwargs'], dict):
            raise ValueError('Invalid function arguments')
    return cases, hashlib.sha256(raw).hexdigest()


def docker_command(docker, host, image, name):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('NPPAI_VALIDATION_IMAGE must be a local immutable sha256 image ID')
    if not (host.startswith('unix:///') or host.startswith('npipe:////./pipe/')):
        raise ValueError('Validation requires a local Unix socket or Windows named pipe')
    return [docker, '--host', host, 'run', '--rm', '-i', '--name', name,
        '--pull=never', '--network=none', '--read-only', '--cap-drop=ALL',
        '--security-opt=no-new-privileges', '--pids-limit=32', '--cpus=1',
        '--memory=128m', '--memory-swap=128m', '--user=65534:65534',
        '--ipc=none', '--log-driver=none', '--no-healthcheck',
        '--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m', '--workdir=/tmp',
        '--entrypoint=python', image, '-I', '-B', '-c', SUPERVISOR]


def validate(source, phase, cases, image, cancel_file=None, timeout=20):
    if phase not in ('syntax', 'tests') or not source or len(source.encode('utf-8')) > 131072 or '\0' in source:
        raise ValueError('Unsupported phase or candidate size')
    docker = os.getenv('NPPAI_DOCKER', 'docker')
    host = os.getenv('NPPAI_DOCKER_HOST', 'npipe:////./pipe/docker_engine' if os.name == 'nt' else 'unix:///var/run/docker.sock')
    name = 'nppai-validate-' + secrets.token_hex(12)
    receipt = secrets.token_hex(32)
    command = docker_command(docker, host, image, name)
    # No host API keys, proxy settings or Docker context are inherited by CLI.
    env = {k: v for k, v in os.environ.items() if k.upper() in
           ('PATH', 'SYSTEMROOT', 'WINDIR', 'HOME', 'USERPROFILE', 'TEMP', 'TMP')}
    payload = json.dumps(dict(code=source, phase=phase, cases=cases, worker=WORKER, receipt=receipt)).encode()
    output = bytearray()
    overflow = threading.Event()
    process = None
    def read(stream):
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            if len(output) + len(chunk) <= 16384:
                output.extend(chunk)
            else:
                overflow.set()
    def write(stream):
        try:
            stream.write(payload)
            stream.flush()
        except (OSError, BrokenPipeError):
            pass
        finally:
            stream.close()
    try:
        if cancel_file and Path(cancel_file).exists():
            return 3, 'Validation cancelled'
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=env)
        reader = threading.Thread(target=read, args=(process.stdout,), daemon=True)
        writer = threading.Thread(target=write, args=(process.stdin,), daemon=True)
        reader.start(); writer.start()
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if cancel_file and Path(cancel_file).exists():
                return 3, 'Validation cancelled'
            if overflow.is_set():
                return 1, 'Candidate exceeded diagnostic output limit'
            if time.monotonic() >= deadline:
                return 1, 'Candidate validation timed out'
            time.sleep(.05)
        reader.join(2); writer.join(2)
        text = output.decode('utf-8', errors='replace')
        proof = f'NPPAI:{receipt}:{phase}:{len(cases)}'
        if process.returncode == 0 and text.strip() == proof and not overflow.is_set():
            return 0, f'{phase} passed ({len(cases)} authored cases configured)'
        # Docker infrastructure errors must never trigger model repairs.
        status = 2 if process.returncode in (125, 126, 127) else 1
        return status, (text.replace(receipt, '[receipt]')[-4096:] or 'Validation did not produce a verified result')
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            try:
                subprocess.run([docker, '--host', host, 'rm', '-f', name], env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('syntax', 'tests'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--cancel-file', type=Path, required=True)
    parser.add_argument("--image", default=os.getenv("NPPAI_VALIDATION_IMAGE", ""))
    args = parser.parse_args()
    try:
        if args.source.stat().st_size > 131072:
            raise ValueError('Candidate exceeds 128 KiB')
        source = args.source.read_text(encoding='utf-8')
        cases, digest = load_cases(args.cases)
        status, feedback = validate(source, args.phase, cases, args.image, args.cancel_file)
        print(f'{feedback[:4096]}\nTest suite SHA-256: {digest}', flush=True)
        return status
    except (OSError, ValueError) as error:
        print(f'Validation unavailable: {error}', flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
