"""Start a checked NppAI-only loopback server and optionally launch Notepad++."""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def connection_environment(model, port, existing=None):
    path = Path(model).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != '.nppai':
        raise ValueError('Wskaz wyeksportowany model NppAI .nppai (nie .pth ani archiwum).')
    if not 1 <= port <= 65535:
        raise ValueError('Port musi nalezec do zakresu 1..65535.')
    environment = dict(os.environ if existing is None else existing)
    # One private session key is shared only by the server and child editor.
    environment.update(NPPAI_INFERENCE_MODEL=str(path), NPPAI_PLUGIN_MODE='auto',
                       NPPAI_ALLOW_REMOTE='1', NPPAI_API_KEY=secrets.token_urlsafe(32),
                       NPPAI_GENERATION_URL=f'http://127.0.0.1:{port}/api/v1/generate')
    return environment


def readiness(port, key):
    opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
    req = request.Request(f'http://127.0.0.1:{port}/api/v1/health', headers={'X-API-Key': key})
    with opener.open(req, timeout=2) as response:
        raw = response.read(4097)
    if len(raw) > 4096:
        raise ValueError('Nieprawidlowa odpowiedz gotowosci NppAI.')
    result = json.loads(raw)
    if (result.get('status') != 'ready' or result.get('backend') != 'nppai-dense-v2' or
            result.get('protocol_version') != 1 or not str(result.get('model_id', '')).startswith('sha256:') or
            type(result.get('context_bytes')) is not int or result['context_bytes'] <= 0):
        raise ValueError('Serwer nie potwierdzil zaladowania modelu NppAI.')
    return result


def wait_ready(server, port, key, timeout=60):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError('Serwer NppAI zakonczyl prace. Sprawdz komunikat w tej konsoli i zainstalowane zaleznosci.')
        try:
            return readiness(port, key)
        except error.HTTPError as exc:
            # A reachable server with a failed readiness check is actionable;
            # repeating a missing/corrupt model request cannot repair the file.
            detail = exc.read(4096).decode('utf-8', errors='replace')
            if exc.code == 503 and json.loads(detail).get('detail') == 'Inference is busy':
                time.sleep(1)
                continue
            raise RuntimeError(f'NppAI readiness HTTP {exc.code}: {detail}') from None
        except (error.URLError, TimeoutError, ConnectionError):
            time.sleep(1)
    raise RuntimeError('Uplynal czas startu NppAI. Sprawdz plik modelu i komunikaty serwera.')


def run(model, port=8000, notepad=None):
    environment = connection_environment(model, port)
    editor = Path(notepad).expanduser().resolve(strict=True) if notepad else None
    if editor and not editor.is_file():
        raise ValueError('Nie znaleziono pliku notepad++.exe.')
    # Avoid attaching to an unrelated process already using this port.
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError('Port jest zajety. Zakoncz poprzedni serwer albo podaj --port, np. 8001.')
    server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'server_backend:app',
                               '--host', '127.0.0.1', '--port', str(port)], cwd=ROOT, env=environment)
    try:
        health = wait_ready(server, port, environment['NPPAI_API_KEY'])
        print('NppAI gotowy:', health['model_id'], flush=True)
        print('Kontekst modelu (bajty):', health['context_bytes'], flush=True)
        print('Serwer:', environment['NPPAI_GENERATION_URL'], flush=True)
        if editor:
            # A fresh instance inherits this session's URL/key, unlike an
            # already running Notepad++ launched from the Start menu.
            subprocess.Popen([str(editor), '-multiInst'], env=environment)
            print('Uruchomiono Notepad++ z konfiguracja NppAI. Pozostaw konsole otwarta.', flush=True)
        else:
            print('Aby uruchomic wtyczke z ta konfiguracja, uzyj parametru --notepad SCIEZKA_DO_EXE.', flush=True)
        server.wait()
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Wyeksportowany autorski model NppAI .nppai')
    parser.add_argument('--notepad', type=Path, help='Sciezka do notepad++.exe; startuje nowa instancje')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    try:
        run(args.model, args.port, args.notepad)
    except KeyboardInterrupt:
        print('\nSerwer NppAI zatrzymany.')
    except (OSError, ValueError, RuntimeError) as exc:
        print('NppAI: '+str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
