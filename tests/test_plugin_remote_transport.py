"""Exercise the plugin's actual WinHTTP client against an authored loopback server."""
import json
import ctypes
import hashlib
import os
from pathlib import Path
import struct
import tempfile
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main(executable, dll_path):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        status, raw = 200, None
        def log_message(self, *args): pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.path, self.headers.get('X-API-Key'), payload))
            self.send_response(self.status)
            self.send_header('Location', '/forbidden-redirect')
            self.end_headers()
            self.wfile.write(self.raw if self.raw is not None else json.dumps({
                'protocol_version': 1, 'model_id': 'fixture', 'output': payload['assistant_prefix'] + 'ok',
                'generated_tokens': 3, 'finish_reason': 'length'}).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/api/v1/generate'
    def run(endpoint=url):
        return subprocess.run([executable, '--http', endpoint], capture_output=True,
                              text=True, encoding='utf-8', timeout=45)
    try:
        result = run()
        assert result.returncode == 0, result.stderr
        assert result.stdout == 'def f(x):\n    ok', result.stdout
        assert seen[0] == ('/api/v1/generate', 'test-key', {
            'protocol_version': 1, 'prompt': 'task', 'context': 'context',
            'assistant_prefix': 'def f(x):\n    ', 'max_new_tokens': 3}), seen
        for status in (302, 403, 422, 503):
            Handler.status = status
            assert run().returncode != 0
        assert len(seen) == 5, 'redirect was followed'
        Handler.status, Handler.raw = 200, b'x' * 16385
        assert run().returncode != 0
        Handler.raw = b'{"protocol_version":2}'
        assert run().returncode != 0
        for endpoint in (url + '?secret=value', url + '#fragment',
                         url.replace('/api/v1/generate', '/wrong'),
                         'http://example.org/api/v1/generate'):
            assert run(endpoint).returncode != 0
        Handler.raw = None
        # Call actual AIManager inside the test DLL, without an editor process.
        plugin = ctypes.CDLL(dll_path)
        generate = plugin.generateForTesting
        generate.argtypes, generate.restype = [ctypes.c_char_p, ctypes.c_char_p], ctypes.c_char_p
        load = plugin.loadModelForTesting
        load.argtypes, load.restype = [ctypes.c_char_p], None
        os.environ.update(NPPAI_PLUGIN_MODE='auto', NPPAI_ALLOW_REMOTE='0',
                          NPPAI_GENERATION_URL=url, NPPAI_API_KEY='test-key')
        before = len(seen)
        assert generate(b'task', b'').startswith(b'ERROR:')
        assert len(seen) == before, 'remote called without opt-in'
        os.environ['NPPAI_ALLOW_REMOTE'] = '1'
        assert generate(b'task', b'') == b'ok'
        assert len(seen) == before + 1, 'missing local model did not route online'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.nppai'
            def checkpoint(token):
                # dim=2, hidden=2, layers=1, context=128, byte vocabulary.
                values = [1.] * 512 + [0.] * 256 + [1.] * 2
                values += [0.] * 16 + [1.] * 2 + [0.] * 12 + [1.] * 2
                output = [0.] * 512
                if token:
                    output[token] = output[256 + token] = 2.
                values += output
                payload = struct.pack('<' + 'f' * len(values), *values)
                path.write_bytes(b'NPPAI\0\0\0' + struct.pack('<I5iQ', 2, 2, 2, 1, 128, 256, len(payload))
                                 + hashlib.sha256(payload).digest() + payload)
                load(str(path).encode())
            checkpoint(65)
            before = len(seen)
            output = generate(b'task', b'')
            assert output == b'A' * 512, output
            assert len(seen) == before, 'fitting local request sent online'
            assert generate(b'x' * 128, b'') == b'ok'
            assert len(seen) == before + 1, 'context overflow not routed online'
            checkpoint(0)
            before = len(seen)
            assert generate(b'task', b'') == b'ok'
            assert len(seen) == before + 1, 'empty local completion not retried exactly once'
            os.environ['NPPAI_PLUGIN_MODE'] = 'local'
            before = len(seen)
            assert generate(b'task', b'').startswith(b'ERROR:')
            assert len(seen) == before, 'forced local mode sent data online'
            os.environ['NPPAI_PLUGIN_MODE'] = 'online'
            assert generate(b'@python-signature f(x)\nReturn x.', b'') == b'def f(x):\n    ok'
            assert seen[-1][2]['prompt'] == 'Implement f(x). Return x.'
            assert seen[-1][2]['assistant_prefix'] == 'def f(x):\n    '
            os.environ['NPPAI_PLUGIN_MODE'] = 'typo'
            before = len(seen)
            assert generate(b'task', b'').startswith(b'ERROR:')
            assert len(seen) == before
            os.environ['NPPAI_PLUGIN_MODE'] = 'auto'
            plugin.stopGenerationForTesting()
            before = len(seen)
            assert b'cancelled' in generate(b'task', b'')
            assert len(seen) == before, 'cancelled preparation sent a request'
            plugin.beginGenerationForTesting()
            assert generate(b'task', b'') == b'ok'
        print('Native WinHTTP and actual plugin automatic selection tests passed')
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
