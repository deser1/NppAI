from contextlib import redirect_stdout
import io
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from urllib import error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from scripts import start_nppai as launcher
import server_backend
from test_model_backends import checkpoint
from test_server_backend import call_app


class ReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, key='test-key'):
        return await call_app(b'', '0', extra_headers={'X-API-Key': key} if key else {},
                              path='/api/v1/health', method='GET')

    async def test_actual_nppai_load_auth_and_no_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.nppai'
            raw = checkpoint(path)
            with patch.dict(os.environ, {'NPPAI_API_KEY': 'test-key', 'NPPAI_INFERENCE_MODEL': str(path)}), \
                 patch.object(server_backend, 'inference_backend', None), \
                 patch.object(server_backend.DenseLocalBackend, 'generate', side_effect=AssertionError('readiness must not generate')):
                self.assertEqual((await self.call(None))[0], 401)
                self.assertEqual((await self.call('wrong'))[0], 403)
                status, body, _ = await self.call()
                self.assertEqual(status, 200)
                health = json.loads(body)
                self.assertEqual(health['backend'], 'nppai-dense-v2')
                self.assertEqual(health['context_bytes'], 128)
                self.assertEqual(health['model_id'], 'sha256:'+hashlib.sha256(raw).hexdigest())
                self.assertFalse(server_backend.inference_lock.locked())

    async def test_missing_incompatible_model_and_busy_are_explicit(self):
        with patch.dict(os.environ, {'NPPAI_API_KEY': 'test-key', 'NPPAI_INFERENCE_MODEL': ''}), \
             patch.object(server_backend, 'inference_backend', None):
            status, body, _ = await self.call()
            self.assertEqual(status, 503)
            self.assertIn(b'NPPAI_INFERENCE_MODEL', body)
            self.assertFalse(server_backend.inference_lock.locked())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'bad.nppai'
            path.write_bytes(b'unsupported')
            with patch.dict(os.environ, {'NPPAI_API_KEY': 'test-key', 'NPPAI_INFERENCE_MODEL': str(path)}), \
                 patch.object(server_backend, 'inference_backend', None):
                self.assertEqual((await self.call())[0], 503)
                self.assertFalse(server_backend.inference_lock.locked())
        with patch.dict(os.environ, {'NPPAI_API_KEY': 'test-key'}):
            server_backend.inference_lock.acquire()
            try:
                status, body, _ = await self.call()
                self.assertEqual(status, 503)
                self.assertIn(b'busy', body)
            finally:
                server_backend.inference_lock.release()


class LauncherTests(unittest.TestCase):
    def test_only_own_model_and_session_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'zażółć.nppai'
            checkpoint(path)
            original = {'UNCHANGED': 'value', 'NPPAI_API_KEY': 'old-key'}
            env = launcher.connection_environment(path, 8001, original)
            self.assertEqual(original['NPPAI_API_KEY'], 'old-key')
            self.assertEqual(env['NPPAI_INFERENCE_MODEL'], str(path.resolve()))
            self.assertEqual(env['NPPAI_PLUGIN_MODE'], 'auto')
            self.assertEqual(env['NPPAI_GENERATION_URL'], 'http://127.0.0.1:8001/api/v1/generate')
            self.assertGreaterEqual(len(env['NPPAI_API_KEY']), 32)
            bad = Path(temporary)/'model.pth'
            bad.touch()
            with self.assertRaises(ValueError): launcher.connection_environment(bad, 8000)

    def test_editor_starts_after_checked_model_with_identical_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)/'model.nppai'
            editor = Path(temporary)/'notepad++.exe'
            checkpoint(model)
            editor.touch()
            process = Mock()
            process.poll.return_value = 0
            health = {'model_id': 'sha256:'+'a'*64, 'context_bytes': 128}
            output = io.StringIO()
            with patch.object(launcher.subprocess, 'Popen', return_value=process) as popen, \
                 patch.object(launcher, 'wait_ready', return_value=health) as ready, \
                 patch.object(launcher.socket, 'socket') as socket, redirect_stdout(output):
                socket.return_value.__enter__.return_value.connect_ex.return_value = 1
                launcher.run(model, 8123, editor)
                self.assertEqual(popen.call_count, 2)
                server_env = popen.call_args_list[0].kwargs['env']
                editor_env = popen.call_args_list[1].kwargs['env']
                self.assertEqual(server_env, editor_env)
                self.assertEqual(popen.call_args_list[1].args[0], [str(editor.resolve()), '-multiInst'])
                self.assertEqual(ready.call_args.args, (process, 8123, server_env['NPPAI_API_KEY']))
                self.assertNotIn(server_env['NPPAI_API_KEY'], output.getvalue())
                self.assertIn('--host', popen.call_args_list[0].args[0])
                self.assertIn('127.0.0.1', popen.call_args_list[0].args[0])

    def test_readiness_uses_auth_loopback_no_proxy_and_no_generation(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                seen.append((self.path, self.headers.get('X-API-Key')))
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({'protocol_version': 1, 'status': 'ready',
                    'backend': 'nppai-dense-v2', 'model_id': 'sha256:'+'a'*64, 'context_bytes': 256}).encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {'http_proxy': 'http://127.0.0.1:1'}):
                self.assertEqual(launcher.readiness(server.server_port, 'private-key')['status'], 'ready')
            self.assertEqual(seen, [('/api/v1/health', 'private-key')])
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_slow_model_busy_retries_but_missing_model_fails(self):
        busy = error.HTTPError('loopback', 503, 'busy', {}, io.BytesIO(b'{"detail":"Inference is busy"}'))
        server = Mock(); server.poll.return_value = None
        with patch.object(launcher, 'readiness', side_effect=[busy, {'status': 'ready'}]), \
             patch.object(launcher.time, 'sleep'):
            self.assertEqual(launcher.wait_ready(server, 8000, 'key'), {'status': 'ready'})
        missing = error.HTTPError('loopback', 503, 'missing', {}, io.BytesIO(b'{"detail":"NppAI model unavailable"}'))
        with patch.object(launcher, 'readiness', side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, 'model unavailable'):
                launcher.wait_ready(server, 8000, 'key')
