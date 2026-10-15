import asyncio
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from model_backends import (BackendRouter, BackendUnavailable, DenseLocalBackend,
                            GenerationRequest, GenerationResponse, RemoteBackend)
import server_backend
from test_server_backend import call_app


def checkpoint(path, token=65):
    output = np.zeros((2, 256), dtype='<f4')
    if token:
        output[:, token] = 2
    arrays = [np.ones((256, 2)), np.zeros((128, 2)), np.ones(2)]
    arrays.extend(np.zeros((2, 2)) for _ in range(4))
    arrays.append(np.ones(2))
    arrays.extend(np.zeros((2, 2)) for _ in range(3))
    arrays.extend([np.ones(2), output])
    payload = b''.join(a.astype('<f4').tobytes() for a in arrays)
    raw = (b'NPPAI\0\0\0' + struct.pack('<I5iQ', 2, 2, 2, 1, 128, 256, len(payload))
           + hashlib.sha256(payload).digest() + payload)
    path.write_bytes(raw)
    return raw


class RuntimeTests(unittest.TestCase):
    def test_cli_defaults_to_local_and_reports_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.nppai'
            checkpoint(path)
            command = [sys.executable, 'model_backends.py', '--model', str(path),
                       '--prompt', 'task', '--max-new-tokens', '3']
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)['output'], 'AAA')
            result = subprocess.run([sys.executable, 'model_backends.py', '--mode', 'online',
                '--endpoint', 'https://example.org/api/v1/generate', '--prompt', 'task'],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('explicitly enabled', result.stderr)

    def test_real_checkpoint_generates_completion_only_and_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.nppai'
            raw = checkpoint(path)
            model = DenseLocalBackend(str(path))
            result = model.generate(GenerationRequest(prompt='hello', max_new_tokens=3))
            self.assertEqual(result.output, 'AAA')
            self.assertEqual(result.generated_tokens, 3)
            self.assertEqual(result.finish_reason, 'length')
            self.assertEqual(result.model_id, 'sha256:' + hashlib.sha256(raw).hexdigest())
            with self.assertRaises(ValueError):
                model.generate(GenerationRequest(prompt='x' * 128))
            checkpoint(path, token=0)
            result = DenseLocalBackend(str(path)).generate(GenerationRequest(prompt='hello'))
            self.assertEqual((result.output, result.generated_tokens, result.finish_reason), ('', 1, 'eos'))

    def test_assistant_prefix_is_included_once_and_counted_in_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.nppai'
            checkpoint(path)
            model = DenseLocalBackend(str(path))
            result = model.generate(GenerationRequest(prompt='task', assistant_prefix='def f():\n    ', max_new_tokens=3))
            self.assertEqual(result.output, 'def f():\n    AAA')
            self.assertEqual(result.generated_tokens, 3)
            with self.assertRaises(ValueError):
                model.generate(GenerationRequest(prompt='task', assistant_prefix='x' * 128))

    def test_bad_checkpoints_are_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.nppai'
            for raw in (b'', b'NPPAI\0\0\0', b'bad' * 100):
                path.write_bytes(raw)
                with self.assertRaises(BackendUnavailable):
                    DenseLocalBackend(str(path))
            raw = checkpoint(path)
            path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with self.assertRaises(BackendUnavailable):
                DenseLocalBackend(str(path))

    def test_contract_rejects_version_limits_and_unknown_fields(self):
        for values in ({'protocol_version': 2}, {'max_new_tokens': 513},
                       {'max_new_tokens': True}, {'execute': True}, {'prompt': ''}):
            with self.assertRaises(ValidationError):
                GenerationRequest.model_validate({'prompt': 'task', **values})


class RouterTests(unittest.TestCase):
    def test_local_never_calls_remote_and_hybrid_falls_back_only_on_unavailable(self):
        calls = []
        class Backend:
            def __init__(self, name, failure=None):
                self.name, self.failure = name, failure
            def generate(self, payload):
                calls.append(self.name)
                if self.failure:
                    raise self.failure
                return GenerationResponse(model_id=self.name, output='ok', generated_tokens=2, finish_reason='length')
        request = GenerationRequest(prompt='private task')
        remote = Backend('remote')
        BackendRouter('local', local=Backend('local'), remote=remote).generate(request)
        self.assertEqual(calls, ['local'])
        for mode in ('online', 'hybrid'):
            with self.assertRaises(ValueError):
                BackendRouter(mode, local=Backend('local'), remote=remote)
        calls.clear()
        router = BackendRouter('hybrid', local=Backend('local', BackendUnavailable()),
                               remote=remote, allow_remote=True)
        self.assertEqual(router.generate(request).model_id, 'remote')
        self.assertEqual(calls, ['local', 'remote'])
        calls.clear()
        router.local = Backend('local', ValueError('context too long'))
        with self.assertRaises(ValueError):
            router.generate(request)
        self.assertEqual(calls, ['local'])
        calls.clear()
        BackendRouter('online', remote=remote, allow_remote=True).generate(request)
        self.assertEqual(calls, ['remote'])


class TransportTests(unittest.TestCase):
    def test_explicit_transfer_and_https_rules(self):
        with self.assertRaises(ValueError):
            RemoteBackend('https://example.org/api/v1/generate', 'key')
        for endpoint in ('http://example.org/api/v1/generate',
                         'https://user:pass@example.org/api/v1/generate',
                         'https://example.org/api/v1/generate?key=secret',
                         'https://example.org/other'):
            with self.assertRaises(ValueError):
                RemoteBackend(endpoint, 'key', allow_remote=True)

    def test_http_transport_auth_contract_redirect_and_failure_classification(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            status, response = 200, {'model_id': 'test', 'output': 'ok', 'generated_tokens': 2, 'finish_reason': 'length'}
            def log_message(self, *args): pass
            def do_POST(self):
                seen.append((self.path, self.headers.get('X-API-Key'),
                             json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(self.status)
                if self.status == 302:
                    self.send_header('Location', '/stolen')
                self.end_headers()
                self.wfile.write(json.dumps(self.response).encode())
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            backend = RemoteBackend(f'http://127.0.0.1:{httpd.server_port}/api/v1/generate',
                                    'test-key', allow_remote=True)
            payload = GenerationRequest(prompt='task', context='context', max_new_tokens=2)
            self.assertEqual(backend.generate(payload).output, 'ok')
            self.assertEqual(seen[0], ('/api/v1/generate', 'test-key', payload.model_dump()))
            for status, exception in ((503, BackendUnavailable), (403, ValueError), (302, ValueError)):
                Handler.status = status
                with self.assertRaises(exception):
                    backend.generate(payload)
            self.assertEqual(len(seen), 4)  # Redirect was not followed.
            Handler.status = 200
            Handler.response['generated_tokens'] = 3
            with self.assertRaises(ValueError):
                backend.generate(payload)
            Handler.response = {'protocol_version': 2}
            with self.assertRaises(ValidationError):
                backend.generate(payload)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join()


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.environment = patch.dict(os.environ, {'NPPAI_API_KEY': 'test-key'})
        self.environment.start()
        server_backend.rate_limit_hits.clear()

    async def asyncTearDown(self):
        self.environment.stop()
        server_backend.rate_limit_hits.clear()

    async def call(self, payload=None, key='test-key'):
        body = json.dumps(payload or {'prompt': 'task', 'max_new_tokens': 3}).encode()
        return await call_app(body, str(len(body)),
            extra_headers={'X-API-Key': key} if key else {}, path='/api/v1/generate')

    async def test_auth_validation_and_no_config(self):
        self.assertEqual((await self.call(key=None))[0], 401)
        self.assertEqual((await self.call(key='wrong'))[0], 403)
        self.assertEqual((await self.call({'prompt': 'task', 'protocol_version': 2}))[0], 422)
        with patch.object(server_backend, 'inference_backend', None), \
             patch.dict(os.environ, {'NPPAI_INFERENCE_MODEL': ''}):
            self.assertEqual((await self.call())[0], 503)
            self.assertFalse(server_backend.inference_lock.locked())

    async def test_real_runtime_same_as_local_and_no_dataset_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.nppai'
            checkpoint(path)
            model = DenseLocalBackend(str(path))
            dataset = Path(directory) / 'dataset.txt'
            with patch.object(server_backend, 'inference_backend', model), \
                 patch.object(server_backend, 'DATASET_PATH', dataset):
                status, body, _ = await self.call()
                self.assertEqual(status, 200)
                expected = model.generate(GenerationRequest(prompt='task', max_new_tokens=3))
                self.assertEqual(json.loads(body), expected.model_dump())
                self.assertFalse(dataset.exists())
                self.assertEqual((await self.call({'prompt': 'x' * 128}))[0], 422)

    async def test_busy_and_rate_limit(self):
        server_backend.inference_lock.acquire()
        try:
            self.assertEqual((await self.call())[0], 503)
        finally:
            server_backend.inference_lock.release()
        server_backend.rate_limit_hits.clear()
        with patch.dict(os.environ, {'NPPAI_RATE_LIMIT_REQUESTS': '1'}):
            await self.call()
            self.assertEqual((await self.call())[0], 429)

    async def test_lazy_loading_pins_model_and_internal_errors_are_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.nppai'
            checkpoint(path)
            with patch.object(server_backend, 'inference_backend', None), \
                 patch.dict(os.environ, {'NPPAI_INFERENCE_MODEL': str(path)}):
                status, body, _ = await self.call()
                self.assertEqual(status, 200)
                original = json.loads(body)
                checkpoint(path, token=66)
                self.assertEqual(json.loads((await self.call())[1]), original)
        class BrokenBackend:
            def generate(self, payload):
                raise RuntimeError('private filesystem path or prompt')
        with patch.object(server_backend, 'inference_backend', BrokenBackend()):
            status, body, _ = await self.call()
            self.assertEqual(status, 500)
            self.assertNotIn(b'private', body)
            self.assertFalse(server_backend.inference_lock.locked())

    async def test_cancellation_does_not_release_running_worker(self):
        started, finish = threading.Event(), threading.Event()
        class BlockingBackend:
            def generate(self, payload):
                started.set()
                finish.wait(5)
                return GenerationResponse(model_id='fixture', output='', generated_tokens=1, finish_reason='eos')
        with patch.object(server_backend, 'inference_backend', BlockingBackend()):
            task = asyncio.create_task(server_backend.generate_code(GenerationRequest(prompt='task')))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 3))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(server_backend.inference_lock.locked())
                self.assertEqual((await self.call())[0], 503)
            finally:
                finish.set()
                for _ in range(100):
                    if not server_backend.inference_lock.locked():
                        break
                    await asyncio.sleep(.01)
                self.assertFalse(server_backend.inference_lock.locked())


if __name__ == '__main__':
    unittest.main()
