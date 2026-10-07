import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server_backend
from server_backend import MAX_REQUEST_BYTES, SubmitKnowledgeRequest, submit_knowledge, app


async def call_app(body: bytes, content_length: str | None, extra_headers=None, client_host="127.0.0.1"):
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    headers = [(b"content-type", b"application/json")]
    if content_length is not None:
        headers.append((b"content-length", content_length.encode("ascii")))
    if extra_headers:
        headers.extend(
            (name.lower().encode("ascii"), value.encode("ascii"))
            for name, value in extra_headers.items()
        )

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/submit_knowledge",
        "raw_path": b"/api/submit_knowledge",
        "query_string": b"",
        "headers": headers,
        "client": (client_host, 12345),
        "server": ("testserver", 80),
        "root_path": "",
    }
    await app(scope, receive, send)
    start = next(message for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    response_headers = {name.decode("latin1"): value.decode("latin1") for name, value in start.get("headers", [])}
    return start["status"], response_body, response_headers


class DatasetConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_samples_are_written_once_each(self):
        samples = 8
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.txt"
            payloads = [
                SubmitKnowledgeRequest(
                    prompt=f"prompt-{i}",
                    final_code=f"code-{i}",
                    thought_process=f"thought-{i}",
                    user_id=f"user-{i}",
                )
                for i in range(samples)
            ]

            class BackgroundTasksStub:
                def add_task(self, *args, **kwargs):
                    pass

            with patch.object(server_backend, "DATASET_PATH", dataset_path), \
                 patch.object(server_backend, "TRAINING_THRESHOLD", samples + 1):
                server_backend.new_samples_count = 0
                await asyncio.gather(
                    *[
                        submit_knowledge(payload, BackgroundTasksStub())
                        for payload in payloads
                    ]
                )

            content = dataset_path.read_text(encoding="utf-8")
            self.assertEqual(content.count("<|endoftext|>"), samples)
            for i in range(samples):
                self.assertEqual(content.count(f"prompt-{i}"), 1)
                self.assertEqual(content.count(f"code-{i}"), 1)
                self.assertEqual(content.count(f"thought-{i}"), 1)

    async def test_concurrent_samples_schedule_training_at_threshold(self):
        samples = 10
        threshold = 5

        class BackgroundTasksSpy:
            def __init__(self):
                self.tasks = []

            def add_task(self, func, *args, **kwargs):
                self.tasks.append((func, args, kwargs))

        background_tasks = BackgroundTasksSpy()
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.txt"
            payloads = [
                SubmitKnowledgeRequest(
                    prompt=f"threshold-prompt-{i}",
                    final_code=f"threshold-code-{i}",
                    user_id=f"threshold-user-{i}",
                )
                for i in range(samples)
            ]

            with patch.object(server_backend, "DATASET_PATH", dataset_path), \
                 patch.object(server_backend, "TRAINING_THRESHOLD", threshold):
                server_backend.new_samples_count = 0
                await asyncio.gather(
                    *[
                        submit_knowledge(payload, background_tasks)
                        for payload in payloads
                    ]
                )

            self.assertEqual(len(background_tasks.tasks), samples // threshold)
            self.assertEqual(server_backend.new_samples_count, samples % threshold)
            for func, args, kwargs in background_tasks.tasks:
                self.assertIs(func, server_backend.run_training_process)
                self.assertEqual(args, ())
                self.assertEqual(kwargs, {})


class ApiKeyAuthenticationTests(unittest.IsolatedAsyncioTestCase):
    def valid_body(self):
        return json.dumps({
            "prompt": "auth-test",
            "final_code": "print('ok')",
            "user_id": "test",
        }).encode("utf-8")

    async def test_requires_api_key_when_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict("os.environ", {server_backend.API_KEY_ENV: "secret"}, clear=False), \
             patch.object(server_backend, "DATASET_PATH", Path(tmpdir) / "dataset.txt"):
            status, body, _ = await call_app(self.valid_body(), None)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body), {"detail": "API key required"})

    async def test_rejects_invalid_api_key(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict("os.environ", {server_backend.API_KEY_ENV: "secret"}, clear=False), \
             patch.object(server_backend, "DATASET_PATH", Path(tmpdir) / "dataset.txt"):
            status, body, _ = await call_app(
                self.valid_body(),
                None,
                {"X-API-Key": "wrong"},
            )
        self.assertEqual(status, 403)
        self.assertEqual(json.loads(body), {"detail": "Invalid API key"})

    async def test_accepts_valid_api_key(self):
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict("os.environ", {server_backend.API_KEY_ENV: "secret"}, clear=False), \
             patch.object(server_backend, "DATASET_PATH", Path(tmpdir) / "dataset.txt"), \
             patch.object(server_backend, "TRAINING_THRESHOLD", 100):
            server_backend.new_samples_count = 0
            status, body, _ = await call_app(
                self.valid_body(),
                None,
                {"X-API-Key": "secret"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "success")


class RateLimitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server_backend.rate_limit_hits.clear()

    def valid_body(self):
        return json.dumps({
            "prompt": "rate-test",
            "final_code": "print('ok')",
        }).encode("utf-8")

    async def test_returns_429_and_retry_after_when_limit_exceeded(self):
        env = {
            server_backend.RATE_LIMIT_REQUESTS_ENV: "1",
            server_backend.RATE_LIMIT_WINDOW_ENV: "60",
            server_backend.API_KEY_ENV: "rate-test-secret",
        }
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict("os.environ", env, clear=False), \
             patch.object(server_backend, "DATASET_PATH", Path(tmpdir) / "dataset.txt"), \
             patch.object(server_backend, "TRAINING_THRESHOLD", 100):
            server_backend.new_samples_count = 0
            first, _, _ = await call_app(self.valid_body(), None, {"X-API-Key": "rate-test-secret"})
            second, body, headers = await call_app(self.valid_body(), None, {"X-API-Key": "rate-test-secret"})

        self.assertEqual(first, 200)
        self.assertEqual(second, 429)
        self.assertEqual(json.loads(body), {"detail": "Rate limit exceeded"})
        self.assertIn("retry-after", headers)
        self.assertGreaterEqual(int(headers["retry-after"]), 1)

    async def test_allows_request_after_window_expires(self):
        env = {
            server_backend.RATE_LIMIT_REQUESTS_ENV: "1",
            server_backend.RATE_LIMIT_WINDOW_ENV: "10",
            server_backend.API_KEY_ENV: "rate-test-secret",
        }
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict("os.environ", env, clear=False), \
             patch.object(server_backend, "DATASET_PATH", Path(tmpdir) / "dataset.txt"), \
             patch.object(server_backend, "TRAINING_THRESHOLD", 100), \
             patch.object(server_backend, "monotonic_time", side_effect=[100.0, 111.0]):
            server_backend.new_samples_count = 0
            first, _, _ = await call_app(self.valid_body(), None, {"X-API-Key": "rate-test-secret"})
            second, _, _ = await call_app(self.valid_body(), None, {"X-API-Key": "rate-test-secret"})

        self.assertEqual(first, 200)
        self.assertEqual(second, 200)

    async def test_tracks_clients_independently(self):
        env = {
            server_backend.RATE_LIMIT_REQUESTS_ENV: "1",
            server_backend.RATE_LIMIT_WINDOW_ENV: "60",
            server_backend.API_KEY_ENV: "rate-test-secret",
        }
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.dict("os.environ", env, clear=False), \
             patch.object(server_backend, "DATASET_PATH", Path(tmpdir) / "dataset.txt"), \
             patch.object(server_backend, "TRAINING_THRESHOLD", 100):
            server_backend.new_samples_count = 0
            first, _, _ = await call_app(self.valid_body(), None, {"X-API-Key": "rate-test-secret"}, client_host="10.0.0.1")
            second, _, _ = await call_app(self.valid_body(), None, {"X-API-Key": "rate-test-secret"}, client_host="10.0.0.2")

        self.assertEqual(first, 200)
        self.assertEqual(second, 200)


class RequestSizeLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_advertised_oversized_body(self):
        status, body, _ = await call_app(b"{}", str(MAX_REQUEST_BYTES + 1))
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(body), {"detail": "Request body too large"})

    async def test_rejects_invalid_content_length(self):
        status, body, _ = await call_app(b"{}", "not-a-number")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"detail": "Invalid Content-Length"})

    async def test_rejects_actual_oversized_body_without_content_length(self):
        oversized = b"x" * (MAX_REQUEST_BYTES + 1)
        status, body, _ = await call_app(oversized, None)
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(body), {"detail": "Request body too large"})


if __name__ == "__main__":
    unittest.main()
