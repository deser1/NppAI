import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server_backend
from server_backend import MAX_REQUEST_BYTES, SubmitKnowledgeRequest, submit_knowledge, app


async def call_app(body: bytes, content_length: str | None):
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
        "client": ("127.0.0.1", 12345),
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
    return start["status"], response_body


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


class RequestSizeLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_advertised_oversized_body(self):
        status, body = await call_app(b"{}", str(MAX_REQUEST_BYTES + 1))
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(body), {"detail": "Request body too large"})

    async def test_rejects_invalid_content_length(self):
        status, body = await call_app(b"{}", "not-a-number")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"detail": "Invalid Content-Length"})

    async def test_rejects_actual_oversized_body_without_content_length(self):
        oversized = b"x" * (MAX_REQUEST_BYTES + 1)
        status, body = await call_app(oversized, None)
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(body), {"detail": "Request body too large"})


if __name__ == "__main__":
    unittest.main()
