import asyncio
import json
import unittest

from server_backend import MAX_REQUEST_BYTES, app


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
