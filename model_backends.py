"""Shared, versioned generation contract for the standalone NppAI agent client.

Generation returns data only: it never executes a completion or modifies files.
"""
import hashlib
import ipaddress
from pathlib import Path
from typing import Literal, Protocol
from urllib import error, request
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    protocol_version: Literal[1] = 1
    prompt: str = Field(min_length=1, max_length=100_000)
    context: str = Field(default="", max_length=100_000)
    assistant_prefix: str = Field(default="", max_length=1024)
    max_new_tokens: int = Field(default=128, ge=1, le=512)


class GenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    protocol_version: Literal[1] = 1
    model_id: str = Field(min_length=1, max_length=128)
    output: str = Field(max_length=2048)
    generated_tokens: int = Field(ge=0, le=512)
    finish_reason: Literal["eos", "length"]


class BackendUnavailable(RuntimeError):
    """An unavailable backend, distinct from an invalid request or response."""


class ModelBackend(Protocol):
    def generate(self, payload: GenerationRequest) -> GenerationResponse: ...


def format_prompt(payload: GenerationRequest) -> str:
    prefix = ("[SYSTEM]: Kontekst poprzednich modyfikacji dla tego pliku:\n"
              + payload.context + "\n\n") if payload.context else ""
    return prefix + "[USER]: " + payload.prompt + "\n[AI]:\n"


class DenseLocalBackend:
    """CPU FP32 reference inference for the existing dense v2 byte checkpoint.

    The same runtime is used locally and by the HTTP service. Load a backend
    once; replace/restart it explicitly when changing checkpoints.
    """
    def __init__(self, model_path: str):
        from scripts.check_dense_runtime_parity import load_weights
        try:
            path = Path(model_path)
            raw = path.read_bytes()
            self.weights = load_weights(path, raw=raw)
            self.model_id = "sha256:" + hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError) as exc:
            raise BackendUnavailable("Cannot load dense v2 checkpoint") from exc
        import numpy as np
        w = self.weights
        arrays = [w["embedding"], w["position"], w["norm"], w["output"]]
        arrays.extend(v for layer in w["layers"] for v in layer.values())
        if not all(np.isfinite(a).all() for a in arrays):
            raise BackendUnavailable("Checkpoint contains non-finite weights")

    def generate(self, payload: GenerationRequest) -> GenerationResponse:
        import numpy as np
        from scripts.check_dense_runtime_parity import numpy_logits
        # Do not silently drop the user's task to fit a tiny context window.
        tokens = list((format_prompt(payload) + payload.assistant_prefix).encode("utf-8"))
        if len(tokens) > self.weights["context"]:
            raise ValueError("Formatted prompt exceeds this model's byte context")
        output = bytearray(payload.assistant_prefix.encode("utf-8"))
        reason = "length"
        count = 0
        for _ in range(payload.max_new_tokens):
            logits = numpy_logits(self.weights, tokens).reshape(-1)
            if not np.isfinite(logits).all():
                raise RuntimeError("Non-finite model logits")
            token = int(logits.argmax())
            count += 1
            if token == 0:  # NppAI byte-token EOS, matching training evaluation.
                reason = "eos"
                break
            output.append(token)
            tokens.append(token)
        return GenerationResponse(model_id=self.model_id,
            output=output.decode("utf-8", errors="replace"),
            generated_tokens=count, finish_reason=reason)


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RemoteBackend:
    def __init__(self, endpoint: str, api_key: str, *, allow_remote: bool = False,
                 timeout: float = 30):
        if not allow_remote:
            raise ValueError("Remote prompt transfer must be explicitly enabled")
        url = urlsplit(endpoint)
        try:
            loopback = ipaddress.ip_address(url.hostname or "").is_loopback
        except ValueError:
            loopback = url.hostname == "localhost"
        if (url.scheme not in ("http", "https") or not url.hostname
                or url.username or url.password or url.query or url.fragment
                or url.path != "/api/v1/generate"
                or (url.scheme == "http" and not loopback)):
            raise ValueError("Use HTTPS, or loopback HTTP, at /api/v1/generate")
        if not api_key or "\r" in api_key or "\n" in api_key:
            raise ValueError("A valid API key is required")
        if not 0 < timeout <= 300:
            raise ValueError("Timeout must be in (0, 300] seconds")
        self.endpoint, self.api_key, self.timeout = endpoint, api_key, timeout
        # Disable ambient proxies and redirects so credentials stay at endpoint.
        self.opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())

    def generate(self, payload: GenerationRequest) -> GenerationResponse:
        req = request.Request(self.endpoint, data=payload.model_dump_json().encode(),
            headers={"Content-Type": "application/json", "X-API-Key": self.api_key},
            method="POST")
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                raw = response.read(16_385)
        except error.HTTPError as exc:
            if exc.code in (429, 502, 503, 504):
                raise BackendUnavailable("Remote backend temporarily unavailable") from None
            raise ValueError(f"Remote backend rejected request (HTTP {exc.code})") from None
        except (error.URLError, TimeoutError, OSError):
            raise BackendUnavailable("Remote backend connection failed") from None
        if len(raw) > 16_384:
            raise ValueError("Remote response exceeds protocol limit")
        result = GenerationResponse.model_validate_json(raw)
        if result.generated_tokens > payload.max_new_tokens:
            raise ValueError("Remote response exceeds requested token budget")
        return result


class BackendRouter:
    def __init__(self, mode: Literal["local", "online", "hybrid"], *,
                 local: ModelBackend | None = None, remote: ModelBackend | None = None,
                 allow_remote: bool = False):
        if mode not in ("local", "online", "hybrid"):
            raise ValueError("Unknown backend mode")
        if mode != "local" and (not allow_remote or remote is None):
            raise ValueError("Online/hybrid requires explicit remote transfer and backend")
        if mode != "online" and local is None:
            raise ValueError("Local/hybrid requires a local backend")
        self.mode, self.local, self.remote = mode, local, remote

    def generate(self, payload: GenerationRequest) -> GenerationResponse:
        if self.mode == "online":
            return self.remote.generate(payload)
        try:
            return self.local.generate(payload)
        except BackendUnavailable:
            if self.mode != "hybrid":
                raise
            return self.remote.generate(payload)


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "online", "hybrid"), default="local")
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--context-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    try:
        payload = GenerationRequest(prompt=args.prompt,
            context=args.context_file.read_text(encoding="utf-8") if args.context_file else "",
            max_new_tokens=args.max_new_tokens)
        local = None
        if args.mode != "online":
            if not args.model:
                raise ValueError("--model is required for local/hybrid")
            # Defer loading: an unavailable checkpoint can trigger explicit hybrid fallback.
            class LazyLocal:
                def generate(self, payload):
                    return DenseLocalBackend(args.model).generate(payload)
            local = LazyLocal()
        remote = RemoteBackend(args.endpoint or "", os.getenv("NPPAI_API_KEY", ""),
            allow_remote=args.allow_remote) if args.mode != "local" else None
        result = BackendRouter(args.mode, local=local, remote=remote,
            allow_remote=args.allow_remote).generate(payload)
        print(result.model_dump_json())
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Generation failed: {exc}\n")


if __name__ == "__main__":
    main()
