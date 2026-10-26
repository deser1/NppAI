import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
import threading
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from model_backends import (BackendUnavailable, DenseLocalBackend,
                            GenerationRequest, GenerationResponse)

DATASET_PATH = Path("datasets/instruct_dataset.txt")
MODEL_PATH = Path("models/NppAI-model-v1.nppai")
TRAINING_SCRIPT = Path("train_nppai.py")
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_DATASET_ENTRY_BYTES = MAX_REQUEST_BYTES
MAX_PROMPT_CHARS = 100_000
MAX_THOUGHT_CHARS = 200_000
MAX_CODE_CHARS = 500_000
MAX_USER_ID_CHARS = 128
TRAINING_THRESHOLD = 5
MODEL_CHUNK_SIZE = 1024 * 1024
API_KEY_ENV = "NPPAI_API_KEY"
LOG_LEVEL_ENV = "NPPAI_LOG_LEVEL"
RATE_LIMIT_REQUESTS_ENV = "NPPAI_RATE_LIMIT_REQUESTS"
RATE_LIMIT_WINDOW_ENV = "NPPAI_RATE_LIMIT_WINDOW_SECONDS"
DEFAULT_RATE_LIMIT_REQUESTS = 60
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60

logging.basicConfig(
    level=getattr(logging, os.getenv(LOG_LEVEL_ENV, "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("nppai.backend")


app = FastAPI(
    title="NppAI Cloud Backend",
    description="Backend for NppAI knowledge ingestion and model updates.",
)

training_task: asyncio.Task | None = None
new_samples_count = 0
training_lock = asyncio.Lock()
dataset_lock = asyncio.Lock()
rate_limit_lock = asyncio.Lock()
rate_limit_hits: dict[str, deque[float]] = defaultdict(deque)
model_version_cache_key: tuple[int, int] | None = None
model_version_cache_value: str | None = None
inference_lock = threading.Lock()
inference_backend: DenseLocalBackend | None = None


def configured_inference_backend() -> DenseLocalBackend:
    """Called while inference admission is held; pin one NppAI checkpoint."""
    global inference_backend
    if inference_backend is None:
        path = os.getenv("NPPAI_INFERENCE_MODEL")
        if not path:
            raise BackendUnavailable("Inference model is not configured")
        options = {}
        try:
            width = int(os.getenv("NPPAI_PROMPT_BYTES", "0"))
        except ValueError:
            raise BackendUnavailable("Invalid NPPAI_PROMPT_BYTES") from None
        profile = os.getenv("NPPAI_PYTHON_FUNCTIONS", "0")
        if profile not in ("0", "1"):
            raise BackendUnavailable("Invalid NPPAI_PYTHON_FUNCTIONS")
        if width: options["prompt_bytes"] = width
        if profile == "1": options["python_functions"] = True
        try:
            inference_backend = DenseLocalBackend(path, **options)
        except ValueError as exc:
            raise BackendUnavailable("Invalid inference profile: " + str(exc)) from None
    return inference_backend


def generate_with_configured_model(payload: GenerationRequest) -> GenerationResponse:
    """Called with admission held; releases it even after client cancellation."""
    try:
        return configured_inference_backend().generate(payload)
    finally:
        inference_lock.release()


def inspect_configured_model() -> dict:
    try:
        backend = configured_inference_backend()
        return {"protocol_version": 1, "status": "ready", "backend": "nppai-dense-v2",
                "model_id": backend.model_id, "context_bytes": backend.weights["context"],
                "prompt_bytes": getattr(backend, "prompt_bytes", 0),
                "decode_profile": "python_function_beam8" if getattr(backend, "python_functions", False) else "greedy"}
    finally:
        inference_lock.release()


class SubmitKnowledgeRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    final_code: str = Field(min_length=1, max_length=MAX_CODE_CHARS)
    thought_process: str = Field(default="", max_length=MAX_THOUGHT_CHARS)
    user_id: str = Field(default="anonymous", max_length=MAX_USER_ID_CHARS)


class SubmitKnowledgeResponse(BaseModel):
    status: str
    message: str


class CheckModelUpdateResponse(BaseModel):
    update_available: bool
    version: str | None = None
    download_url: str | None = None


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected_key = os.getenv(API_KEY_ENV)
    if not expected_key:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key required")
    if not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(status_code=403, detail="Invalid API key")


def positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def monotonic_time() -> float:
    return time.monotonic()


async def enforce_rate_limit(request: Request) -> None:
    limit = positive_int_env(RATE_LIMIT_REQUESTS_ENV, DEFAULT_RATE_LIMIT_REQUESTS)
    window = positive_int_env(RATE_LIMIT_WINDOW_ENV, DEFAULT_RATE_LIMIT_WINDOW_SECONDS)
    client_key = request.client.host if request.client else "unknown"
    now = monotonic_time()

    async with rate_limit_lock:
        hits = rate_limit_hits[client_key]
        cutoff = now - window
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= limit:
            retry_after = max(1, int(window - (now - hits[0])) + 1)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

        hits.append(now)


@app.middleware("http")
async def request_size_limit(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})

    chunks: list[bytes] = []
    total_bytes = 0
    async for chunk in request.stream():
        total_bytes += len(chunk)
        if total_bytes > MAX_REQUEST_BYTES:
            return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        chunks.append(chunk)

    # Starlette's downstream request handling can reuse the cached body without
    # reading the already-consumed ASGI receive channel again.
    request._body = b"".join(chunks)
    return await call_next(request)


def model_version() -> str:
    global model_version_cache_key, model_version_cache_value

    if not MODEL_PATH.is_file():
        model_version_cache_key = None
        model_version_cache_value = None
        return "0"

    stat = MODEL_PATH.stat()
    cache_key = (stat.st_mtime_ns, stat.st_size)
    if cache_key == model_version_cache_key and model_version_cache_value is not None:
        return model_version_cache_value

    digest = hashlib.sha256()
    with MODEL_PATH.open("rb") as f:
        while chunk := f.read(MODEL_CHUNK_SIZE):
            digest.update(chunk)

    version = f"{stat.st_mtime_ns:x}-{stat.st_size:x}-{digest.hexdigest()[:16]}"
    model_version_cache_key = cache_key
    model_version_cache_value = version
    return version


async def run_training_process():
    global training_task
    async with training_lock:
        if training_task is not None and not training_task.done():
            return

        if not TRAINING_SCRIPT.is_file():
            logger.error("event=training_script_missing path=%s", TRAINING_SCRIPT)
            return

        async def run():
            logger.info("event=training_started script=%s", TRAINING_SCRIPT)
            process = await asyncio.create_subprocess_exec(
                "python",
                str(TRAINING_SCRIPT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                logger.info("event=training_completed return_code=%s", process.returncode)
            else:
                logger.error(
                    "event=training_failed return_code=%s stderr=%r",
                    process.returncode,
                    stderr.decode("utf-8", errors="replace")[-2000:],
                )

        training_task = asyncio.create_task(run())


@app.get("/api/v1/health", tags=["Inference"])
async def inference_health(
    _: None = Depends(require_api_key),
    __: None = Depends(enforce_rate_limit),
):
    """Readiness checks load the actual NppAI file; no generation/training."""
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Inference is busy")
    try:
        future = asyncio.get_running_loop().run_in_executor(None, inspect_configured_model)
    except BaseException:
        inference_lock.release()
        raise
    try:
        return await asyncio.shield(future)
    except BackendUnavailable:
        raise HTTPException(status_code=503, detail=
            "NppAI model unavailable. Set NPPAI_INFERENCE_MODEL to a valid dense v2 byte .nppai file.") from None
    except Exception:
        logger.error("event=inference_readiness_failed")
        raise HTTPException(status_code=500, detail="NppAI model readiness failed") from None


@app.post("/api/v1/generate", response_model=GenerationResponse, tags=["Inference"])
async def generate_code(
    payload: GenerationRequest,
    _: None = Depends(require_api_key),
    __: None = Depends(enforce_rate_limit),
):
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Inference is busy")
    try:
        # Shield keeps the worker's admission slot held after HTTP cancellation.
        future = asyncio.get_running_loop().run_in_executor(
            None, generate_with_configured_model, payload)
    except BaseException:
        inference_lock.release()
        raise
    try:
        return await asyncio.shield(future)
    except BackendUnavailable:
        raise HTTPException(status_code=503, detail="Inference model unavailable") from None
    except ValueError:
        raise HTTPException(status_code=422, detail="Request exceeds model capacity") from None
    except Exception:
        logger.error("event=inference_failed")
        raise HTTPException(status_code=500, detail="Inference failed") from None


@app.post(
    "/api/submit_knowledge",
    response_model=SubmitKnowledgeResponse,
    tags=["Knowledge"],
)
async def submit_knowledge(
    payload: SubmitKnowledgeRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_api_key),
    __: None = Depends(enforce_rate_limit),
):
    global new_samples_count

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        "\n[USER]:\n"
        f"{payload.prompt}\n\n"
        "[SYSTEM]:\n"
        f"{payload.thought_process}\n\n"
        "[AI]:\n"
        f"{payload.final_code}\n<|endoftext|>\n"
    )

    if len(entry.encode("utf-8")) > MAX_DATASET_ENTRY_BYTES:
        raise HTTPException(status_code=413, detail="Training sample too large")

    should_schedule_training = False
    try:
        async with dataset_lock:
            with DATASET_PATH.open("a", encoding="utf-8") as f:
                f.write(entry)

            new_samples_count += 1
            if new_samples_count >= TRAINING_THRESHOLD:
                new_samples_count = 0
                should_schedule_training = True
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Unable to persist training sample") from exc

    logger.info(
        "event=training_sample_accepted user_id=%r sample_bytes=%d",
        payload.user_id,
        len(entry.encode("utf-8")),
    )

    if should_schedule_training:
        background_tasks.add_task(run_training_process)

    return SubmitKnowledgeResponse(
        status="success",
        message="Knowledge saved.",
    )


@app.get(
    "/api/check_model_update",
    response_model=CheckModelUpdateResponse,
    tags=["Model"],
)
async def check_model_update(client_version: str = "0"):
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=404, detail="Model is not available")

    server_version = await asyncio.to_thread(model_version)
    if client_version != server_version:
        return CheckModelUpdateResponse(
            update_available=True,
            version=server_version,
            download_url="/api/download_model",
        )

    return CheckModelUpdateResponse(update_available=False)


@app.get("/api/download_model", tags=["Model"])
async def download_model():
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=404, detail="Model is not available")

    def iterfile():
        with MODEL_PATH.open("rb") as file_like:
            while chunk := file_like.read(MODEL_CHUNK_SIZE):
                yield chunk

    return StreamingResponse(
        iterfile(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": (
                "attachment; filename=NppAI-model-v1.nppai"
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("event=backend_starting host=0.0.0.0 port=8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
