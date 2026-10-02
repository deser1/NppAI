import asyncio
import hashlib
import json
import os
import secrets
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

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

app = FastAPI(
    title="NppAI Cloud Backend",
    description="Backend for NppAI knowledge ingestion and model updates.",
)

training_task: asyncio.Task | None = None
new_samples_count = 0
training_lock = asyncio.Lock()
dataset_lock = asyncio.Lock()


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
        return
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key required")
    if not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.middleware("http")
async def request_size_limit(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})

    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Request body too large"})

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = receive
    return await call_next(request)


def model_version() -> str:
    if not MODEL_PATH.is_file():
        return "0"
    stat = MODEL_PATH.stat()
    digest = hashlib.sha256()
    with MODEL_PATH.open("rb") as f:
        while chunk := f.read(MODEL_CHUNK_SIZE):
            digest.update(chunk)
    return f"{stat.st_mtime_ns:x}-{stat.st_size:x}-{digest.hexdigest()[:16]}"


async def run_training_process():
    global training_task
    async with training_lock:
        if training_task is not None and not training_task.done():
            return

        if not TRAINING_SCRIPT.is_file():
            print(f"[{datetime.now()}] Training script not found: {TRAINING_SCRIPT}")
            return

        async def run():
            print(f"[{datetime.now()}] Starting background training...")
            process = await asyncio.create_subprocess_exec(
                "python",
                str(TRAINING_SCRIPT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                print(f"[{datetime.now()}] Training completed successfully.")
            else:
                print(
                    f"[{datetime.now()}] Training failed with code "
                    f"{process.returncode}: "
                    f"{stderr.decode('utf-8', errors='replace')}"
                )

        training_task = asyncio.create_task(run())


@app.post(
    "/api/submit_knowledge",
    response_model=SubmitKnowledgeResponse,
    tags=["Knowledge"],
)
async def submit_knowledge(
    payload: SubmitKnowledgeRequest,
    background_tasks: BackgroundTasks,
    _: None = Header(default=None, alias="X-NppAI-Auth-Checked"),
):
    require_api_key()
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

    print(
        f"[{datetime.now()}] Accepted training sample from "
        f"user_id={payload.user_id!r}"
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

    server_version = model_version()
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

    print("Starting NppAI Cloud Backend...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
