# Local and online model backends

The standalone Python generation client and the FastAPI service now share a
versioned request/response contract and the same dense v2 `.nppai` FP32 reference
runtime. This is an executable foundation for a common agent, not a completed
autonomous coding agent. The Notepad++ panel now has its own automatic selector using the same HTTP
contract; see [automatic-plugin-mode](automatic-plugin-mode.md). Existing
telemetry/knowledge ingestion and model updates are separate paths.

## Run locally

Install `numpy>=1.26,<3` and `pydantic>=2.7,<3`. Supply your exported **dense v2,
256-byte vocabulary** checkpoint; other formats/architectures are rejected.
The reference implementation favors portability and testability over speed.
It does not require PyTorch, a GPU, an external model API, or internet access.

```powershell
python model_backends.py --model models/custom.nppai --prompt "Implement add(a, b)." --max-new-tokens 128
```

The default mode is `local`. The result is JSON containing `protocol_version`,
`model_id` (SHA-256 of exactly the loaded file), `output` (completion only),
`generated_tokens` (including EOS if produced), and `finish_reason` (`eos` or
`length`). Invalid UTF-8 byte sequences become replacement characters. The
formatted input must fit the model's byte context; oversized tasks/context
fail rather than silently dropping the task. Generation uses greedy decoding,
a rolling context for subsequent tokens, and a maximum of 512 new byte tokens.
An optional `--context-file path` reads explicitly selected UTF-8 context.

## Run the service and online client

For a checked Windows startup with one session key shared by the server and a fresh Notepad++ instance, see the [Polish setup guide](notepad-setup-pl.md) and `scripts/start_nppai.py`. `GET /api/v1/health` requires the API key and loads the actual NppAI checkpoint before reporting readiness. The launcher never substitutes an external model.


Install the server dependencies in `requirements.txt`. Configure authentication
and an explicit inference checkpoint before starting the service:

```powershell
$env:NPPAI_API_KEY = "replace-with-your-private-key"
$env:NPPAI_INFERENCE_MODEL = "models/custom.nppai"
python -m uvicorn server_backend:app --host 127.0.0.1 --port 8000
python model_backends.py --mode online --allow-remote --endpoint http://127.0.0.1:8000/api/v1/generate --prompt "Implement add(a, b)."
```

`NPPAI_API_KEY` supplies the client's `X-API-Key` header too; it is not a CLI
argument. For a server on another machine, configure HTTPS and use its URL.
The client refuses non-loopback HTTP, embedded URL credentials, query strings,
and redirects. It bypasses ambient proxy settings. The default timeout is 30s.
Starting a local server does not deploy a hosted service or provision hardware.

`POST /api/v1/generate` accepts:

```json
{"protocol_version":1,"prompt":"Implement add(a, b).","context":"","assistant_prefix":"","max_new_tokens":128}
```

Unknown fields, unsupported versions, invalid types and token budgets are
rejected. Authentication, the existing body size cap and per-client rate limit
apply. One inference runs per process; competing requests receive 503. A
cancelled HTTP request leaves that slot occupied until its bounded worker
finishes. Inference runs outside the event loop. Requests do not append to the
training dataset, trigger training, or execute returned code. The model loads
lazily and stays pinned until process restart; training/update endpoints cannot
silently swap an in-flight inference checkpoint. Missing configuration or an
invalid checkpoint returns 503; context overflow returns 422. Internal failures
return a generic 500 without exposing filesystem paths or prompts.

## Routing semantics

| Mode | Behavior |
| --- | --- |
| `local` | Calls only the local model. No remote fallback. |
| `online` | Calls only the explicitly configured server. Requires `--allow-remote`. |
| `hybrid` | Tries local first; only `BackendUnavailable` enables server fallback. Requires both model and endpoint, plus `--allow-remote`. |

Hybrid does not judge model quality or silently transfer oversized context.
Invalid inputs, model capacity errors and invalid remote responses fail. HTTP
429/502/503/504 and connection failures are classified as remote unavailability;
authentication failures are not. Example:

```powershell
python model_backends.py --mode hybrid --model models/custom.nppai --allow-remote --endpoint https://your-server.example/api/v1/generate --prompt "Implement add(a, b)."
```

Enabling remote transfer permits sending **both the prompt and selected context**
when that backend is used. The local backend does not scan a repository or select
files by itself. This policy covers this new client, not the plugin's preexisting
knowledge-ingestion or telemetry paths.

## Validation and remaining work

`tests/test_model_backends.py` exercises a real authored `.nppai` fixture through
local inference and ASGI, plus actual loopback HTTP transport, routing isolation,
explicit opt-in, authentication, limits, corrupt checkpoints, EOS, model identity,
busy responses and cancellation admission. These are engineering tests, not
measurements of model intelligence or useful coding quality.

The plugin is connected through its automatic selector and the initial
[Python validation/repair/review workflow](validated-generation.md). Multi-file
repository tool policies and complete-task measurements with retained
model-quality gates remain open. No checkpoint is
promoted or trained by this change; the observed arbitrary-name copying limitation
remains open. Larger server models and more architectures need separate runtime
adapters and evaluations.
