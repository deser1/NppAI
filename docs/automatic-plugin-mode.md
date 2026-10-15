# Automatic inference selection in the Notepad++ plugin

The generation button uses `auto` mode by default. The plugin chooses where to
request a completion; the user does not choose a backend for each question.
The server is the user's configured NppAI service and model, not a built-in
subscription to an external model provider.

| Condition | Automatic decision |
| --- | --- |
| Local checkpoint loaded and entire formatted input fits its context | Generate locally |
| Local checkpoint unavailable | Use the configured online backend |
| Input exceeds local context | Use the configured online backend |
| Local model produces only EOS/whitespace after an optional function prefix | Retry once online, before publishing code |
| Online generation not permitted/configured and local cannot serve task | Display an error; do not truncate the task |
| Local code has already been streamed | Finish locally; no second completion is appended automatically |

Context fit is conservative: UTF-8 **bytes** of the complete formatted prompt,
including retrieved context and any Python function prefix, are compared against
native model token capacity. With BPE this can send an otherwise fitting task
online; it avoids optimistic estimates that discard the user's task. Current
native/server architecture and model limits still apply. A server with the same
tiny context cannot solve a context-overflow request merely by being online.

These are capability checks, not a learned complexity classifier or confidence
score. Nonempty output may still contain incorrect code. Compilation/tests and a
bounded repair workflow remain separate work; the existing weak-model quality
results are not improved by routing. Nothing executes generated code.

## One-time configuration

Start the inference service described in [model-backends](model-backends.md).
Then set the following environment variables **before launching Notepad++**:

```powershell
$env:NPPAI_PLUGIN_MODE = "auto"
$env:NPPAI_ALLOW_REMOTE = "1"
$env:NPPAI_GENERATION_URL = "https://your-server.example/api/v1/generate"
$env:NPPAI_API_KEY = "your-private-api-key"
# Launch Notepad++ from this environment.
```

Use `http://127.0.0.1:8000/api/v1/generate` for a service on the same machine.
The plugin requires HTTPS for other hosts. It sends the prompt, selected context
and optional signature prefix only when its selector actually calls online
inference. Remote capability requires all three: opt-in, endpoint and API key.
Automatic selection does not require repeated confirmation. No hosted service,
credentials or hardware are provisioned by installing the plugin.

`NPPAI_PLUGIN_MODE=local` forces local inference; `online` forces the server;
`hybrid` is an alias for `auto`. A misspelled mode fails visibly. Without remote
configuration, default `auto` operates locally within the available model limit.
This controls **generation only**, not the existing knowledge-ingestion,
telemetry or model-update paths.

The status label shows the chosen backend and reason and retains it after a
successful response. Backend/protocol errors are displayed in the panel rather
than inserted into the source file or reported as success. Failure after partial
local streaming leaves that partial text available for normal editor undo;
there is no automatic remote retry after code is published.

Native local streaming starts on the first non-whitespace completion byte and
preserves the optional signature prefix exactly once. Online output is buffered
until its HTTP response and schema have been validated, then delivered through
the same editor callback. Generation has a 512-new-token cap. Responses are
bounded to 16 KiB, use strict UTF-8/JSON schema checks, and never follow HTTP
redirects. The client uses no ambient proxy or automatic Windows authentication.
The API key is sent only to the configured endpoint. The server must accept the
optional `assistant_prefix` request field introduced with this integration;
older servers that reject this field need updating.

Stop requests cancel native inference and discard any outstanding online result.
For synchronous network operations, stopping can wait for the current bounded
WinHTTP operation (5s connect/resolve, 30s send/receive timeouts). The button stays
in its stopping state until the worker finishes, preventing overlapping requests.
Loading/updating a local model is serialized with generation so the routing
snapshot and the model used for it agree.

## Tests

`TestAutomaticModelRouting` checks automatic/forced decisions, context boundaries,
request escaping, response schema/version/budget validation and Unicode handling.
`TestPluginRemoteTransport` starts an authored loopback HTTP fixture and invokes
the actual native WinHTTP client, verifying request/auth/prefix parity, redirects,
HTTP errors, oversized/invalid responses and disallowed URLs. The Python runtime
suite checks exact prefix preservation and byte-context accounting too. These
fixtures test engineering behavior without evaluating or executing model code.

The same HTTP test loads the test plugin DLL and calls real `AIManager` generation: no remote permission, missing local model, a fitting real local fixture, context overflow, EOS-only local fallback, forced-local isolation and online Python signature preservation. The test-only exports are absent from release DLLs.
