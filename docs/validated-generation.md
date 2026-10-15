# Generate, validate, repair and review

The Notepad++ generation button now stages a replacement for the captured
selection (or an insertion at the captured caret). It keeps the original UTF-8
buffer unchanged during generation and validation. The first supported profile
is a Python document: a `.py` filename or an explicit `@python-signature` task.

The flow connects the existing `AgentFeedbackLoop`, `BoundedRepairLoop`,
`StructuredPatch`, `PatchReview` and `AgentTaskOutcome` components:

1. Capture the document, buffer identity and selection.
2. Ask the automatically selected local/online model for a replacement.
3. Compile the entire proposed Python buffer without executing it.
4. Run authored function test cases in isolated child processes inside the
   same restricted local Docker profile.
5. On a candidate failure, supply actual diagnostics and the previous proposal
   as feedback. There are **three candidates total**, so at most two repairs.
6. After two failed local candidates, automatic mode may force the last attempt
   online, when online generation is configured and allowed. Ordinary context
   limits can already route earlier repairs online. Forced-local mode never
   escalates; infrastructure errors and cancellation stop the loop.
7. Show the diff, attempt count, backend decisions and diagnostics. Enable
   **Apply** only after both syntax and the configured tests pass.
8. **Apply** checks that the original document is active, writable and byte-for-
   byte unchanged, then changes the editor buffer as one undo action. It does
   not save the file. **Reject** discards the proposal.

The repair context includes a bounded excerpt around the captured selection;
validation always uses the complete proposed document. Current review limits
are 128 KiB and 512 newline characters per document/candidate. A successful
validation means these configured cases passed, not general program correctness.
No real-model success rate is claimed by the deterministic fixture tests.

## Local setup (Windows, Docker Linux containers)

Install Python and Docker Desktop with its Linux engine. Use a trusted runtime
image with the Python standard library (custom dependencies must be prepared in
that image beforehand). Pulling/building is a separate setup action; validation
never pulls images or installs packages.

Run this in PowerShell before launching Notepad++:

```powershell
$dockerHost = 'npipe:////./pipe/docker_engine'
docker --host $dockerHost pull python:3.12-slim
$env:NPPAI_VALIDATION_IMAGE = docker --host $dockerHost image inspect python:3.12-slim --format '{{.Id}}'
$env:NPPAI_VALIDATION_PYTHON = python -c "import sys; print(sys.executable)"
$env:NPPAI_TEST_CASES = (Resolve-Path '.\examples\python-validation-cases.json').Path
# Launch Notepad++ from this environment.
```

The installed plugin directory must contain `sandbox_validate.py` beside the
DLL. CMake builds and new release ZIP packages copy it there. For an MSBuild-only
local build, copy that file beside the resulting DLL yourself. The Python
executable and test-suite path must be absolute. The validator image must be a
local immutable `sha256:` image ID, not a mutable tag.

For the example suite, use an empty UTF-8 document and this task:

```text
@python-signature add(a, b)
Return the sum of a and b.
```

Author cases for **your actual task**, rather than accepting tests invented by
the generating model. The JSON suite must contain 1..100 cases (maximum 64 KiB):

```json
[
  {"function":"add","args":[2,3],"kwargs":{},"expected":5},
  {"function":"add","args":[-4,4],"kwargs":{},"expected":0}
]
```

Each case calls one named top-level function with JSON arguments and compares
its JSON-compatible return value with `expected`. Extra prints, import errors,
wrong results and premature process exit fail. The suite is captured once per
panel request; diagnostics record its SHA-256. No tests or missing Docker/runtime
configuration means validation is blocked and Apply stays disabled. Other
languages and multi-file builds need separate validator profiles; this release
does not claim to validate them.

Online generation remains configured as in
[automatic-plugin-mode](automatic-plugin-mode.md). Local validation never needs
online generation permission. If a repair uses the online backend, the selected
context, previous proposal and bounded diagnostics are transferred under that
existing one-time opt-in. The validator itself permits only a local Unix socket
or Windows named-pipe Docker endpoint, even if Docker's ambient context points
at a remote daemon.

## Execution boundary and limits

`sandbox_validate.py` is a trusted host driver; it never imports or executes the
candidate on the host. Native code launches that fixed script with argument
quoting, restricted inherited handles, bounded output and a Windows job object.
Generated source is a temporary data file, never a command line or shell command.

The driver feeds the candidate and cases over stdin to a local container. It
uses no host bind mounts, no host environment variables or credentials, no
network, a read-only root filesystem, an unprivileged UID, dropped capabilities,
no-new-privileges, and CPU/memory/process limits. `/tmp` is a small temporary
filesystem. The image entrypoint is replaced by the trusted Python supervisor.
Expected answers and a per-run receipt stay in the supervisor, outside the
candidate child's input; an exit code of zero or a printed `pass` alone cannot
count as validation success. Test coverage is still limited by authored cases.

Each child function test has a two-second timeout and each container phase has
a twenty-second wall deadline. Output is bounded. Cancellation requests cleanup
of the named container; the driver attempts forced removal on every completion.
The native wrapper has a further deadline and kills its helper process tree
when necessary. A broken Docker daemon can prevent cleanup and needs operator
attention. This is a restricted local container boundary, not a formal sandbox
escape guarantee; use a dedicated disposable validation host for hostile code.
See the official [Docker run reference](https://docs.docker.com/reference/cli/docker/container/run/)
for the runtime controls.

Container/suite/runtime failures are reported separately from candidate errors
and do not consume repeated model repairs. Closing the panel or stopping a task
prevents its pending proposal from being applied. Editing or switching away
from the captured document blocks application. Nothing is auto-saved or
silently rolled back over subsequent user edits.

## Evidence

- Portable coordinator tests cover syntax-before-tests, diagnostic repairs,
  the three-candidate budget, permitted online escalation, forced-local
  isolation, cancellation, infrastructure blocking and review/stale-buffer rules.
- Windows tests exercise the native helper process using authored I/O fixtures
  and the actual plugin's editor review boundary, including one undo action.
- A dedicated Linux CI job runs real containers for correct/incorrect code,
  syntax errors, premature exit/forged success text, nontermination, network and
  root-filesystem restrictions, credential isolation and container cleanup.
- These tests use authored fixtures. They do not execute model-generated code
  on a regular host or promote any checkpoint.
