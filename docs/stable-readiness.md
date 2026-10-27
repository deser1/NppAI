# Stable release checklist

Status: **blocked on model quality and interactive Windows validation**.
Passing infrastructure CI does not waive either gate. Evidence is specific to a
source commit, checkpoint, package, runtime and task set.

| Gate | Current evidence | Remaining work |
| --- | --- | --- |
| Builds and deterministic tests | CI builds Release/Debug x64, x86 and ARM64 | Require green CI on the final proposed source commit |
| Installable artifacts | Public v0.5.0-rc.2 plugin ZIPs, checksums and standalone service; real fresh x64 editor smoke | Existing-installation upgrade, rollback and cancellation during generation/validation |
| Editor compatibility | Actual portable Notepad++ 8.9.8.1 x64 on the hosted windows-2022 runner | Interactive Windows 10/11 + Docker Desktop validation; actual x86/ARM64 editor runs. Other versions remain unverified |
| Validated editor operation | Linux Docker validator and Windows request/review verified separately | Run `scripts/windows_validated_smoke.ps1`; require actual successful validation, enabled Apply, exact applied bytes and one Undo restoring the buffer |
| Model distribution | Permanent model-factorial-v1 checkpoint/service/card/raw evidence and SHA-256 | Preserve checkpoint identity in every future measurement; do not replace failed source weights silently |
| Complete-task quality | Narrow factorial development success; broader paired evaluation in `released-generalization-v1` | Predeclare a target for the intended product scope, improve on training/development data and evaluate a newly frozen independent suite after freezing a candidate |
| Staged generation | Existing planner/code experiments, recorded retention and new-task failures | Functional gains and protection of original direct/specification/plan/code hits before production routing |
| Native runtime support | Dense v2 native runtime; extra factorial decoding runs in the own Python service | Do not claim the optional search profile works in native C++; certify any future architecture/export separately |
| Public backend deployment | Authentication/TLS requirements and correction-data privacy policy documented | Deployment-specific TLS/proxy/log review and an enforceable data-retention schedule before real user collection |

## Run the missing interactive Windows check

Use an interactive Windows x64 session with Python 3.12, GitHub CLI and Docker
Desktop's Linux engine. Run from a checkout containing the script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows_validated_smoke.ps1
```

Retain the printed evidence directory. Require successful positive/negative
Docker controls, real model identity, actual UI events and recording, applied
buffer equality and Undo equality. A successful Linux run or request-only GIF
cannot substitute for these checks.

For an existing installation, separately record the previous plugin/model
versions and checksums, close the editor, back up configuration, update matching
architecture assets, run the same validation, then restore the previous version
and verify it starts. Record Windows/editor/Docker versions and failures. No
existing user installation is modified by the hosted fresh-install test.

## Model development rule

The 40-task released-model suite is evaluation-only. Once read, its rows are
exposed regressions. Do not train on them or use repeated scores to describe a
later candidate as independently evaluated. New training data must have its own
provenance, program-family split, bilingual/name/argument views and reviewed
functional controls. Freeze a candidate before authoring the next independent
measurement. A syntactically valid header is application assistance when the
decoder supplied it; report body correctness separately.

Do not create a stable model release while these gates remain unresolved.
