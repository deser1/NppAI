# Experimental specification → plan → code generation

This is an opt-in **local Python experiment using our own NppAI dense v2 byte
checkpoint**, not a newly trained reasoning model or a production plugin mode.
No external model, cloud adapter, reference-derived signature, tool execution or
editor modification is involved. Existing Notepad++ validation, repair and Apply
remain separate and unchanged.

`staged_generation.py` makes three sequential calls to one backend:

1. A short task specification: required inputs, output and edge cases.
2. A short implementation plan, receiving the original task and raw specification.
3. Python code, receiving the original task and both raw prior artifacts.

These artifacts are fallible generated text, not neural internals or verified
requirements. The runner checks nonempty output, EOS termination, response schema,
per-call budget and stable checkpoint identity. It does **not** judge whether a
specification or plan is semantically correct. No automatic reference correction,
context truncation or fallback hides a failed stage. Empty, truncated, over-budget,
changed-checkpoint and context-overflow responses stop the pipeline and retain the
partial trace. `completed` means the three calls completed; it does not mean the
code is correct, safe to apply, or tested. `execution_status` is always `not_run`.

## Run

Install `requirements-inference.txt`, then:

```powershell
py -3.12 staged_generation.py --model "C:\Models\own-dense-v2.nppai" `
  --task "Define silnia(n) in Python to return n factorial." `
  --output staged-result.json

py -3.12 scripts/evaluate_task_plan.py --model "C:\Models\own-dense-v2.nppai" `
  --dataset evaluation/instruction-v2/test.json --output task-plan-report.json
```

The standalone runner prints actual stage-start names to stderr. JSON retains
exact requests, raw responses, model identity, token counts, elapsed time and
failures. Exit 2 indicates an incomplete/erroring experiment; a report is retained
when inference-stage failures occur. Load/input failures produce an explicit error.
No prompts or traces leave the machine. Reports contain user text and generated
artifacts; choose their destination accordingly.

The evaluator compares the same checkpoint and task in three arms:

| Arm | Calls | Default output-token ceilings |
| --- | --- | --- |
| Staged | At most 3 | 128 per stage, 384 total |
| Direct, final-code budget | 1 | 128 |
| Direct, aggregate budget | 1 | 384 |

The permitted stage budget is 1–170, so the aggregate remains within the existing
512-token protocol limit. Shared ceilings do not equalize actual output tokens,
input lengths, memory, time or compute. The staged arm requires EOS for every
stage; direct raw outputs are scored even on length termination. Both comparisons
must retain this conservative acceptance difference. References and requested
function names enter scoring only, never model requests. Scores include raw exact
response, full reference AST, syntax and requested function name; none execute
code or establish functional correctness. Empty/error cases stay in denominators.
Dataset and implementation hashes are recorded. Existing synthetic test tasks
are exposed regression tasks, not new independent generalization evidence.

## First frozen-checkpoint measurement (18 October 2026)

Source: run [38314121142](https://github.com/deser1/NppAI/actions/runs/38314121142),
artifact `nppai-broader-instruction-v2`, ID `11775752592`.
Archive SHA-256:
`80649135c7a910742984a48fd8c60aff2bf58da9579f6457f18f3e99e61575c2`.
Checkpoint SHA-256:
`6a49081e1d94e85d31efe6bdf2a4e3d9e4d7b06bb381c492de3a98d1300ed9ac`.
This historical 3,000-step, seed-42 model has **256 bytes of context**. It is not a
new checkpoint or a claim about every NppAI model. Weights were not modified. A same-host repeat matched every request, raw output,
score and failure after excluding elapsed time; source/archive/report hashes are
retained in [source.json](evidence/task-plan/source.json).

Full requests/responses and failures are retained in
[evidence/task-plan/report.json](evidence/task-plan/report.json).

| Metric, all 28 exposed v2 test tasks | Staged | Direct 128 | Direct 384 |
| --- | ---: | ---: | ---: |
| Nonempty final response | 0 | 26 | 26 |
| Any Python function syntax | 0 | 13 | 13 |
| Requested function name | 0 | 0 | 0 |
| Full reference AST / exact response | 0 / 0 | 0 / 0 | 0 / 0 |
| Stage/inference failures | 28 | 0 | 0 |

Staged failures: **14 empty specifications**, **one specification at the token
limit**, and **13 context overflows at the plan stage**. No case reached code
production. Some nonempty specifications were malformed code-like text; passing
nonempty/EOS checks does not validate their meaning. No generated source was
executed. This checkpoint does not support the proposed stage protocol adequately;
**do not enable the path by default or promote the checkpoint**.

## Next quality gate

- Prepare authored specification/plan/code examples from the training split only;
  review correctness rather than labeling generated plans as valid automatically.
- Train the own model on that explicit protocol with a context window sufficient
  for the complete task, artifacts and code. Changing only `<THINK>` markers does
  not provide this capability. Keep a direct-generation control and retention set.
- Freeze new tasks and wording variants before training; record exposure and keep
  previously inspected v2 tasks as regression diagnostics only.
- Compare complete outputs, actual tokens/time and isolated authored functional
  tests, including original-hit retention. Do not promote on syntax/AST alone.
- Connect a successful candidate to `ValidatedGeneration` and its existing bounded
  repair loop, then expose stage events in the panel. Preserve cancellation,
  diagnostic repair, diff review and validated Apply. Test that integration before
  enabling it; the Python experiment is not yet connected to the DLL.

The current work supplies execution/evaluation infrastructure and an actual failed
baseline measurement. Protocol training, held-out semantic quality, functional
execution and production panel integration remain open.
