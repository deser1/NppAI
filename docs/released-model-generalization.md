# Frozen evaluation of the released factorial model

The published checkpoint was frozen before this suite was authored/read out.
No training, repair, checkpoint selection or production decoder changes occur in
this measurement. Protocol, tasks and evaluator were committed in `8e2dac4`
before the first GitHub Actions readout.

[Measured run 40423348120](https://github.com/deser1/NppAI/actions/runs/40423348120)
passed on 27 October 2026 (Europe/Warsaw). Its successful workflow means evidence
collection, controls, audit and cleanup passed; it does **not** mean the model
passed the tasks.

## Functional results

| Task group | Count | Released profile | Legacy greedy |
| --- | --- | --- | --- |
| Six operation families, two name/argument views, English/Polish | 24 | 0/24 | 0/24 |
| Factorial with two additional name/argument views, English/Polish | 4 | 0/4 | 0/4 |
| Previously exposed six-task bilingual regression | 12 | 1/12 | 0/12 |
| Total | 40 | **1/40** | **0/40** |

The only successful row is `new-factorial-en`. This is the first measurement of
this released checkpoint on the existing bilingual suite; the earlier
checkpoint's published 0/12 result remains attached to those older weights.
The new model passes one exposed factorial row, not the full six-task suite.

All 40 authored reference controls pass in restricted Docker, and all 40
constant-None negative controls fail. Both arms preserve their in-memory weights
and checkpoint file. Both emit all 40 records without generation exceptions.
The production profile has 17/40 syntactically parseable replies with matching
requested signatures, but only one passes functional tests. Its explicit
header may be mechanically supplied by the decoder: signature correctness is
not evidence of learned identifier copying or correct bodies. Legacy parsing
also accepts empty modules; inspect `nonempty` separately rather than treating
an empty parse as a useful program.

## Scope and data boundaries

The first 28 rows are fresh authored post-checkpoint probes, with correlated
name/argument/language views. Six operation families are running maximum,
adjacent differences, stable uniqueness, parentheses balance, left rotation and
run counting. They are not 28 statistically independent programming problems.
No claim is made that these families are absent from every historical training
source. The last 12 rows are explicitly exposed historical regressions.

The first readout consumes the fresh status of the 28 new rows. This entire
suite is now an exposed regression set. Do not tune on it and then describe a
later readout as independent generalization. Reference source and expected
values are used only after generation for control/validation, never as prompts,
assistant prefixes, decoder input or fallback code.

## Exact identities and permanent raw evidence

- Checkpoint: `f8ef88dccbbd3280c21e5acd51c41c6744e6ba597aa63686f3f43fd96beac229` from `model-factorial-v1`.
- Authored head: `8e2dac4`; actual PR merge tested by Actions: `36bd46630a795db9203f59ce9601883857ab2fce`, recorded verbatim in the raw report.
- [Frozen manifest](../evaluation/released-generalization-v1/manifest.json) records the exact task-byte hash, two inference profiles and one attempt per row.
- [Unmodified CI report](evidence/released-generalization-v1/report.json): SHA-256 `b5c4afa2fd960c8fca3a7e1692abaed68378123ccc9dc0716ba1dbececfb891b`.
- [Evidence manifest](evidence/released-generalization-v1/manifest.json) retains source, run, artifact and archive identities, independently of artifact expiration.
- [Original artifact](https://github.com/deser1/NppAI/actions/runs/40423348120/artifacts/12698397769): ZIP SHA-256 `d60366912b8dc2714f45b206b0639676a67fa1b89de273edcaa6b7b2870eb27f`.

The report includes actual returned source, output hashes, cases identified by
the frozen suite, Docker diagnostics, generation times, source hashes, runtime
versions, model identity and weight-retention checks. No model weights are
committed. The report is retained in git to avoid dependence on expiring Actions
artifacts.

## Reproduce

Download the exact model from its release. Pull an appropriate trusted Python
3.12 image into local Docker, then resolve its immutable local image ID:

```bash
docker pull python:3.12-slim
image_id=$(docker image inspect python:3.12-slim --format '{{.Id}}')
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/evaluate_released_model.py \
  --model NppAI-factorial.nppai --output measured --image "$image_id"
python scripts/evaluate_released_model.py \
  --model NppAI-factorial.nppai --output measured --audit measured/report.json
```

The evaluator rejects changed checkpoint/task bytes and audits completeness,
score-to-source binding, controls, unchanged weights and model identities. Without
`--image`, it records readout/AST evidence only, with functional status **null**;
it does not turn unavailable execution into success or model failure.
The Docker image ID is recorded; future pulls may resolve different bytes.
Generated source is never imported or executed on the host.

## Next decision

Do not promote this checkpoint as a general-purpose coding model. Continue
training-only work on complete value-returning bodies, variable binding and
operation choice, with replay of original correct responses and separate
reviewed development data. Freeze the next candidate before a fresh independent
functional measurement. The staged path still needs its own successful
functional and retention evidence before plugin integration. Interactive Windows
validation remains a separate [stable-release gate](stable-readiness.md).
