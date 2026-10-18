# Own-model staged-protocol training pilot

This trains the project's own dense NppAI model. It does not use an external
language model or claim to expose a network's internal reasoning. The generated
specification and plan are short, fallible task artifacts. The production plugin
continues to use its existing validated generation path.

## Frozen pilot design

The [dataset manifest](../evaluation/staged-protocol-v1/manifest.json) fixes the
recipe and input hashes. `build_stage_dataset.py` derives **96 training records**
from the existing frozen v2 **train split only**: the first two function names per
16 families, with original wording variants. Each receives a project-authored
specification and concise operation plan. These are supervised targets; generated
plans are not automatically treated as correct training data.

The shared `stage_values` formatter makes training prompts byte-identical to the
specification/plan/code inference protocol. Training uses reviewed prior artifacts
for plan/code examples; evaluation uses the model's own raw earlier outputs.
This teacher-forcing gap is a limitation, not assisted inference success.

Both arms initialize from dense-v2 checkpoint SHA-256
`6a49081e1d94e85d31efe6bdf2a4e3d9e4d7b06bb381c492de3a98d1300ed9ac`
(source artifact `11775752592`, run `38314121142`). Context expands **256 → 1024
bytes**: existing positions are copied, remaining positions initialized once with
seed42. Both arms receive the identical expanded state. RMSNorm epsilon is set to
`1e-5` to match the production dense NumPy/native runtime; this is another change
relative to historical PyTorch measurements.

| Property | Direct control | Staged mixture |
| --- | --- | --- |
| Updates | 1024 | 1024 |
| Examples per update | 4 | 4 |
| Base task schedule | Same seeded schedule | Same seeded schedule |
| Objectives per update | Four direct code examples | One direct, specification, plan and code example |
| Optimizer | AdamW, learning rate 0.001, gradient clip 1 | Same |
| Context | 1024 bytes | 1024 bytes |

Assignment of the four stage objectives rotates across base tasks. Equal updates
and base-task schedules do **not** equalize target tokens, direct-code exposure,
input lengths, time or compute. Both actual token counts and full per-update
loss/task/stage traces are retained. There is one seed, no result-driven checkpoint
selection, and no checkpoint promotion.

## Evaluation

After both optimizations complete, read and score:

- **16 training probes**, one first record per family: memorization and direct
  original-hit retention; explicitly not held-out quality.
- **28 exposed v2 regression tasks**, unchanged from prior work.
- **12 newly frozen synthetic prompt records**: six tasks, each in English and
  Polish, with new function names and authored functional cases. Four task families
  appear in this pilot's training; factorial and positive-only summation do not.
  All operation families have prior project exposure. Wording pairs are correlated;
  this is not a broad independent programming benchmark.

For every model/group, retain the complete autonomous staged trace and both direct
controls (128 final-code tokens and 384 aggregate-budget tokens). References never
enter evaluation model requests. Empty/stopped stages remain failures. Finishing
all three stages does not validate their meaning or establish correct code.

Export both models to production-compatible dense-v2 `.nppai` files. Compare four
fixed training requests in PyTorch FP32 and the exported production NumPy runtime,
retaining exact raw readouts. This bounded check does not certify native INT8 or
all possible prompts. Verify evaluation leaves all parameters unchanged.

The dedicated Actions workflow executes generated new-suite candidates with the
existing restricted Docker validator. It first requires all authored reference
controls to pass and a known incorrect candidate to fail. Empty candidates remain
failures; infrastructure errors abort the experiment. No generated source is ever
executed on the host. The local training measurement has `execution_status:
not_run` because Docker is unavailable there.

## Reproduce

```sh
python scripts/build_stage_dataset.py
python -m unittest discover -s tests -p 'test_stage_training*.py' -v
python scripts/train_staged_protocol.py \
  --initial-model path/to/frozen-dense-v2.nppai \
  --output measured --image sha256:YOUR_LOCAL_VALIDATOR_IMAGE_ID
```

Use the `.github/workflows/staged-protocol-training.yml` runtime pins for CI replay.
Omitting `--image` explicitly skips functional execution and records `not_run`.
Models, traces, reports, raw runtime comparisons and Docker diagnostics are uploaded
as the `nppai-staged-protocol-pilot` Actions artifact. Treat artifact retention as
finite; preserve selected files separately before expiration if needed.

## First local observation

Both arms completed all 1024 updates. Direct generation reconstructs **16/16**
training probes in the direct-control model and **14/16** in the staged-mixture
model. The expanded initialization had **10/16** exact training probes; both
arms retain all ten. Thus 14 versus 16 is a difference between trained arms,
not loss of two initial exact hits. On new tasks, staged generation reaches the end in **8/12** prompts for the
mixture model, versus **0/12** for the direct control, but **neither produces a
syntactically valid function or exact answer in that staged arm**. Both models
remain **0/12 exact new-task replies** in both direct controls as well.

This is a protocol-training experiment with failed coding-quality gates. A model
that emits more stages is not thereby a better coding model. Do not replace the
plugin's checkpoint or enable staged generation by default on these results.
Next work should isolate stage/output-format confusion, direct-code retention and
identifier transfer on development/training diagnostics before another fixed pilot.

Both full local training runs produced byte-identical exported checkpoints per
arm. All four retained PyTorch/production-NumPy readouts matched per arm. These
are same-host observations and a bounded readout check, not cross-host or native
INT8 certification. Raw local records and hashes are retained with this pilot.

Local records: [archive](evidence/staged-protocol/local-records.zip) and
[member/checkpoint hashes](evidence/staged-protocol/local-source.json). Extract the
archive and run `python scripts/audit_staged_pilot.py PATH/local1` for an offline
raw-score, stage-input, denominator and paired-schedule reconstruction. Weights
are supplied by the dedicated Actions artifact, not this evidence archive.
