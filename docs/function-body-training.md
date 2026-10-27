# Paired function-body training pilot

This fixed experiment follows the released model's broader **1/40** result.
The goal is complete value-returning functions, argument binding and operation
choice rather than more header-only replies. This is a development experiment,
not a general-purpose model release.

## Frozen design, before optimization

- Initialization: permanent `model-factorial-v1`, SHA-256
  `f8ef88dccbbd3280c21e5acd51c41c6744e6ba597aa63686f3f43fd96beac229`.
- Authored training bank: **144** English/Polish records over six families:
  absolute difference, counting negatives, sum of squares, reversed text,
  all-positive predicate and zero clamping. Each family has three function
  names, two argument views and four descriptions.
- Exposed development: **24** same-family records with two other names, other
  arguments and different English/Polish descriptions. Names, arguments and
  wording change jointly, so this is not an isolated test of any one factor.
  These rows are never optimizer targets. The original 40-task evaluation is
  not training data and is not used to select this recipe.
- One seed (83), **1024 updates per arm**, AdamW at 0.0003, betas 0.9/0.999,
  epsilon 1e-8, weight decay 0.01, batch 8, gradient norm clipped to 1.0.
  Completion-only cross entropy is averaged per example, including EOS. No
  task/reference truncation is permitted. The aligned format has a 256-byte
  prompt boundary; original direct replay retains its legacy format.
- Replay control: four of the original 96 direct records and four of the 532
  factorial training records per batch. Diverse arm: two originals, two
  factorial records and four new body records. The budget and shared replay
  draws are matched; adding new data also reduces replay frequency, which is
  an explicit tradeoff rather than a perfectly isolated data-only intervention.
- Both candidates are exported and hashed **before** development readout.
  Neither optimizer reads development results and there is no early stopping,
  seed search, checkpoint picking or automatic production routing.

[Manifest](../evaluation/function-body-v1/manifest.json),
[training](../evaluation/function-body-v1/train.json) and
[development](../evaluation/function-body-v1/development.json) retain exact bytes.
The trainer records all draw indices, per-update losses, runtime/source hashes
and both exported candidate identities. No external model is used.

## Readout and gate

The released production profile (`--prompt-bytes 256 --python-functions`) reads
all three models: unchanged source, replay control and diverse-body candidate.
Reference source and test answers never enter inference. It may mechanically
supply an explicitly requested function header, so signature matches are not
learned copying. Body/value-return/functional metrics are reported separately.
A return statement is only a structural observation; even `return None` can
satisfy it and still fail every configured test.

- Development: all 24 actual outputs, exact/AST/signature/value-return counts,
  actual restricted-Docker tests, elapsed generation time and diagnostics.
- Training memorization: 12 preselected English/Polish probes, not all 144.
- Original direct retention: all 96, preserving each previous exact hit rather
  than permitting equal totals to conceal lost and gained rows.
- Factorial retention: all 22 exposed production probes, preserving every
  functionally successful source row, including the familiar Polish request.
- Positive Docker controls: all 168 authored training/development programs.
  Negative controls: a constant-None function for each of the 24 development rows.
- Candidate, initialization and source identities must remain unchanged during
  readout. Infrastructure failures abort measurement; unavailable Docker is
  never reported as a functional success.

Eligibility for a later fresh evaluation requires **at least 12/24** development
functional passes, strict improvement over both source and replay control, all
source original exact hits retained, and every source factorial functional hit
retained. Quality failure is a recorded rejected experiment, not a failed
infrastructure workflow. Passing this gate still does **not** promote weights.
The next independent suite is authored only after an eligible candidate is
frozen; a rejected candidate does not consume that independent measurement.

## Run

The workflow `Paired function-body training` installs the pinned CPU runtime,
downloads the pinned initialization, freezes both optimized candidates, then
executes controls and actual generations in restricted Linux Docker. It retains
models, traces, schedule and raw report as an Actions artifact. No release/tag
or default-model update is performed.

```bash
python scripts/function_body_training.py --phase train \
  --source NppAI-factorial.nppai --output measured
python scripts/function_body_training.py --phase measure \
  --source NppAI-factorial.nppai --output measured --image "$NPPAI_VALIDATION_IMAGE"
python scripts/function_body_training.py --phase audit \
  --source NppAI-factorial.nppai --output measured
```

`NPPAI_VALIDATION_IMAGE` must be the immutable ID of a locally available trusted
Python 3.12 Docker image. Without `--image`, only readout/structural evidence is
recorded and gate status remains `not_measured`. Generated code never executes
on the host. Results will be added after the frozen run, including failures.

## First measured result

[Run 40426278674](https://github.com/deser1/NppAI/actions/runs/40426278674)
executed the frozen recipe and actual restricted Docker tests. All 168 authored
reference controls passed; all 24 negative controls failed. The audit passed.

| Measurement | Source | Replay control | Diverse bodies |
| --- | --- | --- | --- |
| Development functional | 0/24 | 0/24 | 0/24 |
| Fixed training probe exact | 0/12 | 0/12 | 12/12 |
| Original direct exact | 96/96 | 96/96 | 96/96 |
| Exposed factorial functional | 20/22 | 20/22 | 18/22 |

The diverse arm learned the 12 measured training probes but did not transfer to
jointly changed names, arguments and wording. It also lost both production
`policz_silnie` rows (plain and signature-assisted). The declared gate therefore
**rejects** this candidate. No weights or decoder are promoted and no fresh
independent suite is consumed. Rejected weights do not enter plugin routing.

[Evidence manifest](evidence/function-body-v1/manifest.json) and
[raw metadata, schedules and traces](evidence/function-body-v1/pilot.zip) preserve
the unmodified CI JSON bytes in git. The original Actions artifact additionally
contains both exported models. These rejected candidate binaries remain research
artifacts with finite retention; source initialization is a permanent release.

A separate local run used the same recipe and produced the same structural
counts and original/factorial exact counts, but **different model hashes**.
[Local-only summary](evidence/function-body-v1/local-readout-summary.json) retains
that fact; it does not assert Docker functional results. This is not cross-host
bitwise training certification. CI identities remain the primary functional evidence.

## Next diagnosis: exposed orthogonal transfer grid

The initial development changes three factors together. A post-hoc diagnostic
now holds the task family fixed and crosses known/new function name, known/new
arguments and known/new wording independently: eight cells × six families ×
English/Polish = **96 probes per model**. It uses one known/development alias pair
per family. The all-known cell reproduces 12 training requests exactly.

All three fixed models use the unchanged production decoding profile. Each grid
probe has a restricted-Docker positive and negative control. Authored reference
ASTs are renamed only to construct controls; generated model code is never
rewritten. Every actual output, timing, failure and cell count is retained.
This is exposed-data diagnosis, not a fresh benchmark, model selection or proof
that any unmeasured family is supported. It will identify which isolated input
changes destroy the learned bodies before another training recipe is chosen.
