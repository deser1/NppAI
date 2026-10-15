# Matched per-draw function-name training

The frozen local smoke failed all six requested function contracts. Prior alias
experiments assigned stable random names to each training function. This bounded
experiment tests a different training signal: a new function-name alias for each
renamed example draw, compared with stable aliases at the same update budget.
It tests a hypothesis; adding the training option does not establish improvement.

Both arms start from the full-response continuation checkpoint
`161d6303c7e77f005cbabba5a1790b9d6a6dfc1812c121bfbb3f1cb6820e5b1a`
from [training run 38702282716](https://github.com/deser1/NppAI/actions/runs/38702282716).
The exact source report and loaded model identity are checked using the pinned
smoke provenance. Existing production defaults and checkpoints remain unchanged.

Both use 1024 AdamW updates, seed 42, learning rate 0.001, batch size 4, gradient
clip 1, fresh optimizer state, FP32 CPU PyTorch with two threads and context 256.
Each batch contains two original and two renamed draws from the frozen training
split with four wording forms. The sampling/shuffle stream is identical. All
response bytes, including the body and EOS, are supervised equally; prompt and
padding targets are ignored. There is no loss reweighting or supplied header.

| Arm | Renamed function names |
|---|---|
| fixed | One deterministic `nc_` plus eight hex characters per source function |
| resampled | New deterministic `nc_` plus eight hex characters for every renamed draw |

All renamed names have the same byte length in both arms. Training source IDs,
batch order, target-byte totals, draw counts, unique aliases and concrete training
content hashes are recorded. Resampling has exactly 2048 unique aliases, checked
for collisions. Name diversity/repetition and which bytes are predicted change;
source examples and budgets do not. Function-name replacement uses Python NAME
tokens in answers and whole-word substitution in prompts. Strings, parameter
names and other body tokens remain intact. Oversized references are rejected.
The training loader opens only `train.json`; frozen SHA-256 values are pinned in
the implementation. Validation/test references never enter optimization.

Before/after greedy full outputs (up to 160 new bytes, no assistant prefix) are
recorded for 16 validation and 28 test prompts, both original and independently
renamed: 88 correlated development views. Evaluation aliases use a disjoint
`ev_` namespace. These source splits are already exposed development data, not a
new independent holdout. The six smoke probes and fresh-probe-v1 are not loaded.

Metrics distinguish single-function syntax, requested function name, full declared
signature, reference AST and exact full response. Name/signature counts require
the complete output to parse as one function. Exact/AST equality is stricter than
semantic equivalence and is not functional execution accuracy. Reports also show
retained/lost/gained signature and exact-response hits, including original tasks,
so renamed improvements cannot hide regressions. Generated source is parsed only.

Exported checkpoint tensors must equal training tensors before after-scoring.
The evidence audit reconstructs data and the full training schedule, rescores raw
outputs and rejects changed provenance, missing/reordered cases or altered counts.
Both raw reports and experimental checkpoints are uploaded for 14 days by
**Matched resampled function-name training**. A successful workflow means the
experiment completed, even if neither arm improves. No checkpoint is selected or
promoted automatically; one seed and one short continuation cannot establish a
robust training policy or solve arbitrary argument binding.

Implementation: `scripts/resampled_name_training.py`; regression tests:
`tests/test_resampled_name_training.py`.

## Observed outcome

Pending the matched training run. Report both arms and original-task retention,
not just the largest count. A later independent evaluation is required before
claiming improved autonomous coding quality.
