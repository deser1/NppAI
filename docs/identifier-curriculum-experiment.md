# Copying pretraining followed by code training

The fixed-format echo task copied 28/28 new names with context 96; this does not
prove copying will survive training on code tasks. This experiment uses three
preselected schedules and the same frozen V3 splits, model dimensions and seed.

| Arm | Echo updates | Code updates | Total updates |
| --- | ---: | ---: | ---: |
| code3000 | 0 | 3000 | 3000 |
| code6000 | 0 | 6000 | 6000 |
| echo3000-code3000 | 3000 | 3000 | 6000 |

All models use context 256, dimension 64/hidden 128, two layers, byte vocabulary,
batch four, seed 42, AdamW 0.001, clip 1 and identifier weight 1. The copying
stage derives one fixed-format echo example per training name (512 examples)
and never reads held-out splits. The code stage uses all 1536 frozen training
records. No position embeddings are resized or fabricated: initialize from a
full compatible context-256 dense checkpoint or reject it. A fresh AdamW is
used after copying; scratch code6000 has a continuous optimizer. Default
training without an initial checkpoint remains unchanged.

The two 6000-update schedules have equal optimizer-step counts, not equal
tokens or compute: echo sequences are shorter. They also differ in objective,
per-example exposure, sampled batches and optimizer reset. Outcomes cannot
isolate one of those changes. The 3000-step arm is a simultaneous baseline.
Budgets and schedules are fixed before reading held-out outputs; there is no
early stopping, test-guided tuning or best-checkpoint selection.

Preserve both stage checkpoints and raw reports, initialization/source/model
hashes, runtime and train-first24 reconstruction. Evaluate final exported,
reloaded code checkpoints with autonomous syntax/requested name, complete
reference AST and exact response, plus separate reference-prefix/free/def-only
identifier diagnostics. Teacher forcing and forced declaration openers do not
count as autonomous complete task success. Source is AST-parsed, never executed.

```sh
python scripts/train_identifier_echo.py --dataset evaluation/instruction-v3 \
  --output echo-stage --steps 3000 --context 256 --training-only
python scripts/train_broader_instruction_experiment.py \
  --dataset evaluation/instruction-v3 --output code-stage --steps 3000 \
  --identifier-weight 1 --training-names 512 \
  --initial-model echo-stage/NppAI-identifier-echo.nppai
```

One seed and exposed synthetic operation families remain limits. No native
INT8 or general programming claim, model promotion or production change.
