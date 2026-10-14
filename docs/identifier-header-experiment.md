# Header-only learning from original code prompts

Fixed-format echo succeeds, but echo-to-code pretraining did not repair code
names. This paired probe preserves all original V3 prompts, record ordering,
styles and name splits. Only the training response changes: the header arm
receives the first LF-terminated `def name(parameters):` line, while the full-code
control receives the original complete reference. Neither receives an assistant
prefix during autonomous evaluation.

Both arms use 6000 updates, seed 42, context 256, dimension 64/hidden 128, two
layers, byte vocabulary, batch four, AdamW 0.001 and clip 1. Both use all 1536
training records (512 names, three styles). With the same record order and
Python RNG seed, the sampling schedule uses the same source record indices.
Shortened response sequences and supervised targets change token/compute
budgets and gradients; equal update counts do not imply equal compute.

Frozen source files remain unchanged. Preserve the source manifest, derived
training hash, raw prompts/headers/outputs, runtime, loss and checkpoint hash.
Held-out splits are read only after the final update. Evaluate 16 validation
and 28 test tasks and a fixed first24 seen-training sample. The header arm
reports autonomous requested name, exact header including parameters/LF and
independent reference-prefix identifier bytes. Header-only output is not a
complete syntactically valid function or functional code. The full-code arm
keeps existing complete AST/exact response metrics and header diagnostics.

The header checkpoint is exported/reloaded and all reported autonomous outputs
must match the pre-export model. No generated source is executed. Settings and
budgets are fixed before results; there is no held-out tuning or checkpoint
promotion. One seed, synthetic/exposed operation families, fixed nine-byte names
and FP32 remain limits. The result can narrow the next experiment; it cannot
establish a universal architecture or optimization claim.

```sh
python -m unittest tests.test_identifier_headers
python scripts/train_identifier_headers.py --dataset evaluation/instruction-v3 \
  --output header-experiment --steps 6000
```
