# Bounded identifier memorization probe

PR #146 observed zero completely correct identifiers even under reference
prefixes in both V3 loss-weight arms. This next fixed experiment compares eight
training names (all their prompt styles) with all 512 V3 training names. Both
arms use 3000 steps, seed 42, batch size four, name weight 1, the same model,
optimizer, context and frozen validation/test splits. The eight names are the
first eight distinct names in training-file order, selected without reading
validation or test records. Every selected name's records retain original order.

The experiment changes training-set size and therefore exposure per example
and sampled batch order. It does not isolate architecture or prove that more
steps fix copying. One small-subset seed is a memorization sanity check, not a
general programming benchmark. No test result selects settings or early stops.

`broader-report.json` adds selection IDs, count and a canonical selection hash,
plus free-generation/AST metrics on the first 24 selected training records.
In the eight-name arm those 24 records cover the selected training set; they
are seen data. In the 512-name arm this is a fixed training sample. Frozen source
files and their manifest hashes remain unchanged. The source manifest describes
the source corpus; `training_selection` describes the actual optimization set.

The reloaded-checkpoint diagnostic compares teacher-forced identifier bytes,
autonomous names and def-only names on training first16, validation and test.
These measures are distinct from complete AST and exact response correctness.
Generated source is parsed, never executed. Default training without the new
option is unchanged and does not perform the additional training probe.

```sh
python -m unittest tests.test_identifier_subset tests.test_identifier_diagnostics
python scripts/train_broader_instruction_experiment.py \
  --dataset evaluation/instruction-v3 --output memorization-8 \
  --steps 3000 --identifier-weight 1 --training-names 8
```

If known-name reconstruction succeeds but new names still fail, the observation
supports memorization without demonstrated general copying. If even this
bounded probe fails, optimization or capacity deserves further controlled
testing; the result alone cannot identify which. Neither checkpoint is promoted
to the plugin. CI success means measurement completed, not model-quality success.
