# Identifier distribution experiment v2

Dataset v1 used `train_`, `validation_` and `test_` function-name prefixes.
These split-specific words and different lengths introduced distribution shift.
Dataset v2 uses the same `fn_<family>_<two digits>` format everywhere, with
disjoint identifiers: 00..11 for training, 12 for validation, 13 for known-family
tests. Four wholly unseen families remain in the test set. Counts, architecture,
seed and budget of 3000 steps are unchanged. All new split hashes were frozen
before training. v1 remains unchanged as a regression benchmark.

## Measured local result, 12 October 2026

| Test metric | v1 | v2 |
| --- | ---: | ---: |
| Nonempty outputs | 27/28 | 26/28 |
| Any syntactically valid function | 0/28 | 15/28 |
| Correct requested function name | 0/28 | 0/28 |
| Exact reference response | 0/28 | 0/28 |
| Mean held-out response loss | 4.27602 | 2.41579 |

All 15 syntactically valid v2 outputs match the reference AST after ignoring
only the top-level function name. This is a diagnostic metric, not a passed task
or functional execution test. For example, the requested `fn_add_13` produced:

```python
def fn_add_01(a, b):
    return a + b
```

The model learned several known operation patterns but still reuses training
identifiers instead of copying the requested unseen name. Unseen families have
no successful requested functions. The naming change is associated with better
syntax on this narrow synthetic benchmark; it does not establish general coding
competence. The full local report is [preserved here](broader-instruction-v2-results.json).
Checkpoint SHA-256:
`65a4a07ffb473cd0ee58573ef6df4afeb26bacd274916750f0b79054498af808`.

## Reproduce

```sh
python -m unittest tests.test_instruction_dataset
python scripts/train_broader_instruction_experiment.py \
  --dataset evaluation/instruction-v2 --output broader-v2-experiment --steps 3000
```

The workflow runs v1 and v2 as separate matrix jobs and preserves both checkpoints
and reports. `ast_match_ignoring_function_name` explicitly remains separate from
requested-name syntax and exact matching. Generated code is never executed.
Dependency/platform variation can change numerical losses; use each run's report
and model hash as its evidence. These exposed tests become regression data for
future tuning. Identifier-copying and generalization to unseen task families remain
open model-quality problems.
