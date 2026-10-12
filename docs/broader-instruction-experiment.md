# Frozen synthetic instruction experiment v1

The project-authored dataset is frozen in `evaluation/instruction-v1/` with
SHA-256 hashes for every split. It contains 576 training prompt/answer records
from 16 template families, 16 validation records and 28 test records. Twelve
test records belong to four families absent from training: sorting, ordered
deduplication, joining strings and clamping. The other sixteen test tasks use
known families with new function identifiers. Prompts, identifiers and answers
do not cross splits. Variations of a template are correlated; 576 records do
not represent 576 independently designed tasks. Names also carry distinct
split prefixes, so this benchmark includes identifier distribution shift.

## Fixed experiment and results

Train from scratch for 3000 steps, seed 42, batch size 4, context 256, two layers,
64-dimensional embeddings. Loss is computed on response bytes and EOS only.
Validation and test files are first read after all optimizer steps finish.
No tuning or checkpoint selection uses their results.

Measured locally on 12 October 2026:

| Metric | Validation | Test |
| --- | ---: | ---: |
| Tasks | 16 | 28 |
| Non-whitespace outputs | 16 | 27 |
| Syntactically valid requested functions | 0 | 0 |
| Exact reference matches | 0 | 0 |
| Mean response loss | 4.79585 | 4.27602 |

Final training batch loss was 0.05047. The much higher held-out response loss
and malformed outputs indicate poor generalization despite the larger synthetic
training set. No functional correctness or useful coding ability is established.
The former 3-task experiment and this 28-task benchmark have different tasks;
their counts should not be treated as a direct before/after comparison.

Checkpoint SHA-256:
`88a6b3b40b85ab50a1ef384a381771ac60fc9b69b6355f5390ca9fdd9be175d6`
(verify the actual run report for the authoritative checkpoint hash).

## Reproduce

```sh
python -m unittest tests.test_instruction_dataset
python scripts/train_broader_instruction_experiment.py \
  --dataset evaluation/instruction-v1 --output broader-experiment --steps 3000
```

The `Frozen broader instruction experiment` workflow publishes the checkpoint
and raw outputs. Green workflow completion means the experiment ran, not that
model quality passed. Generated text is AST-parsed and never executed.
This exposed v1 test becomes regression data for future work; create and freeze
a fresh unseen test set before tuning. Broader varied real instruction data and
task-family separation are still needed; increasing template count alone did
not resolve this measured generalization failure.
