# Identifier copying experiment v3

The v2 model often produces a training function name instead of the name in the
request. V3 removes the operation family from the identifier: all functions use
`fn_` followed by six deterministic random lowercase letters or digits. There
are 512 distinct training names (three prompt styles each), 16 validation names
and 28 test names. Names, prompts and complete reference answers do not overlap
across splits. Every test-name character occurs in training names. V1 and V2
remain unchanged.

Two arms use the same frozen V3 data, seed 42, model dimensions 64/128, two
layers, context 256, batch size four, 3000 updates and AdamW learning rate 0.001.
The only difference is response-identifier loss weight: 1 or 4. Prompt tokens
and padding are excluded; EOS and all remaining response bytes have weight 1.
Loss is normalized by the sum of active weights. Evaluation response loss is
always unweighted so the two arms can be compared.

Both settings and the dataset were fixed before evaluating either arm. Free
greedy generation supplies no signature prefix. Correct requested name, complete
reference AST, reference AST after ignoring the name, exact text and nonempty
responses are distinct metrics. The reference AST comparison includes parameters
and body. Generated code is parsed but never executed.

## Reproduce

```sh
python -m unittest tests.test_instruction_dataset tests.test_identifier_training
python scripts/train_broader_instruction_experiment.py \
  --dataset evaluation/instruction-v3 --output copy-weight1 \
  --steps 3000 --identifier-weight 1
python scripts/train_broader_instruction_experiment.py \
  --dataset evaluation/instruction-v3 --output copy-weight4 \
  --steps 3000 --identifier-weight 4
```

The identifier-copy workflow preserves raw responses, manifests, losses and
exported checkpoints for both arms. Each checkpoint has its own SHA-256.
Successful training jobs mean the measurement completed, not that model quality
passed. The old 300-step checkpoint is not modified or replaced automatically.

## Limits

This is a small synthetic experiment, with correlated operation templates.
Although four families are absent from this training split, those task patterns
were already exposed by earlier V1/V2 experiments. V3 tests new identifiers on
exposed operation families; it is not a pristine test of general programming.
The paired arms isolate loss weighting on V3. Comparison with V2 also changes
name distribution and training-set size, so it cannot isolate either effect.
These reports measure PyTorch FP32; native INT8 results require separate checks.

## Measured local result, 13 October 2026

| Free-generation test metric | Name weight 1 | Name weight 4 |
| --- | ---: | ---: |
| Nonempty outputs | 28/28 | 28/28 |
| Any syntactically valid function | 17/28 | 9/28 |
| Correct requested function name | 0/28 | 0/28 |
| Complete reference AST | 0/28 | 0/28 |
| Reference AST ignoring function name | 13/28 | 4/28 |
| Exact response | 0/28 | 0/28 |
| Unweighted test response loss | 2.10365 | 2.11783 |

Increasing the name loss weight did not repair copying and reduced syntax on
this run. Neither checkpoint is promoted to the default model or a working code
demo. The default loss weight stays 1. This is a negative result from one seed;
it does not prove that loss weighting can never help.

Raw outputs, hashes and measured runtime are preserved in
[weight 1](identifier-copy-weight1-results.json) and
[weight 4](identifier-copy-weight4-results.json). Names are not counted as correct
when the generated function merely resembles the requested operation. Prefix
scaffolding is deliberately excluded from this comparison.
