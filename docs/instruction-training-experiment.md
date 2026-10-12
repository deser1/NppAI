# Instruction training experiment: 12 October 2026

Measured results are preserved in [instruction-experiment-results.json](instruction-experiment-results.json).
Run `python scripts/train_instruction_experiment.py --output experiment --steps 2000`
to reproduce the new checkpoint. Add `--baseline <previous-model.nppai>` to compare
the old model on the same tasks. GitHub Actions publishes the checkpoint and report.
The pinned old training artifact is subject to GitHub's retention limit; preserve
the checkpoint with SHA-256 `9735e3545541b2c6115ba4eead8284c6c1606d93e9558cd054799d9f14846995`
to repeat that comparison after its artifact expires.

## Design

The old checkpoint learns next-token prediction on raw repository files. The new
experiment uses eight explicit instruction/answer pairs and response-only loss,
including an EOS target. Three entire tasks are excluded from training.
The new model is trained from scratch for 2000 steps, seed 42, with context 192.
The old model uses 300 steps and context 64, so multiple factors differ.
This is a diagnostic comparison, not an isolated causal ablation.

## Actual results (FP32 greedy decoding)

| Measurement | Repository baseline | Instruction experiment |
| --- | ---: | ---: |
| Non-whitespace responses on unseen tasks | 0/3 | 3/3 |
| Syntactically valid functions on unseen tasks | 0/3 | 0/3 |
| Exact reconstruction of instruction training examples | Not measured | 8/8 |

The new training loss fell from 5.75577 to 0.000593. Despite that, all held-out
outputs were malformed. This is evidence of memorizing this tiny training set
without useful generalization to the measured new tasks. It does not establish
functional correctness or useful code generation. Generated code was parsed,
never executed. The new checkpoint should not be presented as a working demo.

## Next experiment

Keep these exposed tasks as regression cases. Create a new frozen, unseen test
set before further tuning. Collect a substantially broader licensed instruction
dataset with varied function names, prompt styles and code patterns; split by task
family and source rather than random tokens. Measure held-out loss, syntax and
functional correctness in an isolated evaluator. Compare training and held-out
metrics to detect memorization before claiming improvement.
