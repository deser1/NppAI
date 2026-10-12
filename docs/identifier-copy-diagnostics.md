# Identifier copying diagnosis

The paired V3 experiment produced zero correct requested names in free
generation. A higher name-loss weight did not repair that result. This diagnostic
adds measurements to each reloaded checkpoint without changing training,
datasets, decoding defaults or the deployed model.

The workflow trains the same fixed 3000-step weight-1 and weight-4 arms, reloads
each exported FP32 checkpoint and publishes `identifier-diagnostics.json` in
each arm's artifact. It measures all 16 validation and 28 test records, plus a
fixed first-16 training-record sample (seen data, not a held-out benchmark).

| Measurement | Information supplied to the model | Meaning |
| --- | --- | --- |
| Teacher-forced identifier bytes | Full prompt and all preceding reference bytes at every position | Per-byte argmax and target probability with no accumulated generation mistakes |
| Free header name | Prompt only | Autonomous header copying, even if the body is invalid |
| Def-only header name | Prompt plus `def ` | Header copying when only the declaration opener is supplied; requested name is never inserted |

Teacher-forced all-bytes-correct counts cannot be called generated-name success:
each byte is predicted under its own correct reference prefix. Header-only
metrics cannot be called syntax, body correctness or complete task success.
Existing AST and exact-response metrics remain in `broader-report.json`.

Raw reports preserve expected and predicted bytes, target probabilities,
first wrong byte, both generated responses, dataset manifest, model hash and
runtime. UTF-8 byte offsets are checked with a non-ASCII prompt. Oversized
references, embedded NUL and inconsistent names fail instead of truncating.
Generated source is never executed.

```sh
python -m unittest tests.test_identifier_diagnostics
python scripts/diagnose_identifier_copy.py \
  --model identifier-experiment/NppAI-broader-instruction.nppai \
  --dataset evaluation/instruction-v3 \
  --output identifier-experiment/identifier-diagnostics.json
```

If teacher-forced copying also fails, generation drift alone cannot explain
the failure. If it succeeds while def-only copying fails, the observations are
consistent with accumulated mistakes during autoregressive copying. If def-only
helps relative to free generation, the declaration opener contributes to the
failure. These observations narrow further experiments; they do not establish
a causal architecture defect or prove a remedy.

The same previously exposed synthetic operation families and single seed are
used. No test-guided parameter tuning is performed. Native INT8 is not measured.
CI success means the diagnostic ran, not that copying improved.
