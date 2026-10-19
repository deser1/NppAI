# Coding with actual fallible own-model planner artifacts

This paired experiment uses only the existing 96 authored training records.
Its frozen planner is checkpoint
`4859ad69e93ff2ca80d8951bca617476d118f8ef127abc4b313b8d2f0671c6fd`.
Actual planner requests/responses come from the previously retained, hashed
training trace. They are input context, never treated as correct target plans.
Code targets remain the matching authored training answers.

Of 96 records, **83** have a complete bounded observed specification and plan.
The remaining **13** use explicitly counted authored fallback inputs because the
planner did not reach the code call. No validation/test cases enter optimization.

Both coders initialize from the same own-model weights and receive the same
seed43 base indices, 512 updates, batch size four, 50% direct-answer replay and
50% code objectives. The control conditions code on authored priors; the other
uses observed fallible priors where available. Different context lengths mean
equal updates do not equal compute or token exposure. Inference uses the same
frozen planner in both arms. Per-case direct/code retention and all planner calls
are checked, not just total scores.

The existing exposed validation and bilingual sets are read after both fits.
They remain regressions, not a fresh independent holdout. Docker execution must
pass all authored controls and reject a known bad program before scoring actual
generated outputs. Without Docker the report says `not_run`. No checkpoint is
promoted by this script. Results, losses, traces, hashes and failures are uploaded
by `Coder training on observed fallible plans`.

```sh
python scripts/fallible_planner_training.py --archive source.zip --output measured
```

Use the exact pinned source artifact, and add `--image sha256:VALIDATOR_IMAGE`
for isolated functional execution. Missing sources or changed hashes fail closed.

## Local measurement, 19 October 2026

| Exact training replies, 96 records | Authored priors | Observed priors |
| --- | ---: | ---: |
| Direct generation | 96 | 96 |
| Autonomous frozen-planner pipeline | 10 | 32 |
| Code with authored specification/plan | 86 | 18 |

Both preserve all 88 original direct hits, both original autonomous hits and
every frozen planner call. However, the observed-prior coder retains only 6/16
original assisted-code hits, losing ten. The control retains all 16.
Both arms have 0/16 validation and 0/12 bilingual-regression exact responses,
directly and autonomously. The control checkpoint exactly reproduces the prior
measured coder hash on this local allocation. This is a narrow training-context
adaptation with a regression, not new-task improvement. There is no promotion.
Local functional execution is `not_run`; the dedicated CI job measures it in Docker.
