# Task-signature body wording diagnostic

PR #161's frozen learned-header pipeline has 13/28 original exact test responses
but 0/28 on both novel wording probes in FP32/native INT8. This inference-only
diagnostic asks whether failure persists when the body model receives the plain
signature explicitly present in the task. It is assisted generation, not a
remedy for autonomous header copying.

Use the fixed `nppai-native-staged-comparison` artifact from run
[38591181871](https://github.com/deser1/NppAI/actions/runs/38591181871).
Verify both cached report SHA-256 values, raw-output metrics against frozen
manifests/cases, zero source infrastructure errors, body checkpoint SHA-256,
architecture, 10,000-step/seed-42 provenance and four-form augmentation policy.
Reject unavailable or mismatching sources rather than choosing another model.

For every six-form validation/test case, derive a single plain `def` line from
the prompt's explicit signature using the existing restricted task parser.
Keep LF, omit body indentation to match the learned-header prefix policy, and
require the combined prompt/header to fit context 256. No reference name,
parameters or answer is inserted by generation. References are used only for
scoring after outputs are captured. No fallback for ambiguous signatures.

Run the same frozen body model in PyTorch FP32 and native INT8 with 160 generated
tokens. Compare full-response/AST/syntax metrics and paired exact successes to
the cached learned-header outputs. Retain cached free-body outputs separately.
Names in a successful assisted function are forced from the task parser, not
copied autonomously by the model; assisted wording smoke outcomes cannot replace
the failed autonomous gate or trigger checkpoint promotion.

Preserve raw forced prefixes, references, assisted outputs, cached learned
headers/outputs, per-call native diagnostics, body/engine/report hashes, audits
and paired case outcomes. Native infrastructure failures cause job failure after
report preservation. Empty/whitespace generation remains a quality failure.
Generated source is parsed, never executed. No retraining or production changes.

The treatment removes learned-header inference and uses a task-provided prefix,
so resource budgets differ. Correlated exposed synthetic tasks, one seed,
cached-control versus current-run runtime variation and native/PyTorch numerical
differences remain limits. Even a gain demonstrates only assisted body behavior
under this setup, not general instruction understanding.

## Measured results (14 October 2026)

[PR #162](https://github.com/deser1/NppAI/pull/162), commit
`977f71ae5a93e2d6be41bc36a150b1061c076544`, completed
[run 38614638862](https://github.com/deser1/NppAI/actions/runs/38614638862).
All 15 regression tests, source verification, assisted replay and raw-output
audits passed. Native infrastructure errors were zero; CI_build also passed.

Complete exact reference responses on the same 28 test tasks:

| Prompt form | Cached learned-header FP32 | Task-signature FP32 | Task-signature native INT8 |
| --- | ---: | ---: | ---: |
| Original | 13/28 | 15/28 | 15/28 |
| Please prefix | 9/28 | 13/28 | 13/28 |
| Write Python code prefix | 10/28 | 14/28 | 14/28 |
| Code-only suffix | 11/28 | 13/28 | 13/28 |
| Novel polite/Python prefix | 0/28 | 0/28 | 0/28 |
| Novel implementation prefix | 0/28 | 0/28 | 0/28 |

Cached native learned-header exact test counts equal the cached FP32 column.
Equal aggregate counts do not establish identical raw outputs or success sets;
paired case outcomes are retained in the artifact. The task-signature columns
are assisted generation: the parser supplied function names and parameters.

For the four known validation forms, assisted FP32 exact responses are 14/16,
15/16, 14/16, 15/16; assisted native INT8 is 14/16 for each. Both novel forms
have zero exact responses on validation and test in both runtimes. However,
the novel polite/Python form has 2/16 validation and 2/28 test full-reference AST
matches in each runtime, so zero exact text is not zero structural correctness.
The novel implementation form has zero reference AST matches. Both assisted
wording smoke flags remain false because they require nonzero exact responses.

Known-form test AST counts match assisted exact counts. Novel test syntax is
20/28 for polite/Python and 12/28 for implementation in each runtime; syntax and
forced names do not establish correct bodies. Cached free-body exact responses
remain zero throughout, as measured in the source replay.

Supplying the task's explicit signature improves known-form aggregate reference
matching but does not repair exact response robustness to novel wording. The
remaining failures under an explicit correct header show that learned-header
failure alone is insufficient to explain the pipeline's wording failures in
this setup. They do not identify a single cause in body learning, positions,
decoding or formatting; the two polite/Python AST matches are a limited partial
result. No autonomous-copying or production-quality improvement is claimed,
and no checkpoint is promoted.

Raw prefixes, assisted/reference/control outputs, source/engine/report hashes,
native diagnostics, audits and paired outcomes are in
[artifact 11919645744](https://github.com/deser1/NppAI/actions/runs/38614638862/artifacts/11919645744).
GitHub reports archive SHA-256
`a1c7e324246accbe7c3442bfb73657d3c7a46b47823b44f737f24b597ad8c2ee`.
Artifact retention is 14 days; this committed summary preserves measurements
and provenance, not the raw archive contents.
