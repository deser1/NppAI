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
under this setup, not general instruction understanding. Results await CI.
