# Frozen staged FP32 versus native INT8 evaluation

Compare the existing single-head four-form control pair from run
[38582420228](https://github.com/deser1/NppAI/actions/runs/38582420228), artifact
`nppai-diverse-evaluation-four_forms`. No retraining or architecture selection.
Download fails if the fixed source artifact is unavailable; do not choose a
different checkpoint based on scores.

The replay checks frozen manifests, checkpoint SHA-256, architecture, source
10,000-step budget/seed and four-form policy/counts. First replay PyTorch FP32;
then replay native INT8 using the same checkpoints, 160-token budget, all 16
validation and 28 test tasks and the same six wording forms. Native emits its
own raw header, which must pass the existing bounded header acceptance rule;
its body model receives that exact prefix with LF bytes preserved. Reference
names/parameters are never inserted or corrected. Compare free body generation
alongside staged generation and pair exact successes by source case ID.

The dedicated `StagedNativeRunner` accepts an optional token budget after the
prefix-file argument, bounded to 1–256 and defaulting to 160 to match FP32.
Existing TestEngine and its 256-token default remain unchanged.
Native EOS and conversation-marker handling, RMSNorm epsilon and
quantization/numerics differ from PyTorch; this is a runtime comparison, not
a quantization-only causal ablation. The production engine is unchanged.

Both report audits recompute raw-output metrics against frozen cases. Valid
empty generation (runner exit 5) is a quality failure in the denominator;
load errors, missing output on exit 0 and timeouts are infrastructure failures,
recorded before the job fails. Preserve raw headers/outputs, acceptance decisions,
per-call diagnostics, source/engine/report hashes, both audits and paired counts.
Native output bytes are UTF-8-decoded with replacement, without newline conversion.

Successful execution is not model quality. Minimal wording smoke checks can
remain false. Exposed correlated synthetic tasks, one seed and two-model
inference remain limitations. No promotion or generated-code execution.
## Measured results (14 October 2026)

[PR #161](https://github.com/deser1/NppAI/pull/161), commit
`752670a49966a4515b49692b99cf6d143be06cdc`, completed
[run 38591181871](https://github.com/deser1/NppAI/actions/runs/38591181871).
The Windows replay, raw-output audits and invalid-token-budget checks passed;
native infrastructure errors were zero. The separate CI_build also passed.
Both runtimes used the same frozen four-form header/body pair and 160-token limit.

Complete staged exact reference responses on the same 28 test tasks:

| Prompt form | PyTorch FP32 | Native INT8 |
| --- | ---: | ---: |
| Original | 13/28 | 13/28 |
| Please prefix | 9/28 | 9/28 |
| Write Python code prefix | 10/28 | 10/28 |
| Code-only suffix | 11/28 | 11/28 |
| Novel polite/Python prefix | 0/28 | 0/28 |
| Novel implementation prefix | 0/28 | 0/28 |

Full-reference AST counts match the exact-response counts in this measurement.
Validation exact responses on the four known forms are 10/16, 9/16, 11/16,
12/16 in FP32 versus 10/16, 8/16, 11/16, 11/16 in native INT8. Both novel
validation forms score zero in both runtimes. Free body-model exact responses
are zero on every form/split in both runtimes. Original test header exactness
is 20/28 for both; Please-prefix header exactness is 17/28 FP32 versus 16/28
native. Other syntax metrics also differ.

Equal aggregate test exact counts do not establish byte-identical outputs or
identical per-case success sets. Both runtime audits record
`wording_smoke_passed: false`. This narrow replay preserves original test
reference matching in native inference but does not repair novel wording or
establish general runtime parity, production readiness, or multi-head/RoPE native
compatibility. The older native baseline's whitespace-only failure remains a
separate model-quality blocker. No checkpoint is promoted.

Raw FP32/native cases, generated headers, per-call diagnostics, source copies,
engine/report hashes, audits and paired case outcomes are in
[artifact 11908680639](https://github.com/deser1/NppAI/actions/runs/38591181871/artifacts/11908680639).
GitHub reports archive SHA-256
`2bf968891f307f80ec6141a627ffc0c351b55c09d26e13f1db859d8d7f5bc35e`.
Retention is 14 days; this committed summary retains the measured counts and
provenance, not the raw archive contents.

Frozen instruction JSON files use `eol=lf` in `.gitattributes` to preserve
manifest hashes on Windows checkouts. Regression coverage checks each manifest
against raw file bytes and exercises Git checkout with `core.autocrlf=true`.
Do not normalize bytes in the hash verifier or rewrite manifest hashes to accept
platform changes. The older native baseline quality gate remains unchanged;
its whitespace-only model output is still a failed generation, not a loader fix.
