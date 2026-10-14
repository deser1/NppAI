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

TestEngine accepts an optional token budget after the prefix-file argument,
bounded to 1–256. Its existing default remains 256; this replay uses 160 to
match FP32. Native EOS and conversation-marker handling, RMSNorm epsilon and
quantization/numerics differ from PyTorch; this is a runtime comparison, not
a quantization-only causal ablation. The production engine is unchanged.

Both report audits recompute raw-output metrics against frozen cases. Valid
empty generation (TestEngine exit 5) is a quality failure in the denominator;
load errors, missing output on exit 0 and timeouts are infrastructure failures,
recorded before the job fails. Preserve raw headers/outputs, acceptance decisions,
per-call diagnostics, source/engine/report hashes, both audits and paired counts.
Native output bytes are UTF-8-decoded with replacement, without newline conversion.

Successful execution is not model quality. Minimal wording smoke checks can
remain false. Exposed correlated synthetic tasks, one seed and two-model
inference remain limitations. No promotion or generated-code execution.
Measured results await CI.
