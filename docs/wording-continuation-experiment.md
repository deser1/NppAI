# Frozen four-form versus eight-form continuation

Eight-form training from scratch lost known-form reference matching in PR #159.
PR #162's explicit task signatures did not repair exact novel-wording responses.
This next fixed experiment continues the previously measured four-form pair,
testing retention and broader wording coverage without changing architecture.

Both policies start from the same fixed header/body checkpoints in
`nppai-diverse-evaluation-four_forms` from run 38582420228. Verify source model
hashes, manifests, seed 42, context 256, 10,000-step source budget and four-form
policy/counts; compatible dense 64/128, two-layer byte weights are fully loaded.
No alternate source selection if the artifact is unavailable.

Train four models: each policy continues a header and body model for 10,000
additional updates. Per model, source 10,000 plus continuation 10,000 equals
20,000 optimizer updates. Fresh AdamW in both arms, seed 42, batch four, learning
rate 0.001, clip 1, body name weight 1. No whitespace offsets or forced headers.
Four-form control and eight-form treatment reuse the policies fixed in PR #159;
both exact novel evaluation prefixes remain excluded from training. Targets,
source names/splits, architecture and defaults remain unchanged.

Record initialization hash/method, source/additional/total updates, policy/forms,
derived training hashes and optimizer reset. Preserve both starting checkpoints
alongside final models/reports. Verify header export/reload outputs; additionally
require body reload to reproduce all 24 fixed seen-training probe outputs before
replay. Reject absent/mismatching initialization or budget metadata.

Evaluate both continued pairs on all 16 validation and 28 test tasks under the
same six forms, retaining raw generated headers, uncorrected staged outputs,
free body outputs, paired outcomes and raw-output wording audits. No reference
correction, fallback, early stopping, held-out checkpoint selection or promotion.
No native INT8 quality claim or generated-code execution.

Optimizer updates match, not tokens, compute, sampled batches or per-record
exposure. A warm-start versus previous cold-start comparison is not a matched
ablation because cumulative budgets differ. Previously exposed correlated
synthetic tasks and one seed remain limits. Report all arms and distinguish
successful experiment execution from model quality. Results await CI.
