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
successful experiment execution from model quality.

## Measured results (14 October 2026)

[PR #163](https://github.com/deser1/NppAI/pull/163), commit
`a09ee261154b8182f28867fe99b46aa831d1876d`, completed
[run 38618406741](https://github.com/deser1/NppAI/actions/runs/38618406741).
All four trainings and both replays passed, including the real PyTorch full-weight
initialization test and checkpoint reload checks. All six PR workflows passed.

Complete staged exact/reference AST test matches on the same 28 tasks:

| Form | Four-form continuation | Eight-form continuation |
| --- | ---: | ---: |
| Original | 13/28 | 14/28 |
| Please prefix | 12/28 | 13/28 |
| Write Python code prefix | 13/28 | 11/28 |
| Code-only suffix | 13/28 | 15/28 |
| Novel polite/Python prefix | 0/28 | 0/28 |
| Novel implementation prefix | 0/28 | 0/28 |

Known-form validation exact counts are 12/16, 15/16, 13/16, 14/16 for four-form
versus 11/16, 13/16, 11/16, 10/16 for eight-form. Both novel validation forms
score zero in both arms. Free-body exact scores remain zero throughout. Original
test header exactness is 21/28 versus 22/28; novel headers are zero except 1/28
for eight-form implementation, with no complete-code success.

Both actual wording smoke audits fail. Eight-form continuation retains known-form
matching unlike the earlier cold-start run, but cumulative budgets differ and
do not isolate warm starting. It does not repair novel wording; validation favors
the simultaneous four-form control and test changes are mixed. No promotion.

Raw reports, audits, preserved starting checkpoints and paired cases are in
[four-form artifact 11921870012](https://github.com/deser1/NppAI/actions/runs/38618406741/artifacts/11921870012)
and [eight-form artifact 11921600155](https://github.com/deser1/NppAI/actions/runs/38618406741/artifacts/11921600155).
GitHub archive SHA-256 values are respectively
`81bdd1d4e1e5afbbd7fa430ff3a9f58c4dd657a4afae68a9048bc7a58c64a560`
and `4ee858805ef6f3da0bcd378ae9c82d12c3e7ce1563c8943f7f55ef987370a65b`.
Retention is 14 days; this committed summary preserves counts/provenance, not
the raw archive contents. The [fresh evaluation probe](fresh-probe-v1.md) creates
new task instances for subsequent measurements instead of reusing only these 28.
