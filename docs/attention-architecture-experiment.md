# Attention head and position architecture probe

Training broad whitespace offsets did not repair novel wording and harmed
known-form copying. This independent fixed experiment compares three arms:

| Arm | Attention heads | Position scheme |
| --- | ---: | --- |
| legacy1 | 1 | Existing learned absolute embeddings |
| absolute4 | 4 | Same learned absolute embeddings |
| rotary4 | 4 | RoPE on queries/keys; no active absolute embeddings |

The legacy arm uses the existing NppAIModel forward directly. Four-head arms
reuse the same projections, FFN, RMSNorm and output weights but split dimension64
into four width16 heads. Absolute4 isolates head partitioning relative to
legacy1. Rotary4 versus absolute4 changes position scheme, including removal
of active learned-position parameters. RoPE rotates adjacent coordinate pairs
at angle position * 10000^(-2i/head_width), leaving values unrotated. Sources:
[RoFormer paper](https://arxiv.org/abs/2104.09864) and
[authors' implementation](https://github.com/ZhuiyiTechnology/roformer).

Each arm trains separate header/body models from scratch for10000 updates per
model, seed42, context256, dim64/hidden128, two layers, batch4, AdamW0.001 and
clip1. All use the same6144 four-form augmented records and source batch order;
there are no random whitespace offsets. Budgets, arms and forms are fixed before
evaluation; no early stopping, test-guided tuning or checkpoint selection.
Shared tensors initialize identically. Rotary retains an unused frozen absolute
embedding tensor to preserve initialization order and stored layout; report
active and stored parameter counts separately. Compute differs with head layout.

These checkpoints are PyTorch state/configuration archives (`.pt`) only. Native
NPPAI v2 does not encode head count or RoPE, so no native export is attempted.
Production model classes, native inference and defaults remain unchanged.
Configuration validation and strict state loading reject incompatible probes.
Export/reload must reproduce all16 fixed seen-training probe outputs exactly.

After all optimization, evaluate each arm's two-model pipeline and free body
generation on all16 validation and28 test tasks under four trained forms and two
untrained wording prefixes. Reuse the uncorrected-header acceptance/rejection
policy and full AST/exact-reference metrics. Preserve raw source/reference/header
and both outputs, paired outcomes, hashes, runtime and source model/report copies.
Generated code is parsed, never executed. One seed and correlated previously
exposed synthetic tasks do not establish general programming competence. No
automatic promotion, native INT8 or universal architecture claim.

Tests verify strict arm/configuration selection, identical shared initialization,
future-token masking, rotary norm preservation and attention-dot invariance under
a common position shift. The latter does not itself prove task robustness.

## Measured results (13 October 2026)

[PR #155](https://github.com/deser1/NppAI/pull/155) evaluated commit
`5c48a47a5a7e5bc96c4fd1f6fba4a99743bbe196` in
[run 38345447362](https://github.com/deser1/NppAI/actions/runs/38345447362).
All six training jobs and three evaluation jobs passed. Each training job passed
20 regression tests and the seen-probe checkpoint reload comparison. The separate
[Windows CI_build](https://github.com/deser1/NppAI/actions/runs/38345447268) also
passed. CI success confirms execution and regression checks, not model quality.

Complete staged responses exactly matching the references on the same 28 test
tasks (full-reference AST counts were identical):

| Prompt form | legacy1 | absolute4 | rotary4 |
| --- | ---: | ---: | ---: |
| Original | 13/28 | 14/28 | 0/28 |
| Please prefix | 11/28 | 14/28 | 0/28 |
| Write Python code prefix | 12/28 | 14/28 | 0/28 |
| Code-only suffix | 12/28 | 13/28 | 1/28 |
| Novel polite/Python prefix | 0/28 | 0/28 | 0/28 |
| Novel implementation prefix | 0/28 | 0/28 | 0/28 |

On the 16 validation tasks, exact staged responses for the four trained forms
were respectively 11, 9, 11, 12 for legacy1; 11, 13, 10, 11 for absolute4;
and zero throughout for rotary4. Both novel validation forms scored zero in
every arm. Free generation from the body model scored zero exact responses on
every validation/test form in all three arms.

Original-form test header exactness was 20/28, 22/28 and 0/28 respectively.
Rotary4 still produced syntactically valid test functions on 24–28/28 tasks
across the trained forms; syntax validity does not establish requested-name or
complete-reference correctness.

The four-head absolute arm has slightly higher test reference matching on the
trained forms in this one-seed comparison. No arm generalizes to either novel
wording probe. Rotary4 is a negative result under this fixed configuration and
budget despite low training loss; this is not evidence that RoPE fails universally.
These are correlated measurements on exposed synthetic tasks, with differing
active position parameters/compute and two-model inference. No checkpoint is
selected, promoted or exported to native NPPAI; no generated code was executed.

Replay artifacts include raw per-case prompts, references, headers, decisions,
staged/free outputs, paired metrics and verified source checkpoints/reports:

| Arm | Replay artifact | Archive SHA-256 |
| --- | --- | --- |
| legacy1 | [11787829938](https://github.com/deser1/NppAI/actions/runs/38345447362/artifacts/11787829938) | `e3e219ab542b96da78f453b49af3d3cbb6b3bc63b4dc0b4a30c8dc91b5c51d3d` |
| absolute4 | [11789035315](https://github.com/deser1/NppAI/actions/runs/38345447362/artifacts/11789035315) | `a76fbfd01cca644b62b9c02bd615c7370b4ee5e8957f9d58e08b7f725c3af7b1` |
| rotary4 | [11787889960](https://github.com/deser1/NppAI/actions/runs/38345447362/artifacts/11787889960) | `2fff6c67f6f3c17e411e29475cae474ce07924c006107c59b575e19e554e2cff` |

GitHub lists these archives as expiring on 27 October 2026. This committed summary
preserves the reported measurements and provenance, not the artifact contents.
