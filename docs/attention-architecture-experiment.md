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
