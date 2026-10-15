# Source-aligned function-name learning objective

The frozen byte diagnostic found persistent suffix errors even after supplying
correct prior name bytes. This experiment adds a training-only source-position
attention objective and compares it with exactly the prior resampled-name recipe.
It tests an intervention, not an assumed improvement or architectural diagnosis.

Both arms start from the pinned full-response checkpoint
`161d6303c7e77f005cbabba5a1790b9d6a6dfc1812c121bfbb3f1cb6820e5b1a`.
They share the frozen training split, four wording forms, per-draw resampled
aliases, 1024 updates, seed 42, batch size 4, 50% original replay by draw, AdamW
learning rate 0.001, gradient clipping 1, context 256 and two CPU threads.
All response bytes and EOS retain the original equal-weight cross entropy.

The aligned arm adds **0.1 times attention alignment cross entropy**. Each output
name byte has an explicit UTF-8 mapping to its corresponding input name byte in
the prompt's single signature. Because training labels are next-token shifted,
the supervised query is the position immediately before the output byte. The
last layer's query/key projections yield causal attention scores over input
positions; the target is the mapped source position. Missing/ambiguous signatures,
future positions, unequal bytes and overflowing frames are rejected.

Query/key tensors are observed through temporary training hooks, removed even on
failure. The existing forward method, parameter set, inference architecture and
`.nppai` export remain unchanged. There is no new pointer head or hard copying
rule at inference. Attention alignment can improve its own objective without
producing correct byte logits or useful programs; evaluate both independently.
The control computes the alignment loss for logging but backpropagates only the
original response loss. The aligned arm backpropagates the combined objective.
No coefficient search or best-of-seeds selection is performed.

The control must reproduce the previous resampled checkpoint digest
`fc059649eacf8840536fecd1619451bac702eb7ce858d94908527aed2344780a`
and all cached original/renamed development outputs from #183. The common
initial outputs must also reproduce that cache. Both arms record identical
source-example order/exposure metadata; exported tensors must equal training
tensors. A failed control reproduction blocks the comparison.

Evaluation records 164 correlated exposed views: 16 original training draws,
16 actual resampled training draws, original and `ev_` validation/test views
(16/28 each), and same-suffix `nc_` controls (16/28). The matched-prefix names
must be absent from the training alias schedule. Full greedy responses use up
to 160 new bytes with no supplied declaration. Metrics retain complete-function
syntax, requested name, full signature, reference AST and exact response, plus
retained/lost/gained hits. These development groups are already exposed and are
not an independent holdout. The training loader opens only the frozen train split.

After-training models also undergo the existing conditional name-byte and
separate `def `-scaffolded name diagnostic, with per-byte causal-prefix checks.
Those assisted measurements stay separate from complete response scores.
Generated source is parsed only; no functional execution occurs. No local smoke
or fresh-probe task is loaded. There is no automatic checkpoint promotion.

The workflow **Matched source-aligned name-copy objective** uploads raw reports,
audits and experimental checkpoints for 14 days. Source checkpoint availability
is bounded by its artifact retention. Tests cover UTF-8/shifted alignment,
ambiguous signatures, actual gradient flow and future-key masking, unchanged
forward logits/parameters, hook cleanup, control replay and raw evidence audits.

Implementation: `scripts/aligned_name_copy.py`; tests:
`tests/test_aligned_name_copy.py`.

## Observed outcome

Pending the matched experiment. Read complete response and retention counts even
if alignment loss decreases. Assisted names alone do not prove correct bodies,
functional behavior or autonomous coding ability.
