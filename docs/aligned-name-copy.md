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
The control runs the original forward/backward graph without attention hooks or
auxiliary computation and records response loss only. The aligned arm records both
losses and backpropagates the combined objective.
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

Valid matched run [38809724735](https://github.com/deser1/NppAI/actions/runs/38809724735)
completed both arms on 2026-10-15. All six tests passed in each PyTorch job. The
control reproduced the exact prior checkpoint and all cached full outputs.
Both arms have identical initial full outputs and complete training schedules.
Artifact ZIP digests were verified and both raw evidence audits repeated locally.

**Both arms obtain 0/132 development names, signatures, reference AST matches and
exact complete responses.** Complete conditional names and scaffolded name hits
also remain 0/132. The extra attention objective did not demonstrate improved
complete name copying at this budget. Neither checkpoint is promoted.

| Logged training minibatch loss | Control initial / final | Aligned initial / final |
|---|---:|---:|
| Response cross entropy | 1.2386 / 0.2505 | 1.2386 / 0.2806 |
| Attention alignment cross entropy | Not computed | 6.5236 / 2.1037 |

These are the first and last sampled training minibatches, not a held-out mean or
same-example before/after loss. A smaller auxiliary loss does not establish a
correct copying mechanism. It also does not offset zero complete-name results.
CPU floating-point trajectories can differ with instrumentation; this small
experiment does not identify a unique causal effect or justify coefficient tuning
on these exposed development cases.

| Group | Views | Control full signatures / exact responses | Aligned full signatures / exact responses |
|---|---:|---:|---:|
| Original training draws | 16 | 5 / 4 | 6 / 5 |
| Resampled training draws | 16 | 0 / 0 | 0 / 0 |
| All development groups | 132 | 0 / 0 | 0 / 0 |

The baseline had two exact-response hits in the 16 original training draws.
Control retained none, lost two and gained four. Aligned retained one, lost one
and gained four. These small exposed training changes are not generalization.
Development retention is vacuous because the baseline had zero signature/exact
hits. Syntax-only changes are mixed: on original test, control/aligned parse as
functions in 26/28 versus 23/28 cases; on renamed test, 19/28 versus 24/28.
Correct-function counts remain zero in both cases.

A first [attempt](https://github.com/deser1/NppAI/actions/runs/38809328488) failed
the strict control checkpoint check despite identical cached full replies. The
control was changed to omit the unused auxiliary graph completely, and an exact
control-gradient regression test was added. That failed comparison was not used
for model selection; the coefficient and budget remained fixed. The valid run
above passes the original checkpoint requirement.

Raw evidence is retained in two chunk manifests, each with a full before snapshot
and 22 after-case chunks. Reassembling the metadata and ordered cases with the
shared `base.encoded` serializer reproduces the original report bytes:

- [Control manifest](evidence/aligned-name-copy/control/manifest.json): report
  SHA-256 `dbd029fb6dfb6239c6b4a73b367aad9520e35cd863ab04abc89a4e1227c3c173`;
  artifact 12008381667 ZIP SHA-256
  `6ba38af4476e202f115fcf3e15e5d3309e378ea20a6b3a90f3381212e3d6731a`.
- [Aligned manifest](evidence/aligned-name-copy/aligned/manifest.json): report
  SHA-256 `482f18b8974c02f58d6fa380144f5aa147a0df949c575f26c1f3dca9084e27ed`;
  artifact 12008106938 ZIP SHA-256
  `a18eb684c6c2256c858c17bdc3291015ace116fb62caef99d9401c23811c3e3f`.

The aligned checkpoint SHA-256 is
`c7df99da89bc14c92d85fead93a70fde6f62645ba1d0814b10a938bc7bb82153`.
No native/INT8 quality or functional-execution result is claimed for it. Before
another training scale-up, compare a dedicated source-byte copy readout or other
explicit copying mechanism against this baseline on training-only diagnostics,
then require separate complete-code and independent functional evidence.

Recorded evidence uses enforced LF checkout bytes so SHA-256 checks remain exact on Windows as well as Linux.
