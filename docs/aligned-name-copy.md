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

The workflow runs control, a fresh control repeat, then aligned training in one
process on the same host with the original PyTorch algorithm settings. The two complete
control reports, including weights, losses, generations and byte diagnostics,
must be byte-identical. Both arms must share initial outputs, runtime and training
schedule. All cached original/renamed control replies and common initial replies
from #183 must also reproduce exactly. Exported tensors must equal training tensors.
A failed same-host repetition or cached-output reproduction blocks the comparison.

Historical control checkpoint `fc059649eacf8840536fecd1619451bac702eb7ce858d94908527aed2344780a`
identity is recorded explicitly as true/false. Cross-runner repetitions produced
different weight hashes with identical cached replies, so historical weight identity
is not a portable gate. The same-host full-report repeat is the strict reproducibility
requirement; numerical differences are disclosed rather than accepted with a
post-hoc weight tolerance. These are distinct checks, not interchangeable claims.

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

Final same-host run [38811231542](https://github.com/deser1/NppAI/actions/runs/38811231542)
completed on 2026-10-15. All seven tests passed with actual PyTorch gradients.
Control, fresh control repeat and aligned training ran sequentially on the same
runner. The two complete control reports are byte-identical, including checkpoint,
losses, full generations and assisted byte diagnostics. Initial outputs and complete
training schedules agree across arms. In this run, historical control checkpoint
identity is also true; this is recorded, not assumed for arbitrary hardware.

**Both arms obtain 0/132 development names, signatures, reference AST matches and
exact full responses.** Complete conditional-name and scaffolded-name counts also
remain 0/132. The attention objective did not demonstrate complete copying at
this budget. Neither checkpoint is promoted.

| Logged training minibatch loss | Control initial / final | Aligned initial / final |
|---|---:|---:|
| Response cross entropy | 1.2386 / 0.2505 | 1.2386 / 0.2791 |
| Attention alignment cross entropy | Not computed | 6.5236 / 2.1081 |

These are first/last sampled minibatches, not a held-out mean or same-example
before/after loss. A smaller auxiliary loss does not establish a correct copying
mechanism. CPU floating-point trajectories can vary between runners, so this
small experiment does not isolate a unique internal cause.

| Group | Views | Control signatures / exact responses | Aligned signatures / exact responses |
|---|---:|---:|---:|
| Original training draws | 16 | 5 / 4 | 6 / 5 |
| Resampled training draws | 16 | 0 / 0 | 0 / 0 |
| All development groups | 132 | 0 / 0 | 0 / 0 |

The initial model had two exact hits in the 16 original training draws. Control
retained none, lost two and gained four. Aligned retained one, lost one and gained
four. These exposed training changes do not establish generalization. Development
retention is vacuous because the initial model had zero signature/exact hits.
Syntax-only changes are mixed: original test control/aligned functions parse in
26/28 versus 25/28 cases; renamed test in 19/28 versus 22/28. Requested names
remain zero. Functional behavior was not executed or established.

Earlier two-runner attempts exposed different control weight hashes despite
matching cached replies; one runner reproduced the historical checkpoint and
another did not. An attempted deterministic-algorithm override also changed
cached replies and was reverted to the original algorithm settings. Those attempts
are not treated as the final paired comparison. The protocol now requires the
strict same-host report repeat and exact historical reply replay, with historical
weight identity reported separately. No coefficient, seed or budget search was
performed. [Earlier run](https://github.com/deser1/NppAI/actions/runs/38809724735).

The paired ZIP was verified against SHA-256
`4a09e5df01b1eb070a560d708cfa16d56722a4e5f7fe810894cfaf0baf110ac0`
(artifact 12008593059). Raw metadata, full before snapshots and 22 after chunks
per arm are retained in git and reconstruct the original report bytes:

- [Control manifest](evidence/aligned-name-copy/control/manifest.json): report
  SHA-256 `7712e75cbe5f2f873ac918023a5471234bc36bbeee82b8faa84eccf96da719c5`.
  The fresh repeat has the same digest; both reports are in the original ZIP.
- [Aligned manifest](evidence/aligned-name-copy/aligned/manifest.json): report
  SHA-256 `0cd48bff5c59983f2dce5f97cc7b1c20ea0dcd4ae059cca752560a5635a275ac`.
- [Paired audit](evidence/aligned-name-copy/pair-audit.json): exact control repeat,
  initial-output and schedule agreement, plus explicit historical identity flag.

Aligned checkpoint SHA-256:
`8140e457692bada9b6db209b1a45e096c78f5ee06d592d2a846529fe058bea65`.
No native/INT8 quality result is claimed for these new weights. Before scaling
training, evaluate an explicit source-byte copy readout or another copying
mechanism on training-only diagnostics, then require separate complete-code and
independent functional evidence. Recorded files enforce LF checkout bytes so
SHA-256 validation remains exact on Windows and Linux.
