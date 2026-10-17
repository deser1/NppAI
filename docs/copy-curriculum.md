# Training-only identifier-copy curriculum

Status: frozen experiment, measurement pending. No default or weight promotion.

The preceding portable checkpoint readout scored 0/132 requested development
names/signatures and 0/6 functional tasks. This experiment trains a dedicated
copying curriculum rather than inferring quality from numerical repeatability.
The complete specification and actual schedule hashes are frozen before training
in `evaluation/copy-curriculum-v1.json`.

## Fixed comparison

Both arms initialize the base from the exact portable both64 export
`1bd024a162d9cbc79929520db6fb6a28d256be98a87a3bcc4b8149670c9e1da7`
and initialize an identical Python-only gated readout with seed 4242. The source
archive and original repeat bytes are checked by the preceding quality harness.
This is a **new continuation control**, not historical checkpoint reproduction.
Both use native FP32 gated-readout arithmetic under the portable CPU environment;
the prior cross-CPU certificate does not extend to this new model/objective.

Both get the same 1024 batches of four training examples: two original and two
deterministically resampled aliases per update. AdamW is reset, learning rate
0.001, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01, gradient clipping 1.0.
For the first 256 updates both freeze the base and train the readout; the remaining
768 updates unfreeze all parameters without resetting the optimizer.

| Arm | First 256 targets | Last 768 targets | Loss |
|---|---|---|---|
| Control | Full function plus EOS | Full function plus EOS | Response NLL |
| Curriculum | `def name(` plus EOS | Full function plus EOS | Response NLL + source-position NLL + 0.1 gate BCE |

The curriculum explicitly teaches each function-name byte's aligned **prompt**
position and sets its gate target to one. Other supervised response bytes,
including EOS, have gate target zero. Pointer supervision distinguishes repeated
source bytes by their positions. Labels come only from training references;
they are never supplied during autonomous inference. UTF-8 alignment is byte
based. Future tokens and response positions remain ineligible copy sources.
The intervention bundles target curriculum and two auxiliary objectives; it
does not separately identify their effects. Response-token exposure differs
between arms and is disclosed in the frozen schedule; this is not a matched-token
compute experiment. There is one setting and seed, with no search or early stop.

## Evidence and gates

Train each arm twice from scratch on one host. Require identical complete trace
and versioned `.nppaicopy` bundle bytes, finite parameters/gradients and exact
export/load parameter identity. The bundle remains Python-only; native loaders
must reject it. Retain every update's loss components, parameter hash and actual
batch hash, plus initialization and CPU/library provenance.

Measure common initial and final autonomous full replies on the 132 exposed
development views and 32 training regression views using unassisted greedy
decoding (context 256, up to 160 new bytes). Recompute name/signature, syntax,
reference AST and exact-response metrics, with lost/retained/gained original-task
hits. Separately report **teacher-forced training-only** gate probabilities,
aligned pointer probabilities, conditional name bytes and `def `-scaffolded
decoding. Improved supervised probabilities do not establish autonomous copying.

Run the six previously exposed functional regression probes using only the
restricted Docker validator. They are optimization-independent but correlated
and previously exposed, not a fresh holdout or broad programming benchmark.
Keep `fresh-probe-v1` and other independent evaluation out of this training path.
Require complete-code and original-task regression gates, then a frozen candidate
and genuinely fresh broader functional evaluation before any promotion. A
successful workflow only establishes that the measurement completed.
