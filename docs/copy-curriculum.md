# Training-only identifier-copy curriculum

Status: fixed experiment measured; promotion rejected. No production weight change.

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

## Measured result

[Run 39018130210](https://github.com/deser1/NppAI/actions/runs/39018130210)
completed on AMD EPYC 7763 with the pinned Python 3.12.15 / Torch 2.14.1+cpu /
NumPy 2.5.2 runtime. All four 1024-update trainings finished; both arms' repeat
trace and bundle bytes match exactly. The two arms have identical initial
parameter hashes. The original artifact ZIP SHA256/CRC, complete repeated bytes,
all minibatch hashes, loss compositions and raw report scores were independently
checked. `docs/evidence/copy-curriculum/manifest.json` reconstructs the exact
report and both full traces from retained chunks. Weights remain in the 14-day
Actions artifact, not in git; retained hashes alone cannot reconstruct them.

| Autonomous complete-response readout | Common initial | Control | Curriculum |
|---|---:|---:|---:|
| Development requested name/signature/exact reply | 0/132 | 0/132 | 5/132 |
| Original development exact reply | 0/44 | 0/44 | 4/44 |
| `ev_` renamed development exact reply | 0/44 | 0/44 | 0/44 |
| `nc_` matched-prefix development exact reply | 0/44 | 0/44 | 1/44 |
| Original training exact reply | 4/16 | 3/16 | 1/16 |
| Original training requested signature | 5/16 | 3/16 | 1/16 |
| Renamed training exact reply/signature | 0/16 | 0/16 | 0/16 |
| Exposed Docker functional passes | Not rerun initially | 0/6 | 0/6 |

The five curriculum development hits are three original test tasks (sum, lower,
upper) plus the same validation positivity task in original and matched-prefix
views. They are **four instances with correlated views**, not five independent
functional successes. The matched-prefix validation alias is unseen in the
training schedule, but matched-prefix test still scores 0/28. Development syntax
scores are 116/132 control versus 114/132 curriculum.

The control retains two of four initial exact original-training replies, loses
two and gains one. The curriculum retains **none**, loses four and gains one.
This original-task regression blocks promotion even before independent functional
gates. Functional breakdown: control has four callable failures, one syntax
failure and one empty reply; curriculum has three callable failures, two syntax
failures and one empty reply. Neither passes a task. Workers stop at first
failure, so do not interpret configured test cases as all executed or body
semantics as established.

## Training-only diagnostics

These values use gold response prefixes on the same 16 original and 16 renamed
training views; they are not autonomous full-code metrics.

| Mean supervised probability | Control | Curriculum |
|---|---:|---:|
| Original training name gate | 0.010189 | 0.923389 |
| Original training other-response gate | 0.000490 | 0.005512 |
| Original training aligned pointer | 0.059191 | 0.874330 |
| Renamed training name gate | 0.487042 | 0.892177 |
| Renamed training other-response gate | 0.012334 | 0.018224 |
| Renamed training aligned pointer | 0.272321 | 0.785446 |

Conditional correct name bytes improve 118/144→125/144 for original training and
111/176→143/176 for renamed training. However all conditional original names
drop 3/16→1/16, while renamed conditional names rise only 0/16→1/16. Separate
name decoding after the disclosed `def ` scaffold scores 3/16→1/16 original and
0/16→0/16 renamed. Gate/pointer improvements do not establish reliable sequential
copying. All causal-prefix checks and export/load parameter checks passed.

The next bounded experiment should diagnose **training-only** namespace-invariant
sequential copying and original-task retention, without selecting coefficients
on these exposed development outcomes. Freeze a candidate before genuinely fresh
broader functional evaluation. This measured result does not justify scaling or
promoting the current curriculum.
