# Fixed CPU arithmetic variants

The [operator localization](first-step-operators.md) observed differences at attention softmax and AdamW sqrt. This experiment tests four predeclared variants without modifying the historical training recipe, plugin or default model.

| Variant | Attention softmax forward/backward | AdamW sqrt |
| --- | --- | --- |
| native | original FP32 | original FP32 |
| softmax64 | FP64 intermediate, FP32 output | original FP32 |
| sqrt64 | original FP32 | FP64 intermediate, FP32 output |
| both64 | FP64 intermediate, FP32 output | FP64 intermediate, FP32 output |

Replacement is scoped through a local dispatch mode. Only `aten._softmax.default` in the forward phase, `aten._softmax_backward_data.default` in backward and `aten.sqrt.default` in the optimizer phase are eligible. Other phases/operators, including log-softmax, RMSNorm, matmuls, SiLU, clipping and other AdamW operations, retain their original arithmetic. The softmax backward promotes the **saved FP32 output** to FP64, computes backward there and returns FP32; it does not recover an unsaved full-precision forward result or claim exact FP64-autograd equivalence. Sqrt is used in optimizer arithmetic without an autograd requirement.

## Frozen inputs and fixed schedule

Prepare only the exact prior softmax input `[4,177,177]` and sqrt input `[256,64]`, checking byte count, SHA256 and dtype/shape/content key against source constants. The workflow downloads immutable prior-run artifact names (38905715952 / 38906998886) once, then packages only the verified input bytes. Missing/expired inputs fail; do not silently regenerate them. Source snapshots expire after 14 days unless separately retained. The prepared small input artifact also has 14-day retention; source hashes and locations remain in code/reports.

Isolated softmax backward uses a fixed upstream gradient `(arange % 17 - 8) / 8`, exact binary fractions. Retain output fingerprints, deduplicated raw result arrays and per-host differences from native. Native is a comparison arm, not a numerical ground truth or accuracy ranking. No generated source is executed.

Four independently provisioned jobs per existing runtime profile (`historical-v1`, `portable-cpu-v1`) each run all four variants twice. Every complete step starts from the same pinned weights and the first fixed resampled training batch, seed 42, original AdamW lr=0.001 and gradient clipping at 1.0. An ordinary untraced source step provides the native reference. Reject a native dispatch mode that changes its loss, logits, raw/clipped gradient or updated-parameter fingerprints. Require exact same-host repeats, finite numerical results, identical initial parameters and actual replacement call counts. No 1024-step training or quality optimization is performed in this experiment.

Compare all six runner pairs per profile and each variant separately: isolated forward/backward/sqrt, logits, loss, raw gradients, clipped gradients and updated weights. Validate matching source, batch, targets and operator inputs first; report actual CPU/vendor coverage and cross-vendor pair counts. One-step agreement on the recorded machines cannot establish full-training repeatability, accuracy, performance or native runtime compatibility. Diagnostic CI succeeds when measurements/checks complete; differences remain explicit results, never automatic model-promotion success. No new baseline is accepted.

README, ROADMAP and CHANGELOG record measured outcomes after the run. Next steps depend on complete-step evidence, not isolated-output equality alone. Measured findings follow.


## Measured run 38950494422

[All ten workflow jobs passed](https://github.com/deser1/NppAI/actions/runs/38950494422): preparation, eight measurement jobs and aggregation. Each measurement job passed all five actual tests. All 64 variant updates (8 jobs × 4 arms × 2 repetitions) repeated exactly on their host; the eight native arms matched ordinary source-step references. Every source/model/batch/input guard passed. This is successful measurement execution, not a successful portability gate.

The historical allocation included Intel Xeon 6973P-C, two AMD EPYC 7763 hosts and AMD EPYC 9V74 (three cross-vendor pairs). The portable allocation included AMD EPYC 9V45 and three EPYC 7763 hosts, with **zero cross-vendor pairs**.

| Historical-profile variant | Isolated softmax forward/backward equal across all pairs | Isolated sqrt equal | Complete logits and loss equal | Raw/clipped gradients and updated weights equal |
| --- | --- | --- | --- | --- |
| native | no | no | no | no |
| softmax64 | yes | no | yes | no |
| sqrt64 | no | yes | no | no |
| both64 | yes | yes | yes | no |

The combined replacement makes the three isolated outputs, complete forward logits and loss agree across the observed historical-profile CPU pairs, including Intel–AMD. Raw gradients, clipped gradients and updated parameter fingerprints still diverge. The first unequal retained step fields are raw gradients, despite isolated softmax backward agreement. This experiment does not record every intermediate/saved forward tensor, so it cannot yet identify a backward operator or establish that all hidden forward values agree. Changing only softmax/sqrt is insufficient for complete-step repeatability in that profile. No candidate is accepted as a replacement arithmetic recipe.

All four variants agree on all isolated outputs and complete-step stages in the portable allocation. This is observed agreement across the recorded AMD models only; it does not retest the previously failed portable AMD–Intel pair and cannot establish a cross-vendor fix. Neither this positive subset nor the FP64 arithmetic implies full-training reproducibility, speed or code quality.

All ten ZIP SHA256 digests were verified locally. Prepared input bytes were checked against source constants; every isolated result array was checked against its dtype/shape/content key and finiteness. Both repetitions, ordinary/native equality and reported per-host deltas from native were independently audited. [Comparison](evidence/arithmetic-variants/comparison.json), [manifest](evidence/arithmetic-variants/manifest.json), and per-run metadata/arm files retain all report content. Recombining metadata with its four listed arm files and applying canonical JSON encoding reconstructs the original report SHA256. Binary inputs/results remain in 14-day artifacts; retained hashes do not reconstruct their bytes. Next: trace forward/saved activations and backward under `both64` to locate the remaining gradient mismatch on unchanged source inputs with measured CPU coverage; keep the original baseline and promotion gates.
