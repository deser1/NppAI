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

README, ROADMAP and CHANGELOG record measured outcomes after the run. Next steps depend on complete-step evidence, not isolated-output equality alone. Measurements pending.
