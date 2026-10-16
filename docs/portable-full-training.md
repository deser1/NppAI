# Fixed portable full-training repeatability experiment

This protocol freezes the next test before measurement. The `portable-cpu-v1` + `both64` recipe already passed complete first-step comparisons on actual AMD/Intel allocations in [both64 tracing](both64-trace.md). Both retained comparison JSON files are bound by fixed SHA256 in the experiment code. Require their four reliable reports, six equal pairs, source identity and cross-vendor coverage before training. On each newly allocated host, require its complete both64 first update (loss/logits, raw/clipped gradients, parameters and replacement counts) to equal both retained references. Preserve a failed preflight before aborting; never replace the reference or retry until green.

## Frozen arms and exposure

Run four independently provisioned CPU jobs. Each job trains both arms twice, independently resetting the same pinned initial weights and seed. In every arm use exactly the existing 1024-step resampled training-only schedule: four rows per update, original AdamW lr=0.001, reset optimizer, clipping at 1.0, seed 42, source model `161d6303…`. Keep the frozen train split and alias schedule; no development/test references are loaded by the training path.

| Arm | Runtime | Arithmetic |
| --- | --- | --- |
| native | portable-cpu-v1 | unchanged original FP32 |
| both64 | portable-cpu-v1 | existing FP64-intermediate attention softmax forward/backward and optimizer sqrt; FP32 outputs |

Both arms retain original loss log-softmax, matrix multiplication, clipping and other operators. The historical two-thread runtime is not trained in this experiment; its separately observed loss-log-softmax/backward-matmul boundaries remain open. This experiment tests the already gated portable recipe over the whole fixed schedule.

Pinned CI: Python 3.12.15, Torch 2.14.1+cpu, NumPy 2.5.2, one thread, MKL_CBWR=COMPATIBLE, ATEN_CPU_CAPABILITY=default, MKLDNN disabled, deterministic algorithms enabled. Record actual CPU model/flags, library configuration, environment and checkout revision. Host allocation is not controlled; report actual vendor coverage.

## Guards and retained outputs

Check finite loss/logits, gradients and updated parameters at every update. Require both64 replacement counts at every step. The training implementation's first update must exactly equal the previously tested `complete_step` implementation in both arms; isolate the reference call's RNG state. Record 1025 trace rows (initial state plus every update) with parameter fingerprints, hexadecimal loss, logit/raw/clipped-gradient fingerprints and actual input fingerprints. Keep original raw trace bytes before any same-host guard failure. Export final checkpoints, verify exact parameter roundtrip and record historical checkpoint equality separately. A new recipe checkpoint is never silently accepted as the historical control.

Same-host repetitions must have byte-identical complete traces/checkpoints and identical full result metadata. Aggregation reads actual repeated files, verifies their digests and step numbers/endpoints, requires identical source/schedule and actual per-step input exposure, then compares all six CPU pairs for each arm. Report the earliest differing step, final-parameter equality and checkpoint equality. Successful measurement execution does not imply a passed numerical repeatability gate.

Observed 1024-step equality is positive evidence for these machines/runtime/seed/schedule only when actual AMD–Intel pairs are present. AMD-only equality is inconclusive about cross-vendor repeatability. Any candidate step/checkpoint difference rejects a repeatability claim; retain the failure and earliest divergence. Native is a control, not numerical ground truth. No operator/accuracy attribution follows from aggregate differences.

No generations, model-quality benchmarks, independent functional probes or generated-source execution are included. Numerical equality does not demonstrate useful code generation, performance, native runtime compatibility or safety. No default model, training recipe, historical baseline or weights are promoted. Next quality/copying work remains separately gated.

All raw reports, traces and experimental checkpoints have 14-day artifact retention. After measurement, retain original reports, reconstructable trace chunks, pairwise comparison, source/CPU coverage, archive digest verification and explicit results/limits in git. Update README, ROADMAP and CHANGELOG with measured outcomes. Measured outcomes follow.


## Initial measurement 38956334198 and aggregation repair

[All four measurement jobs passed](https://github.com/deser1/NppAI/actions/runs/38956334198). Each job passed four actual tests and its current-host first-step gate, then completed native and both64 twice (16 independent 1024-update runs total). All same-host repeats match exactly, with finite updates and exact export roundtrips. Actual coverage is two AMD EPYC 9V45 and two AMD EPYC 7763 hosts, **zero Intel hosts/cross-vendor pairs**.

The original aggregation job failed before numerical comparison: importing training data at module scope required optional pydantic under `python -S`. Keep that failed job visible. Move the training-only import into the training entry point; the fixed aggregator CLI now runs with standard library only, with a regression test. Training/update arithmetic is unchanged. Locally aggregate the exact original four archives after this import repair; preserve comparison origin rather than claiming the original workflow was green.

For both arms, all six AMD pairs agree on all 1025 trace rows, including every update's loss/logits/raw/clipped-gradient and updated-weight fingerprints, and final checkpoint bytes. Native checkpoint SHA256 is `75b01f90c184507f63ee8aba4a1a95d30c1c61f780e3fd595fccd71d56c4a743`; both64 is `1bd024a162d9cbc79929520db6fb6a28d256be98a87a3bcc4b8149670c9e1da7`. Arms are distinct recipes/checkpoints, and neither equals the historical control. Cross-vendor full-training repeatability is **inconclusive** in this allocation. No quality, performance, universal portability or model-promotion conclusion follows.

[Manifest](evidence/portable-full-training/manifest.json), [comparison](evidence/portable-full-training/comparison.json), original per-host reports/preflights and deduplicated canonical trace chunks retain all trace bytes. Four ZIP SHA256 digests and full ZIP CRCs were independently verified. Both repeats' actual trace/checkpoint bytes were directly compared; trace and checkpoint file digests, finite/first-step guards, input exposure and every pairwise result were audited. Recombining each listed trace's chunks into rows and encoding each as canonical JSON + LF reconstructs its original SHA256 exactly. Checkpoint bytes remain in 14-day artifacts; digests do not reconstruct weights. An integrity test verifies reconstruction and all retained pairwise results. The next repaired-head workflow is a separate integration validation, not a replacement for this AMD-only observation or a retry-until-green numerical gate.
