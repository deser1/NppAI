# Vendor-qualified full-training comparison

The fixed [portable experiment](portable-full-training.md) completed full 1024-update equality on recorded AMD hosts, but neither allocation included Intel. This protocol makes hardware eligibility and the numerical verdict explicit. It does not alter source weights, data, updates, native/both64 arithmetic, optimizer, prior first-step gates or promotion rules.

## Qualification before training

One fixed four-slot allocation declares Intel-1/2 and AMD-1/2. Each slot records the guest-observed `/proc/cpuinfo` vendor, CPU model/flags/microcode and OS version before installing Torch or downloading weights. Only `GenuineIntel` in an Intel slot or `AuthenticAMD` in an AMD slot on Linux x86_64 Ubuntu 24.04 is eligible. A slot with another vendor/OS retains `hardware.json` and skips training. A training report on unqualified hardware, a tampered qualification record, missing artifacts or inconsistent training provenance is an error.

Eligible slots run the existing portable native and both64 arms twice for 1024 updates each, with the same pinned runtime, current-host first-step gate, finite/replacement/export checks and raw byte verification. Record the actual hardware, never infer it from an architecture or runner label. The existing aggregator now accepts a variable number of qualified directories. A single host's repeat files are still validated, but zero pairwise comparisons cannot produce a positive equality result. No duplicate/synthetic hosts are inserted.

## Separate outcomes

| Recorded outcome | Candidate gate |
| --- | --- |
| Missing actual Intel or AMD training results | `inconclusive`, passed=false |
| Both vendors present; any both64 complete trace/checkpoint differs | `failed`, passed=false |
| Both present; all validated observed both64 traces/checkpoints equal | `passed`, passed=true |

The diagnostic workflow succeeds when recording/comparison completes; read `candidate_repeatability_gate` for numerical acceptance. Manual `certify=true` adds `--require-passed-gate`: both inconclusive and failed verdicts exit 2 **after** saving the report, so absent hardware cannot pass certification. Unknown vendors never count. Same-host validity and original input/source checks remain mandatory. Observed equality covers only the fixed seed/schedule/runtime and actual machines, not universal portability or useful code generation.

## Existing runner selection

The workflow accepts `intel_labels` and `amd_labels` as JSON arrays naming **existing** runners. Defaults use the ordinary `ubuntu-24.04` pool for one bounded diagnostic allocation; these defaults do not promise a vendor. No automatic retries, replacement sampling, new runner provisioning or paid machine setup occurs. The [GitHub runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) describes standard OS/architecture labels. Actual CPU qualification is mandatory in addition to label selection.

To make allocation controlled, register/provide existing Ubuntu 24.04 x64 runners and give their actual labels in a manual workflow dispatch. For example, labels such as `["self-hosted","linux","x64","intel","ubuntu-24.04"]` are illustrative, not a claim that those machines exist in this repository. Use the actual configured label sets for each vendor. The current connector does not expose runner inventory/provisioning or workflow dispatch; access to existing vendor-specific hardware cannot be assumed.

Reports, hardware records, raw repeats and experimental weights use 14-day artifacts. Record the single allocation's outcome, actual coverage and verification limits in README, ROADMAP and CHANGELOG after execution. Keep prior AMD-only evidence unchanged. No production/default model, historical baseline, training recipe or weights are promoted. Measured outcomes follow.


## Hardware-qualified allocation 38959870620

[All five diagnostic jobs passed](https://github.com/deser1/NppAI/actions/runs/38959870620), while the numerical gate correctly remained **inconclusive, passed=false**. Intel-1/2 received AMD EPYC 7763 and AMD-1 received Intel Xeon 6973P-C; all three skipped installation/source download/training and retained hardware-only artifacts. AMD-2 received EPYC 7763 and completed the four fixed 1024-update runs with exact same-host repeats. There is only one qualified training host and zero actual pairwise comparisons. Local strict certification on the exact downloaded data exits 2 after writing the original comparison byte-for-byte.

[Hardware evidence](evidence/vendor-qualified-training/manifest.json) retains all four probes, the eligible report/preflight and comparison. All five archive SHA256 digests and CRCs were independently verified. The qualified host's repeated trace/checkpoint bytes and file digests were compared directly; trace bytes reconstruct from the prior unchanged AMD trace chunks referenced by SHA256. Hardware-only archives contain only their probe. Existing runner label sets are still needed for predictable vendor assignment. This gate has not been relabeled as passed because another workflow obtained CPU coverage.

## Independent full-training regression allocation 38959870764

The shared aggregator change also triggered the existing [portable full-training experiment](portable-full-training.md). [All five jobs passed](https://github.com/deser1/NppAI/actions/runs/38959870764), with all four measurement jobs completing both fixed arms twice. Actual coverage is Intel Xeon Platinum 8573C, AMD EPYC 9V74 and two AMD EPYC 7763 environments: three Intel–AMD pairs and six pairs total. Every host passed the frozen both64 first-step gate, finite-update/replacement guards, byte-identical repeats and exact export roundtrip.

| Arm | Full 1025-row traces across all six pairs | Final checkpoint bytes | First Intel–AMD trace difference |
| --- | --- | --- | --- |
| portable native | differ | differ | Update 1: parameter fingerprint differs; that update's inputs, logits, loss and raw/clipped-gradient fingerprints agree |
| portable both64 | identical | identical | None through all 1024 updates |

The both64 checkpoint is `1bd024a162d9cbc79929520db6fb6a28d256be98a87a3bcc4b8149670c9e1da7`, matching the prior AMD-only observation. This establishes observed full fixed-schedule repeatability on these recorded Intel/AMD environments. It does not establish universal CPU portability, other seeds/schedules, performance, code-generation quality, native compatibility or readiness for weight/recipe promotion. Native remains a control, not numerical ground truth; the aggregate first-weight difference does not identify an internal instruction.

[Cross-vendor manifest](evidence/cross-vendor-full-training/manifest.json), original reports/preflights, [comparison](evidence/cross-vendor-full-training/comparison.json) and canonical trace chunks preserve the result. All five ZIP SHA256 digests/CRCs were independently verified. Actual repeated trace/checkpoint bytes, report digests, input exposure and every pairwise result were audited locally. Common traces reference immutable prior chunks; the distinct Intel native trace has its own reconstructable chunks. Original checkpoints remain in 14-day artifacts. Integrity tests bind both the inconclusive qualified allocation and complete observed cross-vendor trace result. No default model, baseline or recipe change is made. Next: independent model-quality/functional evaluation before considering this recipe for any promotion, while retaining controlled-hardware certification separately.
