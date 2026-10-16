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

Reports, hardware records, raw repeats and experimental weights use 14-day artifacts. Record the single allocation's outcome, actual coverage and verification limits in README, ROADMAP and CHANGELOG after execution. Keep prior AMD-only evidence unchanged. No production/default model, historical baseline, training recipe or weights are promoted. Measurement pending.
