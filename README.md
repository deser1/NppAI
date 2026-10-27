# NppAI — a native coding assistant for Notepad++

[![CI_build](https://github.com/deser1/NppAI/actions/workflows/CI_build.yml/badge.svg)](https://github.com/deser1/NppAI/actions/workflows/CI_build.yml)

An independent C++ inference engine, custom model format and Python training pipeline, integrated with a Windows editor. NppAI explores local inference, repository context and test-driven code repair using its own models.

**Status: experimental.** The plugin and validation infrastructure are tested; the current small research models are not general-purpose coding assistants.

## What you can inspect

- C++17 Transformer inference, runtime AVX2/FMA dispatch, scalar fallback and INT8 kernels.
- Versioned `.nppai` files with payload length and SHA-256 verification.
- Repository indexing/RAG, incremental updates and measured retrieval benchmarks.
- Local or authenticated NppAI-server generation, visible output and automatic backend routing.
- Python proposals checked in a restricted local Docker container, at most three repair candidates, diff review and guarded **Apply/Reject**.
- Windows x64, x86 and ARM64 builds, native CTest, backend tests and Docker integration in CI.

The online backend can be your own NppAI server on the same computer. An external hosted AI account is not required.

A [recorded factorial task](docs/factorial-model-repair.md#verified-release-and-demonstration)
shows the actual plugin requesting the own model and displaying code that passed
isolated authored tests. The [experimental model release](https://github.com/deser1/NppAI/releases/tag/model-factorial-v1)
contains the checkpoint, matching service, model card, recording and raw evidence.
The earlier [failed-generation recording](docs/live-generation-demo.md) remains available.

## Try the recorded task on Windows x64

1. Download [NppAI-x64.zip](https://github.com/deser1/NppAI/releases/download/v0.5.0-rc.2/NppAI-x64.zip).
   Extract **NppAI.dll and sandbox_validate.py** into `Notepad++/plugins/NppAI/`.
2. Download [NppAI-factorial-service.zip](https://github.com/deser1/NppAI/releases/download/model-factorial-v1/NppAI-factorial-service.zip)
   and [NppAI-factorial.nppai](https://github.com/deser1/NppAI/releases/download/model-factorial-v1/NppAI-factorial.nppai).
   Verify the release checksums and extract the service ZIP. A repository checkout is not required.
3. Open PowerShell in the extracted service directory, using Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
.\.venv\Scripts\python.exe scripts\start_nppai.py --model "C:\Models\NppAI-factorial.nppai" --prompt-bytes 256 --python-functions --notepad "C:\Program Files\Notepad++\notepad++.exe"
```

Adjust the two absolute paths. The launcher verifies model readiness and starts
a fresh editor instance with a private session configuration; keep its console
open. This optional profile uses the own loopback server. Native C++ inference
does not implement its additional decoding rules.

Open an empty UTF-8 `.py` document and enter:

```text
wygeneruj kod na silnię
```

This exact recorded task is a training probe, not an unseen benchmark. The model
supplies the complete function. Configure the authored [factorial cases](examples/silnia-validation-cases.json)
and Docker using the included [Polish setup guide](docs/notepad-setup-pl.md).
**Apply** needs passing configured tests and an unchanged buffer. The recording
has Apply disabled: Linux Docker execution and actual Windows review are verified
separately; Windows Docker Desktop + validated GUI Apply/Undo and upgrade remain
unverified. An [executable full-path check](scripts/windows_validated_smoke.ps1)
is provided for an interactive Windows Docker Desktop environment.

The [model card](https://github.com/deser1/NppAI/releases/download/model-factorial-v1/MODEL_CARD.md)
records its narrow scope and required options. Other plugin architectures are
available in [rc.2](https://github.com/deser1/NppAI/releases/tag/v0.5.0-rc.2);
the recorded model/service path was checked on x64. Older research checkpoints
remain available and retain their reported failures.

## Measured model quality

| Evidence | Observed result | Meaning |
| --- | --- | --- |
| [Function-body continuation](docs/function-body-training.md) | Six-family pilot: original replies 96/96, development 0/24, factorial retention 20/22 → 18/22. Exposed diagnosis: familiar signatures/new wording 10/12; changed names or arguments 0/12 each | Candidate rejected; no production weights or decoder change |
| [Broader released-model evaluation](docs/released-model-generalization.md) | Production profile 1/40; legacy greedy 0/40. New operation views 0/24; additional factorial names 0/4; exposed bilingual regression 1/12 | Frozen released checkpoint, actual restricted Docker execution; no general-purpose promotion |
| [Factorial repair](docs/factorial-model-repair.md) | 12/12 configurations of one development description; reproductions 2/2, Polish TRAIN probes 4/4; new names 2/4. Original training replies 96/96 in the actual production profile, all prior 88 hits retained | Narrow authored factorial tasks; no general-coding quality claim |
| [Functional pilot](https://github.com/deser1/NppAI/actions/runs/39162359982), 12 bilingual prompt records | 0/12 generated programs passed in every measured arm; 12/12 authored controls passed | The validator worked; this checkpoint failed the new-task quality gate |
| [All-stage replay](https://github.com/deser1/NppAI/blob/main/docs/all-stage-protection.md) | 45/96 autonomous training matches, up from 2/96; regression matches remain 0/16 and 0/12 | Better training fit does not establish generalization |
| Native/NumPy/PyTorch and quantization tests | Bounded correctness/parity checks | Engineering evidence, not broad coding competence |

Raw outputs, checkpoint hashes, failures and experimental limitations are retained in the reports. The factorial checkpoint is a separate experimental model, with no stable/general-purpose promotion. Its new tasks are not the older bilingual benchmark, whose result remains attached to the earlier checkpoint. See the [research history](docs/project-details.md) for broader experiments and retained failures.

## Build and test

Requires Windows, MSVC, CMake 3.15+ and the Windows SDK/DirectX 11 libraries.

```powershell
cmake -S . -B build -A x64 -DNPPAI_TEST_HOST_SEAM=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The release-package and portable-editor checks are described in [release readiness](docs/release-readiness.md) and the [stable checklist](docs/stable-readiness.md). Interactive installation/update behavior, supported editor versions and real task quality must be verified before a stable release.

## Explore the project

- [Architecture](https://github.com/deser1/NppAI/blob/main/docs/architecture.md), [model format](https://github.com/deser1/NppAI/blob/main/docs/model-format.md) and [testing](https://github.com/deser1/NppAI/blob/main/docs/testing.md)
- [Detailed overview and research history](https://github.com/deser1/NppAI/blob/main/docs/project-details.md)
- [Roadmap](https://github.com/deser1/NppAI/blob/main/ROADMAP.md) and [changelog](https://github.com/deser1/NppAI/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/deser1/NppAI/blob/main/CONTRIBUTING.md), [security](https://github.com/deser1/NppAI/blob/main/SECURITY.md) and [license](https://github.com/deser1/NppAI/blob/main/license.txt)

The [neutral-name/position experiment](docs/neutral-binding-training.md#measured-result-27-october-2026) improves shifted familiar-argument views from 2/12 to 12/12 and transfers equal-length new function names, while retaining all prior hits. New arguments remain 0/24 in the primary gate; the candidate is rejected and released weights are unchanged.
