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

A [live generation demonstration](https://github.com/deser1/NppAI/blob/main/docs/live-generation-demo.md) records the
actual released plugin sending a Polish request to the own-model service and
showing its unvalidated review. It does not claim successful generated code.

## Try the plugin on Windows

The current installation candidate is described in [v0.5.0-rc.2 notes](docs/releases/v0.5.0-rc.2.md).
When published, it provides plugin packages, a standalone service ZIP and an
explicitly experimental model. Download them from [Releases](https://github.com/deser1/NppAI/releases).
The research model still fails the functional quality gate; this is an integration preview.

1. Download the architecture-matched plugin artifact from a [successful current CI build](https://github.com/deser1/NppAI/actions/workflows/CI_build.yml). Put **NppAI.dll and sandbox_validate.py** in `Notepad++/plugins/NppAI/`.
2. Check out this repository separately for the inference service and launcher. Use Python 3.12+ and a compatible **dense v2, 256-byte-vocabulary** `.nppai` checkpoint.
3. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
.\.venv\Scripts\python.exe scripts\start_nppai.py --model "C:\Models\NppAI-model-v1.nppai" --notepad "C:\Program Files\Notepad++\notepad++.exe"
```

The launcher loads the actual model, verifies authenticated readiness and starts a fresh editor instance with session configuration. Keep its console open.

**[Polska instrukcja: model, serwer, polskie znaki i walidacja](docs/notepad-setup-pl.md).** Installing the DLL alone does not install a model or start its server. Older [v0.4.0 packages](https://github.com/deser1/NppAI/releases/tag/v0.4.0) predate the latest setup and validation changes.

## A reproducible task

In a UTF-8 Python document, enter this in the NppAI panel:

```text
@python-signature silnia(n)
Wygeneruj kod na silnię dla nieujemnej liczby całkowitej n. Zwróć wynik.
```

The application supplies the signature; the model supplies the body. Configure the authored [factorial cases](examples/silnia-validation-cases.json) and Docker as described in the setup guide. Inspect the proposal and test diagnostics; **Apply** requires passing the configured cases and an unchanged original buffer. A configured connection does not guarantee a correct answer.

## Measured model quality

| Evidence | Observed result | Meaning |
| --- | --- | --- |
| [Functional pilot](https://github.com/deser1/NppAI/actions/runs/39162359982), 12 bilingual prompt records | 0/12 generated programs passed in every measured arm; 12/12 authored controls passed | The validator worked; this checkpoint failed the new-task quality gate |
| [All-stage replay](https://github.com/deser1/NppAI/blob/main/docs/all-stage-protection.md) | 45/96 autonomous training matches, up from 2/96; regression matches remain 0/16 and 0/12 | Better training fit does not establish generalization |
| Native/NumPy/PyTorch and quantization tests | Bounded correctness/parity checks | Engineering evidence, not broad coding competence |

Raw outputs, checkpoint hashes, failures and experimental limitations are retained in the reports. No staged checkpoint is promoted on these results. The [next model-quality experiment](docs/fallible-planner-training.md) compares coding with authored versus actual fallible planner artifacts.

## Build and test

Requires Windows, MSVC, CMake 3.15+ and the Windows SDK/DirectX 11 libraries.

```powershell
cmake -S . -B build -A x64 -DNPPAI_TEST_HOST_SEAM=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The release-package and portable-editor checks are described in [release readiness](docs/release-readiness.md). Interactive installation/update behavior, supported editor versions and real task quality must be verified before a stable release.

## Explore the project

- [Architecture](https://github.com/deser1/NppAI/blob/main/docs/architecture.md), [model format](https://github.com/deser1/NppAI/blob/main/docs/model-format.md) and [testing](https://github.com/deser1/NppAI/blob/main/docs/testing.md)
- [Detailed overview and research history](https://github.com/deser1/NppAI/blob/main/docs/project-details.md)
- [Roadmap](https://github.com/deser1/NppAI/blob/main/ROADMAP.md) and [changelog](https://github.com/deser1/NppAI/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/deser1/NppAI/blob/main/CONTRIBUTING.md), [security](https://github.com/deser1/NppAI/blob/main/SECURITY.md) and [license](https://github.com/deser1/NppAI/blob/main/license.txt)
