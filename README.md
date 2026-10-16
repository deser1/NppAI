# NppAI — Native AI Coding Assistant for Notepad++

NppAI is an experimental AI coding assistant for Notepad++ built around a custom C++ inference engine and a Python/PyTorch training pipeline.

The project focuses on **local inference, systems-level implementation, model formats, RAG, and developer tooling** rather than wrapping an external hosted AI API.

> **Project status:** experimental / active development. Model quality, performance and deployment characteristics are still being validated.

## Validated generation and automatic repair

The plugin stages a proposal, checks Python syntax and authored function tests
in a restricted **local Docker container**, then repairs failures with a limit
of **three candidates total**. Automatic mode can escalate failed local
proposals to the configured online model. The panel shows diagnostics and a
diff; **Apply** requires successful validation and an unchanged original buffer,
and creates one undo action. **Reject** leaves the document unchanged.

This first profile supports UTF-8 Python buffers and user-authored JSON function
cases. Python, a prepared local Docker Linux image and a test suite must be
configured; missing infrastructure blocks Apply. Passing those cases does not
establish general coding ability. Other language/build profiles remain open.
See [setup, limits and test evidence](docs/validated-generation.md), the
[example suite](examples/python-validation-cases.json), and
[automatic local/online selection](docs/automatic-plugin-mode.md).

## Highlights

- **Custom C++ inference engine** implementing Transformer-style building blocks.
- **Custom `.nppai` binary model format** with model metadata and tensor weights.
- **Local context / RAG components** for using project files as additional context.
- **CPU optimization** using OpenMP and AVX2 where the selected build target supports it.
- **INT8 weight quantization** for selected model tensors.
- **DirectX 11 compute path** for selected matrix-multiplication workloads, with CPU fallback.
- **Python/PyTorch training and validation tools.**
- **Notepad++ plugin integration** with native Windows APIs.
- **Telemetry / correction-data pipeline** intended for privacy-aware collection and later centralized retraining.

## Architecture

```text
                         Notepad++
                             │
                             ▼
                    ┌─────────────────┐
                    │    NppAI DLL    │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
     AIManager          RAGManager       TelemetryManager
          │                  │                  │
          ▼                  ▼                  ▼
    NppAIEngine        Local context      Backend API
          │
     ┌────┴─────────────┐
     ▼                  ▼
   CPU path          DirectX 11
 OpenMP/AVX2        compute path
     │                  │
     └────────┬─────────┘
              ▼
        Model inference

Training / validation:

Python → PyTorch → .nppai model → C++ inference
```

## Repository structure

```text
NppAI/
├── src/                    # Native plugin and inference engine
├── datasets/               # Training data location
├── models/                 # Local model files (ignored by Git)
├── .github/workflows/      # GitHub Actions
├── docs/                   # Architecture and development docs
├── test_engine.cpp         # Native inference smoke test
├── test_python.py          # Python/model-format validation
├── train_nppai.py          # PyTorch training/export pipeline
├── server_backend.py       # FastAPI backend
└── CMakeLists.txt          # CMake build configuration
```

## Build

### CMake

The native project currently targets Windows and C++17.

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release
```

The plugin also contains a Visual Studio project used by the existing GitHub Actions build.

### Requirements

- Windows
- Visual Studio / MSVC
- CMake 3.10+
- C++17 compiler
- DirectX 11 development libraries
- OpenMP support for the optimized CPU path

For the Python tooling:

```powershell
python -m pip install -r requirements.txt
```

## Installing the Notepad++ plugin

Release tags publish ready-to-install ZIP packages for **x64**, **x86**, and **ARM64**.

1. Download the ZIP matching your Notepad++ architecture from the GitHub release.
2. Extract the included `NppAI` folder into the Notepad++ `plugins` directory. A typical x64 installation ends up with `C:\\Program Files\\Notepad++\\plugins\\NppAI\\NppAI.dll`.
3. Restart Notepad++ and confirm that NppAI appears in the **Plugins** menu.
4. Provide a compatible `.nppai` model as described in [Model format](docs/model-format.md). Model files are intentionally not bundled with the plugin package.

The release ZIP contains the plugin DLL plus README/model-format documentation. The release workflow builds the binaries from the tagged commit, so downloadable packages correspond directly to the published source revision.

## Running the native smoke test

The test executable expects a model file:

```powershell
build\\Release\\TestEngine.exe models\\NppAI-model-v1.nppai
```

You can also pass another model path:

```powershell
build\\Release\\TestEngine.exe path\\to\\model.nppai
```

A missing model is treated as a test failure instead of a successful run.

## Python model validation

```powershell
python test_python.py
```

The script validates loading the custom `.nppai` representation into the matching PyTorch architecture and performs a generation smoke test.

## Backend

The optional FastAPI backend provides endpoints for:

- submitting correction/training samples;
- checking model availability;
- downloading a model;
- starting background retraining after a configurable sample threshold.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run locally:

```powershell
python server_backend.py
```

> **Security note:** do not expose the development backend directly to the public Internet without authentication, rate limiting, request-size limits, TLS and an explicit data-retention/privacy policy.

## Training pipeline

```text
training dataset
      │
      ▼
train_nppai.py
      │
      ▼
PyTorch checkpoint
      │
      ▼
.nppai model
      │
      ▼
NppAI C++ inference
```

## Important technical notes

### AVX2

The x64 build keeps the baseline x64 ISA and selects AVX2/FMA tensor kernels at runtime. The optimized path is enabled only when CPUID reports AVX2, AVX and FMA support and the operating system reports XMM/YMM state support through OSXSAVE/XCR0. Otherwise NppAI uses the scalar CPU path.

### GPU execution

The DirectX 11 path is a workload-specific optimization and has a CPU fallback. The benchmark suite can force identical FP32 workloads through CPU and DirectX 11 paths to compare median latency and maximum absolute output error.

### Quantization

Selected weight matrices can be converted from FP32 to symmetric INT8 using a per-tensor scale. Accuracy and latency should be evaluated against FP32 before treating quantization as a production optimization.

### Telemetry and learning

The current backend implements **centralized collection and retraining of submitted correction data**. It should not be described as true federated learning unless decentralized training/aggregation is implemented.

Similarly, the correction dataset is closer to supervised fine-tuning / human-correction data than a complete RLHF pipeline.

## Code-generation quality evaluation

The [attention/position comparison](docs/attention-architecture-experiment.md)
and experiments #145–155 are integrated into `main` (14 October 2026).
In the recorded single-seed comparison, two-model staged generation exactly
matched 13/28 original test references with single-head attention, 14/28 with
four absolute-position heads, and 0/28 with four rotary heads. All three arms
matched 0/28 on both novel wording probes. These are exposed synthetic tasks;
multi-head/RoPE checkpoints remain PyTorch-only and no checkpoint is promoted.
See the [roadmap](ROADMAP.md#5-model-quality-and-training) for remaining wording
robustness and native FP32/INT8 validation work.

The [identifier copying experiment](docs/identifier-copy-training.md) compares
two loss weights on the same frozen random-name dataset. Complete reference AST
matches remain separate from syntax and from signature-guided generation.

The [Notepad++ Python signature mode](docs/notepad-python-signature-mode.md)
is enabled by `@python-signature name(args)` on the first prompt line, followed
by the task description. The application preserves the header while the model
streams its body; ordinary prompts retain their existing behavior.

[Signature-guided generation](docs/signature-guided-generation.md) can enforce a
task's explicit function header before model-generated body completion. Its raw
and scaffolded results are measured separately; it is an experimental CLI path.

The [identifier-distribution experiment](docs/instruction-identifier-ablation.md)
improves synthetic test syntax to 15/28 locally, but requested identifiers remain
wrong. These outputs are not counted as successfully completed tasks.

The [frozen broader dataset experiment](docs/broader-instruction-experiment.md)
uses 576 synthetic training records and 28 unseen test tasks. Its actual failures
and held-out loss are recorded alongside the dataset hashes.

The [instruction-training experiment](docs/instruction-training-experiment.md)
records an actual 2000-step comparison: nonempty responses improved, but held-out
syntax remained 0/3. Training-example reconstruction is reported separately.

See [reproducible demo and measured generation evidence](docs/demo-evidence.md)
for the current baseline, exact-run reproduction, and package smoke-test scope.

An experimental [offline evaluation harness](docs/code-quality-evaluation.md) scores submitted Python code against unit-test fixtures. It does not yet invoke NppAI or measure the actual model's coding quality; the included fixture tests the harness only.

## Testing

Current validation includes deterministic native unit tests, model-loader and tokenizer tests, end-to-end generation coverage, Python validation, and CI performance regression checks.

See:

- [Testing](docs/testing.md)
- [Architecture](docs/architecture.md)
- [Model format](docs/model-format.md)

The GUI-free DLL contract and host-level Notepad++ integration harness are covered in CI. The host harness exercises runtime initialization, required plugin exports, notifications, command callbacks, active-editor selection, and Scintilla calltip interaction without requiring the Notepad++ GUI.

## Project maturity and limitations

NppAI is an **experimental** local coding assistant, not a validated production-grade autonomous programmer. Green CI confirms tested build configurations and deterministic fixtures; it does **not** establish model usefulness on real repositories, security of public deployment, or compatibility with every Notepad++ installation. A compatible `.nppai` model is required and is not included in release ZIPs.

The repository's [detailed roadmap](ROADMAP.md) tracks implemented features, remaining model-quality evaluation, release-readiness work, and evidence gaps. In particular, the RAG evaluation helper uses a bounded structural JSON reader for its fixture fields and validates JSON syntax, but is not a general-purpose schema validator.

## Roadmap

- [x] Runtime CPU feature detection for AVX2/FMA with scalar fallback
- [x] Deterministic tensor unit tests
- [x] Model serialization/version validation
- [x] Reproducible CPU/GPU benchmarks
- [x] Better tokenizer test coverage
- [ ] Validate public backend deployment hardening end-to-end (policies and controls are documented/implemented; production review remains)
- [x] Model integrity checks / hashes
- [x] Formal release process
- [x] Improved RAG indexing and retrieval
- [x] Experimental MoE inference and benchmark coverage (not a production quality claim)
- [x] Broader quantization correctness and accuracy coverage
- [x] GUI-free Notepad++ plugin DLL contract test
- [x] Host-level Notepad++ integration test

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md).

## License

See [license.txt](license.txt).

---

Built as an independent C++ / AI systems project with a focus on understanding the underlying implementation.


## Performance

NppAI includes an optional tensor microbenchmark for the FP32 and quantized INT8 matrix-multiplication paths. Benchmarks are built separately from correctness tests, and GitHub Actions publishes the raw measurements as the `tensor-benchmark-x64` artifact.

Current multi-size reference measurement from GitHub Actions run #177 (`windows-2022`, x64 Release):

| Matrix size | FP32 median | INT8 median | FP32 / INT8 | Measured INT8 difference |
| --- | ---: | ---: | ---: | ---: |
| 128x128 | 0.00326375 ms/op | 0.00336575 ms/op | 0.9697x | 3.0% slower |
| 256x256 | 0.009333 ms/op | 0.009182 ms/op | 1.0164x | 1.6% faster |
| 512x512 | 0.034189 ms/op | 0.029556 ms/op | 1.1568x | **15.7% faster** |
| 1024x1024 | 0.130766 ms/op | 0.110612 ms/op | 1.1822x | **18.2% faster** |

The CI result is the median across **5 independent benchmark processes** for each matrix size. Inside each process, every size performs a warm-up and reports the median of **7 timed series**. Iteration counts are scaled by matrix size (400, 200, 100, and 50 iterations per series for 128, 256, 512, and 1024 respectively) to keep CI runtime practical.

In this CI measurement, INT8 is slightly slower at 128x128, approximately even at 256x256, and increasingly faster at 512x512 and 1024x1024. This indicates that the current SIMD INT8 kernel benefits more as the matrix workload grows; it should not be interpreted as a universal speedup for every shape or CPU.

The INT8 kernel was progressively optimized by moving quantization scaling outside the SIMD accumulation loop, widening the AVX2 loop, using four independent accumulators, applying FMA, and reducing accumulators directly in SIMD registers.

Hosted GitHub Actions runners have variable hardware load, so these numbers are a reproducible CI reference rather than a hardware-independent performance guarantee. Performance changes should be evaluated across repeated runs instead of from a single timing sample.

The benchmark also runs deterministic **CPU vs DirectX 11 GPU** FP32 comparisons at 128x128 and 256x256. It reports median CPU/GPU latency, their ratio, and maximum absolute output error. CI preserves these measurements in the benchmark artifact; when a runner has no usable D3D11 hardware device, the GPU comparison is reported as unavailable instead of failing the build.

Build and run locally:

```powershell
cmake -S . -B build-bench -A x64 -DNPPAI_BUILD_BENCHMARKS=ON
cmake --build build-bench --config Release --target BenchmarkTensor
.\build-bench\Release\BenchmarkTensor.exe
```

Real-model candidate replay and pass/repair metrics: [evaluation protocol](docs/real-model-e2e.md). These metrics require recorded actual model outputs; fixture CI alone does not measure model competence.

First-attempt output collection from configured real local/online models: [real-model evaluation guide](docs/real-model-e2e.md). No model-quality results are claimed without recorded inference.

Bounded actual-model repair experiments (up to three Docker-validated attempts) are documented in [real-model evaluation](docs/real-model-e2e.md). Results require configured real models; CI fixture tests do not establish model quality.

Replay evaluation now reports separate first-attempt local/online metrics and rejects changing test suites between retries; see [evaluation guide](docs/real-model-e2e.md).

Frozen local-model functional smoke probes (actual inference, Docker-only execution and raw provenance): [protocol](docs/local-model-smoke.md). This is a six-task first-pass diagnostic, not evidence of broad coding competence.

Observed frozen local smoke result: **0/6 tasks passed** (2 syntax failures, 4 missing requested callables), with 52.6 ms median generation time on the CI runner. See [raw evidence and limits](docs/local-model-smoke.md#observed-outcome). No checkpoint promotion.

Matched function-name learning experiment: [fixed versus per-draw aliases](docs/resampled-name-training.md), with full-response supervision and explicit original-task retention. Experimental checkpoints are not promoted automatically.

Measured fixed/per-draw name continuation: both remain **0/88 requested names/signatures and complete reference matches** after 1024 updates. Syntax changes do not establish copying. [Raw reports and limits](docs/resampled-name-training.md#observed-outcome); no model promotion.

Frozen name-copy diagnosis separates conditional next-byte prediction, `def `-scaffolded name decoding and cached complete replies: [protocol](docs/name-byte-diagnostic.md). Assisted scores are not autonomous coding accuracy.

Observed name-byte diagnosis: all three frozen models score **0/132 complete development names**, even with gold preceding name bytes or a `def ` scaffold. Familiar `nc_` prefix prediction improves, but suffix copying remains weak. [Measured results and raw evidence](docs/name-byte-diagnostic.md#observed-outcome); no promotion.

Training-only source-aligned name-copy objective with a matched replay control: [protocol](docs/aligned-name-copy.md). Architecture and inference stay unchanged; improvement requires observed full-response results.

Measured aligned-copy objective: despite lower logged attention alignment loss, both arms remain **0/132 complete development names/signatures**. Same-host control reports reproduced byte-for-byte; historical reply replay passed. No checkpoint is promoted. [Results and raw evidence](docs/aligned-name-copy.md#observed-outcome).

Gated source-byte identifier-copy [protocol](docs/explicit-identifier-copy-protocol.md) now has an experimental Python implementation; [current status](docs/gated-copy-readout.md). Native integration and promotion remain separate gates.

Experimental gated copy readout, distinct `.nppaicopy` format and paired Docker functional probes: [implementation/protocol](docs/gated-copy-readout.md). Python-only; no default-model change or promotion.

Measured gated-readout result: conditional `nc_` test bytes improve 166/308→218/308, but complete development names/signatures remain **0/132** and separate Docker probes **0/6** for both arms. [Raw results and limits](docs/gated-copy-readout.md#observed-outcome). No promotion.

CPU diagnostics show exact same-host training repeats but divergence between recorded AMD/Intel runners after update 1. A separate single-thread deterministic profile also fails cross-runner equality. Raw failures, step hashes and provenance are retained; exact historical checkpoint identity blocks incompatible paired experiments. No baseline or model is promoted. See [diagnostic protocol](docs/training-reproducibility.md).

[First-step operator localization](docs/first-step-operators.md) checks whether tracing preserves the original update, then compares input/output fingerprints at observed ATen boundaries. This diagnoses numerical differences without changing the baseline or claiming improved code generation.

The first-step diagnostic localized a historical-profile difference to FP32 attention softmax for byte-identical inputs on recorded AMD EPYC AVX512/AVX2 hosts (max delta about 6e-8). That allocation did not include Intel; cross-vendor repeatability remains unresolved. The measured evidence and coverage limits are recorded in the operator protocol.

A later allocation with Intel coverage localized the portable-profile AMD–Intel difference to FP32 `sqrt` in AdamW for identical inputs (max 1 ULP). Forward/backward/clipping traces match up to that boundary. Both operator findings and original arithmetic remain recorded; no reproducibility fix or model promotion is claimed.

[Fixed CPU arithmetic variants](docs/arithmetic-variants.md) compares native, FP64-intermediate softmax, FP64-intermediate AdamW sqrt and their combination on frozen operator inputs and complete first updates. This is a bounded experiment; no default recipe or model promotion is changed.

Measured arithmetic variants make isolated operators and forward logits/loss agree with `both64` in a mixed AMD/Intel historical-profile allocation, but backward gradients and updated weights still differ. The portable allocation matched only observed AMD hosts. No candidate recipe is promoted; retained reports distinguish operator agreement from full-step repeatability.

[Both64 saved-tensor localization](docs/both64-trace.md) traces effective forward/backward operators and autograd save-time tensors on the fixed first batch, requiring full traced/untraced equality before interpreting CPU differences. Measured both64 tracing localizes a historical-profile AMD-only difference to loss-phase FP32 log-softmax, whose output is saved by autograd despite equal logits/scalar loss. Portable both64 agrees for the complete first step on the recorded Intel/AMD allocation. Full-training repeatability remains unproven; no recipe promotion.

A second both64 trace allocation includes historical-profile Intel/AMD pairs: some first differ at loss log-softmax, while two pairs first differ at backward FP32 matrix multiplication with matching prior events and saved tensors. Portable both64 again matches the observed complete first step across Intel/AMD. Both boundaries and allocation-specific verification limits are retained in the trace protocol; full-training repeatability remains open.
