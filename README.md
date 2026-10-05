# NppAI — Native AI Coding Assistant for Notepad++

NppAI is an experimental AI coding assistant for Notepad++ built around a custom C++ inference engine and a Python/PyTorch training pipeline.

The project focuses on **local inference, systems-level implementation, model formats, RAG, and developer tooling** rather than wrapping an external hosted AI API.

> **Project status:** experimental / active development. Model quality, performance and deployment characteristics are still being validated.

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

## Testing

Current validation includes deterministic native unit tests, model-loader and tokenizer tests, end-to-end generation coverage, Python validation, and CI performance regression checks.

See:

- [Testing](docs/testing.md)
- [Architecture](docs/architecture.md)
- [Model format](docs/model-format.md)

The GUI-free DLL contract and host-level Notepad++ integration harness are covered in CI. The host harness exercises runtime initialization, required plugin exports, notifications, command callbacks, active-editor selection, and Scintilla calltip interaction without requiring the Notepad++ GUI.

## Roadmap

- [x] Runtime CPU feature detection for AVX2/FMA with scalar fallback
- [x] Deterministic tensor unit tests
- [x] Model serialization/version validation
- [x] Reproducible CPU/GPU benchmarks
- [x] Better tokenizer test coverage
- [ ] Complete public-deployment API hardening (authentication policy, rate limiting policy, request-size enforcement, TLS termination, and data-retention/privacy policy)
- [x] Model integrity checks / hashes
- [x] Formal release process
- [ ] Improved RAG indexing and retrieval
- [ ] Experimental MoE support
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
