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

The optimized x64 build currently enables AVX2 instructions. **x64 alone does not guarantee AVX2 support on every CPU.** Runtime CPU-feature detection and a scalar/SSE fallback are planned improvements.

### GPU execution

The DirectX 11 path is a workload-specific optimization and has a CPU fallback. It should not be considered universally faster until reproducible benchmarks are available.

### Quantization

Selected weight matrices can be converted from FP32 to symmetric INT8 using a per-tensor scale. Accuracy and latency should be evaluated against FP32 before treating quantization as a production optimization.

### Telemetry and learning

The current backend implements **centralized collection and retraining of submitted correction data**. It should not be described as true federated learning unless decentralized training/aggregation is implemented.

Similarly, the correction dataset is closer to supervised fine-tuning / human-correction data than a complete RLHF pipeline.

## Testing

Current validation consists primarily of native and Python smoke/integration tests.

See:

- [Testing](docs/testing.md)
- [Architecture](docs/architecture.md)
- [Model format](docs/model-format.md)

The next testing milestone is a deterministic unit-test suite for tensor operations, quantization and model serialization.

## Roadmap

- [ ] Runtime CPU feature detection for AVX2
- [ ] Deterministic tensor unit tests
- [ ] Model serialization/version validation
- [ ] Reproducible CPU/GPU benchmarks
- [ ] Better tokenizer test coverage
- [ ] API authentication and rate limiting
- [ ] Model integrity checks / hashes
- [ ] Formal release process
- [ ] Improved RAG indexing and retrieval
- [ ] Experimental MoE support

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

Current reference measurement from GitHub Actions run #169 (`windows-2022`, x64 Release, 256x256 matrix):

| Path | Median time |
| --- | ---: |
| FP32 | 0.0091935 ms/op |
| INT8 | **0.009024 ms/op** |
| FP32 / INT8 ratio | **1.01878x** |

The CI result is the median across **5 independent benchmark processes**. Each process performs a warm-up and reports the median of **7 series with 200 iterations per series**. In this measurement, the optimized INT8 path is about **1.9% faster than FP32** while retaining the lower-memory quantized weight representation.

The INT8 kernel was progressively optimized by moving quantization scaling outside the SIMD accumulation loop, widening the AVX2 loop, using four independent accumulators, applying FMA, and reducing accumulators directly in SIMD registers.

Hosted GitHub Actions runners have variable hardware load, so these numbers are a reproducible CI reference rather than a hardware-independent performance guarantee. Performance changes should be evaluated across repeated runs instead of from a single timing sample.

Build and run locally:

```powershell
cmake -S . -B build-bench -A x64 -DNPPAI_BUILD_BENCHMARKS=ON
cmake --build build-bench --config Release --target BenchmarkTensor
.\build-bench\Release\BenchmarkTensor.exe
```
