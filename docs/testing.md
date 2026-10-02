# Testing

## Current tests

### Native targets

The CMake build currently defines five native test executables:

- `TestEngine` — model-backed generation smoke test (requires a supplied `.nppai` model);
- `TestTensor` — deterministic tensor math coverage;
- `TestModelLoader` — malformed/truncated model rejection;
- `TestTokenizer` — tokenizer fallback and BPE behavior;
- `TestGenerationPipeline` — tiny-model loading, BPE, forward pass, sampling, and detokenization.

Four self-contained targets are registered with CTest: `TensorMath`, `ModelLoaderValidation`, `TokenizerFallback`, and `GenerationPipeline`. The model-backed `TestEngine` smoke test is intentionally not registered because it requires an external model artifact.

Run the registered native suite after building:

```powershell
ctest --test-dir build -C Release --output-on-failure
```

Run the model-backed smoke test separately:

```powershell
build\Release\TestEngine.exe path\to\NppAI-model-v1.nppai
```

Exit codes for `TestEngine`:

- `0` — generation succeeded;
- `1` — model could not be loaded;
- `2` — generation returned no output;
- `3` — test output could not be created.

### Python validation

`test_python.py` loads the custom model format into the matching PyTorch model and performs generation.

## Testing roadmap

CTest integration and deterministic native coverage are now part of the project. Remaining work includes expanding coverage for quantization/dequantization, serialization compatibility, additional tensor shapes, and deterministic inference behavior across supported build targets.
