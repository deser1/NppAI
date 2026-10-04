# Testing

## Current tests

### Native targets

The CMake build provides a model-backed smoke test plus a self-contained CTest suite covering the native inference engine, plugin integration, RAG, generation flow, and telemetry behavior.

The following 13 tests are registered with CTest:

- `TensorMath` — deterministic tensor math coverage;
- `ModelLoaderValidation` — malformed/truncated model rejection;
- `TokenizerFallback` — tokenizer fallback and BPE behavior;
- `GenerationPipeline` — tiny-model loading, forward pass, sampling, and detokenization;
- `AIPromptPipeline` — prompt construction and pipeline behavior;
- `PluginPromptActions` — plugin prompt action behavior;
- `GenerationContext` — generation-context selection;
- `PluginGenerationFlow` — integrated plugin generation flow;
- `GenerationStreamRouter` — streamed generation routing;
- `GenerationTracking` — generated-code tracking metadata;
- `TelemetryRedaction` — privacy-oriented telemetry redaction;
- `RAGRetrieval` — local RAG retrieval behavior;
- `PluginDllContract` — loads the built plugin DLL and validates the Notepad++ plugin boundary.

`PluginDllContract` verifies the required exports, plugin name, Unicode support, command-array availability/count, and a side-effect-free `messageProc` smoke check. It deliberately does not call `setInfo`: the current implementation performs runtime initialization and model loading there, which belongs in a real host-level integration test rather than an ABI contract test. The contract test runs without an interactive Notepad++ GUI, so it is suitable for the hosted Windows CI runner.

The separate `TestEngine` executable is a model-backed generation smoke test. It is intentionally not registered with CTest because it requires an external `.nppai` model artifact.

Run the registered native suite after building:

```powershell
ctest --test-dir build -C Release --output-on-failure
```

Run only plugin-related tests:

```powershell
ctest --test-dir build -C Release -L plugin --output-on-failure
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

CTest integration, deterministic native coverage, and a GUI-free Notepad++ DLL boundary test are complete. The remaining plugin-integration milestone is a true host-level Notepad++ scenario that can safely exercise `setInfo`, notifications, command callbacks, and editor interaction inside a real Notepad++ process. Additional engine coverage can continue to expand serialization compatibility, tensor shapes, quantization accuracy, and deterministic inference across supported build targets.
