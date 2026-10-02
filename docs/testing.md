# Testing

## Current tests

### Native smoke test

`test_engine.cpp` loads a supplied `.nppai` model and performs a short generation.

It is intentionally an integration/smoke test rather than a unit-test framework.

Example:

```powershell
TestEngine.exe path\\to\\NppAI-model-v1.nppai
```

Exit codes:

- `0` — generation succeeded;
- `1` — model could not be loaded;
- `2` — generation returned no output;
- `3` — test output could not be created.

### Python validation

`test_python.py` loads the custom model format into the matching PyTorch model and performs generation.

## Testing roadmap

The next step is a deterministic test suite covering:

1. tensor shape validation;
2. matrix multiplication;
3. RMSNorm;
4. SiLU;
5. INT8 quantization/dequantization;
6. model serialization/deserialization;
7. tokenizer round trips;
8. inference determinism with a fixed seed.

A future CTest integration should run these tests without requiring a production model download.
