# Code-generation quality evaluation (experimental)

The offline evaluator measures **submitted Python solutions**, not model quality by itself. The included fixture is a smoke test for the evaluation infrastructure, not a representative coding benchmark.

## Run

Save a candidate implementation at `submissions/sum_positive.py`, then run:

```powershell
python scripts/evaluate_code_quality.py --manifest evaluation/code-quality/manifest.json --submissions submissions --output code-quality-results.json
python -m unittest discover -s tests -p test_code_quality_evaluator.py -v
```

The evaluator reports submission coverage, syntax success, unit-test success, and pass rate. Missing candidates count as failures. Exit code is nonzero if any case fails. Candidate code and fixture tests execute in subprocesses with a per-step timeout.

**Security:** This is NOT a sandbox. Execute only trusted candidate code on an isolated machine/container with no credentials, network access or sensitive workspace mounts. Python `-I` limits interpreter environment influence but does not prevent filesystem or network access.

## Required next steps before measuring NppAI quality

1. Add a versioned, licensed dataset of diverse real-world tasks and hidden/held-out tests.
2. Build a reproducible model inference adapter to save actual NppAI outputs as candidate files; pin model hash, decoding parameters and seed.
3. Evaluate compilable solutions, behavioral correctness, and repair iterations separately, with repeated runs and confidence intervals.
4. Compare to a documented baseline and publish artifacts; never equate infrastructure smoke tests with model competence.

## Import recorded model outputs

Capture actual model responses as a JSON array with records like `{"id":"sum_positive","output":"def sum_positive(numbers): ..."}`. Import into a **new empty** submissions directory:

```powershell
python scripts/import_generation_outputs.py --input recorded-outputs.json --manifest evaluation/code-quality/manifest.json --submissions submissions
python scripts/evaluate_code_quality.py --manifest evaluation/code-quality/manifest.json --submissions submissions
```

The importer accepts raw Python or a single fenced Python block. It does not call the model, strip explanations from raw text, or validate provenance. Record the model checkpoint hash, prompt, decoding settings and generation logs independently; only genuine captured model responses support model-quality claims.

## Capture actual NppAI model responses

Build `TestEngine` and provide a compatible local model checkpoint. The native smoke-test executable now accepts optional arguments: `TestEngine <model> <prompt> <output-file>`. Capture all manifest prompts with:

```powershell
python scripts/capture_model_code_outputs.py --engine build/Release/TestEngine.exe --model models/NppAI-model-v1.nppai --manifest evaluation/code-quality/manifest.json --output recorded-outputs.json --metadata generation-metadata.json
python scripts/import_generation_outputs.py --input recorded-outputs.json --manifest evaluation/code-quality/manifest.json --submissions submissions
python scripts/evaluate_code_quality.py --manifest evaluation/code-quality/manifest.json --submissions submissions --output code-quality-results.json
```

The capture script records the model SHA-256 and outputs, but does not fix a random seed or decoding configuration. These are actual generations only when run with a real checkpoint. CI mocks the capture adapter and does not run model inference. Do not interpret CI green as evidence of coding ability.
