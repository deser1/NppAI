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
