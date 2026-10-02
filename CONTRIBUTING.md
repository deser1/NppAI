# Contributing to NppAI

Thanks for contributing.

## Development priorities

NppAI is primarily a C++ systems/AI project. Changes should prefer:

1. correctness;
2. reproducibility;
3. measurable performance;
4. clear ownership of resources;
5. small, reviewable commits.

## Before opening a pull request

- Build the project on Windows with MSVC.
- Run the available native smoke test with a valid model.
- Run the Python validation script when changing the model format or training code.
- Document changes to the model format or inference behavior.
- Avoid committing model files, datasets containing private code, credentials or generated build output.

## Pull requests

A useful PR should explain:

- what changed;
- why it changed;
- how it was tested;
- whether performance or memory usage changed;
- whether the model format or API is affected.

For performance changes, include before/after measurements whenever possible.

## Commit messages

Prefer concise messages such as:

```text
feat: add model format validation
fix: handle empty tensor input
perf: optimize quantized matmul
docs: document inference pipeline
test: cover tokenizer edge cases
```

## Security and privacy

Never submit real secrets, credentials or private source code as test data.

See [SECURITY.md](SECURITY.md) for security reporting.
