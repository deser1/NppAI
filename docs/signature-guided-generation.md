# Explicit signature-guided generation

The measured model frequently emits a training function name instead of the
requested identifier. This optional path supplies the exact Python function
signature extracted from the user's task as an assistant prefix, then asks the
model to generate the body. It does not rename generated code after scoring and
does not consult the reference answer while generating. The signature parser
supports one explicit name and plain distinct parameter identifiers; it rejects
defaults, nested expressions, ambiguous signatures and keywords.

The prefix is deterministic scaffolding, **not** text copied by the model.
Raw and guided outputs are reported separately. A CLI `TestEngine` optional fifth
argument reads a UTF-8 prefix file (maximum 4096 bytes), appends it to inference
context and includes it in the captured assistant output. A prefix alone cannot
pass the native nonempty-body check. Ordinary generation remains unchanged.
This is an experimental runner path, not an automatic Notepad++ UI feature.

## Actual local results on exposed v2 regression tasks

Using checkpoint SHA-256
`65a4a07ffb473cd0ee58573ef6df4afeb26bacd274916750f0b79054498af808`:

| Metric, 28 tasks | Free FP32 | Signature-guided FP32 |
| --- | ---: | ---: |
| Valid Python syntax (including standalone expressions) | 18 | 18 |
| Valid function with requested name | 0 | 18 |
| Complete reference AST match | 0 | 16 |

The [full report](signature-guidance-local-results.json) includes all raw outputs,
forced prefixes and scores. AST equality checks code structure, not actual
functional execution. The known-family templates are narrow; results on exposed
regression tasks do not establish general coding competence. The Windows workflow
checks native INT8 output separately against the measured CI checkpoint.

## Reproduce

```sh
python -m unittest tests.test_signature_guidance
python scripts/evaluate_signature_guidance.py \
  --model path/to/NppAI-broader-instruction.nppai \
  --dataset evaluation/instruction-v2/test.json --output signature-report.json
```

Add `--native path/to/TestEngine.exe` for the native comparison. The workflow
downloads the v2 checkpoint from run 38314121142; it must be preserved separately
after the training artifact's retention window expires. No generated code is
executed by this evaluation.
