# Real-model end-to-end evaluation

The existing validated-generation tests exercise deterministic fixtures. To measure
actual model quality, collect **real model outputs** from local or configured online
inference, including each repair attempt. Do not put credentials or private source
code into the recorded JSONL.

Each JSONL record contains exactly:
```json
{"task_id":"add-01","attempt":1,"backend":"local","source":"def add(a,b): return a+b","cases_file":"examples/python-validation-cases.json"}
```

Use one task ID per independent task, with consecutive attempts 1–3 and no attempts
after a successful candidate. Cases must be independently authored and fixed
before model generation. The source must be the **complete proposed Python buffer**,
not merely a fragment. The replay checks syntax and then functional tests using
the same restricted local Docker validator as the plugin.

```bash
docker pull python:3.12-slim
IMAGE="$(docker image inspect python:3.12-slim --format '{{.Id}}')"
python -m unittest tests.test_real_model_e2e -v
python evaluate_real_model_e2e.py real-candidates.jsonl --image "$IMAGE" --output e2e-results.json
```

The evaluator reports tasks passed on the first attempt, within three attempts,
and recovered after an initial failure, with per-attempt diagnostics and suite
hashes. Docker errors/cancellation abort evaluation instead of counting as
model failures. This is a **replay evaluator**, not an automatic model invoker:
real model generation and recording must be performed separately. It does not
test the Notepad++ UI or prove the full autonomous agent loop; compare results
against held-out tasks and report model/version, seed and inference settings.

Do not merge or promote a model based on green evaluator unit tests alone.
