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

## Collect first attempts from an actual model

Prepare `tasks.json` with independently authored prompts and hidden test suite
paths, for example:

```json
[{"task_id":"add-01","prompt":"Write a complete Python module defining add(a,b). Return only Python source.","cases_file":"examples/python-validation-cases.json"}]
```

Collect from a local dense checkpoint:

```bash
python collect_real_model_candidates.py tasks.json --mode local --model path/to/checkpoint --output local-candidates.jsonl
```

Or from an explicitly authorized online endpoint (do not commit credentials):

```bash
NPPAI_API_KEY=... python collect_real_model_candidates.py tasks.json --mode online --endpoint https://your-host/api/v1/generate --allow-remote --output online-candidates.jsonl
```

Then replay each JSONL with `evaluate_real_model_e2e.py` and an immutable
Docker image ID as described above. This collector records **only first
attempts**. It does not perform repair loops or measure repair quality; collect
repair attempts separately before claiming pass-within-three performance.
Local dense byte models may have limited context and output budgets. If a model
is not configured, no real model metrics can be reported. Do not commit model
outputs containing private source or prompts.
