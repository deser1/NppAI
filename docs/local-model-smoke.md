# Frozen local-model functional smoke

This experiment measures **actual first-pass model output** using the production
`DenseLocalBackend` FP32 NumPy runtime, greedy decoding, no assistant prefix and a
256-byte-token output budget. Six short prompts in
`evaluation/local-smoke-v1/tasks.json` fit the frozen model's 256-byte context.
Each has three independently authored functional cases, frozen before inference.
No reference answer, case arguments or expected values enter model prompts.

The checkpoint is the existing full-response continuation from training run
[38702282716](https://github.com/deser1/NppAI/actions/runs/38702282716), artifact
`nppai-body-loss-full_response`. The runner verifies its `loss-report.json` SHA-256
`bdac050d83b2efd9b3ad1bd12d6b991142120d4e0e2ffdc2994fb78b4bba4296` and checks
loaded model identity against that report's model digest. No training or model
promotion occurs. The artifact expires; archive it separately to repeat the run
after its retention period. Do not substitute another checkpoint silently.

The **Frozen local model functional smoke** workflow downloads this checkpoint,
resolves a local immutable Python Docker image, generates all six responses and
validates nonblank candidates with the production restricted Docker profile.
Generated source executes only in the container, without network, credentials or
host mounts. A cleanup check follows. Inference itself runs on the CI CPU.

`report.json` records the exact prompt and response, model and suite digests,
manifest digest, generation token count, finish reason and generation wall time,
Python/NumPy versions, workflow commit/run, image ID and bounded diagnostics.
Categories are passed, empty/whitespace output, syntax failure and functional
failure. Generation latency includes prompt processing and decoding, excludes
checkpoint load and Docker validation, and depends on runner hardware/load.
Infrastructure errors abort the run; they are not counted as model failures.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_local_model_smoke.py \
  --source frozen-model --image sha256:YOUR_LOCAL_IMAGE_ID --output report.json
```

The workflow succeeds when the measurement completes, even if **zero tasks pass**.
Read quality counts in the uploaded `nppai-local-model-functional-smoke` artifact;
a green workflow is not a model quality gate. Empty outputs remain recorded as
failures here, while the general replay evaluator deliberately rejects malformed
or empty input records.

These six newly authored smoke probes are inspired by known simple-function
limitations. They were not used to train this frozen checkpoint, but they are
small, correlated and selected with knowledge of prior results. They are not an
unbiased broad holdout. After running they are exposed development probes. There
is no repair loop, online comparison or Notepad++ UI measurement. The tiny context
cannot accommodate the existing full diagnostic repair prompt reliably; increasing
the runtime context without retraining is not a valid workaround.

## Observed outcome

Pending the first actual workflow run. Do not infer results from orchestration
unit tests. Broader task quality and an independently configured online model
remain open work.
