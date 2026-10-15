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

Actual inference and Docker validation completed in [run 38802296124](https://github.com/deser1/NppAI/actions/runs/38802296124)
on 2026-10-15. **0/6 tasks passed**, with no empty responses, two syntax failures
and four functional failures. All four syntactically valid responses defined a
different function name, so the requested callable was missing at the first
functional case. Their body semantics were therefore not established. Cases stop
at the first failure; the run did not execute all 18 authored functional cases.

| Task | Outcome | Generation time |
|---|---|---:|
| offset | Requested callable missing | 45.6 ms |
| distance | Syntax failure | 69.4 ms |
| first_even | Requested callable missing | 51.8 ms |
| ends | Syntax failure | 59.2 ms |
| bounded | Requested callable missing | 53.3 ms |
| count_negative | Requested callable missing | 50.4 ms |

Median generation latency was **52.6 ms** on that runner; this is not a portable
performance guarantee. Checkpoint SHA-256:
`161d6303c7e77f005cbabba5a1790b9d6a6dfc1812c121bfbb3f1cb6820e5b1a`.
The exact unmodified [raw report](evidence/local-model-smoke-report.json) is retained
in git (SHA-256 `39b77422945cf2ac4aa840f4a657a825bf4897981ff6cd943d0c3d0d4786f22c`).
Its evaluator commit is GitHub's temporary PR merge commit, not the branch head.
The downloaded artifact ZIP was verified against GitHub's digest
`73562623ef38e4f22b467c142ca05862718c3b67714ad5bb026d505dcaa11a41`.

This checkpoint failed the requested function contract on this small probe set.
It is not promoted. Improve signature copying and complete-function correctness
on training/development data before a separate broader evaluation. An online
comparison and bounded repair quality measurement remain open.
