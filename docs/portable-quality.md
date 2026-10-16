# Frozen portable checkpoint quality

The readout is fixed before inference in `evaluation/portable-quality-v1.json`.
It measures the exported both64 checkpoint from the recorded 1024-update mixed
Intel/AMD run, separately from numerical repeatability. There is no training,
search, repair, reference prefix or source-copy adapter during evaluation.

The paired control is **portable native**, not the historical control. Both
checkpoints come from the same AMD EPYC 9V74 job (`portable-full-1`, run
[38959870764](https://github.com/deser1/NppAI/actions/runs/38959870764)). The original
artifact ID, ZIP SHA256, checkpoint SHA256s and exact two-repeat bytes are checked
before loading. The artifact expires after 14 days; hashes alone cannot recreate
weights after expiration. No checkpoint is made a default or promoted.

Both exports use native FP32 Python inference under `portable-cpu-v1`: context
256 bytes, greedy decoding, at most 160 new bytes, EOS byte 0, empty assistant
prefix. The **training** candidate used FP64-intermediate attention softmax and
AdamW sqrt; export stores weights, not that arithmetic dispatch. This readout
does not establish both64-forward inference or native C++/INT8 compatibility.

The 132 development cases comprise original, renamed and matched-prefix views
of validation/test tasks; the 32 training cases are a disclosed regression
diagnostic, not held-out evidence. Every complete reply and recomputable syntax,
requested-name/signature, reference-AST and exact-response score is retained.
The six separately authored functional probes are independent of optimization,
but previously exposed and correlated with simple-function families. They are
not a fresh holdout or broad programming benchmark. Generated code runs only in
the existing restricted Docker validator, first syntax then functional tests;
validator infrastructure errors fail the measurement rather than count as model
failures. The immutable local image ID and suite hashes are retained.

Run from repository root with the portable environment variables set before
importing Torch:

```sh
python scripts/portable_quality.py --archive training.zip \
  --dataset evaluation/instruction-v3 \
  --tasks evaluation/gated-copy-probes-v1/tasks.json \
  --output measured --image "$NPPAI_VALIDATION_IMAGE"
```

The workflow pins Python 3.12.15, Torch 2.14.1+cpu and NumPy 2.5.2. It preserves
partial raw reports on failure, records actual CPU/runtime provenance and checks
that evaluation left model parameters unchanged. Workflow success means the
measurement completed, not that the model passed quality gates.

## Measured result: promotion rejected

[Run 38969403072](https://github.com/deser1/NppAI/actions/runs/38969403072)
completed on AMD EPYC 7763. The original artifact ZIP SHA256/CRC, checkpoint
digests, every complete-response metric and byte-exact report reconstruction
were independently checked. Full raw replies, Docker diagnostics and runtime
provenance are retained in `docs/evidence/portable-quality/`; its manifest binds
the original artifact and reconstructed report SHA256.

| Fixed readout | Portable native | Portable both64 |
|---|---:|---:|
| Development single-function syntax | 111/132 | 111/132 |
| Development requested name | 0/132 | 0/132 |
| Development requested signature | 0/132 | 0/132 |
| Development reference AST / exact reply | 0/132 | 0/132 |
| Original training requested signature | 5/16 | 5/16 |
| Original training exact reply | 4/16 | 4/16 |
| Renamed training signature / exact reply | 0/16 | 0/16 |
| Separate functional task passes | 0/6 | 0/6 |

Both arms have five functional failures and one empty reply. The six functional
outputs match exactly between arms. One of 164 development/training replies
differs (test matched-prefix), without changing any scored result; equal totals
do not imply identical generations. Parameters were unchanged during evaluation.
The result does not establish a quality gain or broad non-regression from the
training arithmetic change. Syntax alone is insufficient for promotion.

The next quality step is a dedicated **training-only** identifier-copy curriculum
and gate diagnosis, followed by complete-code/regression checks. A genuinely
fresh, broader functional evaluation is still required after freezing a candidate;
these exposed tasks must not be relabeled as a fresh holdout. Hardware-specific
certification remains a separate inconclusive gate.
