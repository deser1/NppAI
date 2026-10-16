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
measurement completed, not that the model passed quality gates. Results pending.
