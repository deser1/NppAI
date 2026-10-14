# Learned header followed by body generation

The header-only model produced 11/28 exact new test headers under original
prompts. This inference-only measurement composes that frozen model with the
simultaneous frozen 6000-step full-code model. Source run
[38325353585](https://github.com/deser1/NppAI/actions/runs/38325353585) contains
the two artifacts from commit `ab1f788037342bae681458a2759335c1a5d06953`.

The header model receives the original task and no assistant prefix. Accept
its raw output only if it is one valid LF-terminated declaration line and fits
the initial context; validate by parsing with an appended `pass`, never executing.
The body model receives that exact machine-generated line as its assistant
prefix. Acceptance never compares the name/parameters with the reference,
corrects a wrong name or substitutes a reference header. Malformed headers
are rejected with an empty staged result; there is no success-inflating fallback.

Compare final composed source with free generation from the same body checkpoint
on all 16 validation and 28 test records. Report accepted/exact headers separately
from nonempty output, syntax, requested function name, complete reference AST
and exact complete text. Generated source is AST-parsed, never executed. A correct
header with a wrong body is not task success. Record raw prompts, reference,
generated header, rejection decision and both complete outputs for every case.

Verify checkpoint hashes against their source reports, source manifests against
the frozen dataset, and dimensions/context against the expected dense model.
Preserve source checkpoints/reports with the new measurement. No new optimization,
test-guided training or model promotion. The two-model pipeline uses extra weights,
memory and inference compared with its single-model control; this is not a
matched training-budget comparison or an INT8 measurement. Body generation uses
a machine-generated prefix, so its conditional score must remain distinguishable
from single-model autonomous code generation.

```sh
python -m unittest tests.test_staged_generation
python scripts/evaluate_staged_generation.py --headers header-source \
  --bodies body-source --dataset evaluation/instruction-v3 \
  --output staged-experiment --source-run 38325353585
```

The workflow downloads the named artifacts from the fixed source run. If those
artifacts are unavailable, replay from the source copies in this measurement's
artifact; the evaluator fails rather than silently substituting new checkpoints.
