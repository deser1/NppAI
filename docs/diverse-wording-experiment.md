# Four-form versus eight-form training wording experiment

All three attention arms failed both novel wording probes in PR #158's actual
replay audit. This next fixed experiment changes training wording coverage while
retaining the existing single-head absolute-position architecture. It does not
select an architecture using the previously exposed test scores.

Both arms train separate header/body models from scratch for 10,000 updates per
model: seed 42, context 256, dim 64/hidden 128, two layers, batch four, AdamW
0.001, clip 1, body identifier weight 1. No whitespace offsets, initialization
from an earlier checkpoint, early stopping or test-guided checkpoint selection.

| Policy | Training forms | Derived records |
| --- | --- | ---: |
| four_forms | Original, Please prefix, Write Python code prefix, code-only suffix | 6144 |
| eight_forms | All four control forms plus the four additions below | 12288 |

Additional forms are fixed before this run: `Python task: ` prefix,
`Implement this function in Python: ` prefix, `Please produce code: ` prefix,
and ` Provide an implementation.` suffix. Frozen V3 source records, targets,
IDs and name splits remain unchanged. Derived IDs, policy, forms, counts and
training hashes are recorded. All full training sequences must fit context 256.

Neither training policy contains the exact `Please write Python code. ` or
`Provide a Python implementation: ` probe prefix. Their words overlap with
training; this tests composition of already exposed synthetic instructions,
not an untouched general-language benchmark. Header/full-body training targets
differ; budgets match optimizer updates, not tokens, compute, sampled examples
or exposure per source record. One seed remains a limitation.

Replay both fixed pairs on the same six forms and all 16 validation/28 test
records, with raw uncorrected generated headers, no fallback, source checkpoint
hashes and strict expected augmentation policy/count checks. Save all four
checkpoints and both raw replay reports. Audit raw per-case and aggregate
metrics; record minimal novel-wording smoke failures explicitly. Report all
arms rather than choosing a best checkpoint from these exposed tasks.

Successful CI means valid execution/evidence, not successful model generation.
No checkpoint promotion, native INT8 claim, production change or generated-code
execution. Results are pending the measured CI run.
