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
execution.

## Measured results (14 October 2026)

[PR #159](https://github.com/deser1/NppAI/pull/159) evaluated commit
`9ce47a81f30940bbe111722a5247e95c537dac3c` in
[run 38582420228](https://github.com/deser1/NppAI/actions/runs/38582420228).
All four training jobs passed 28 regression tests each; both replay/audit jobs
passed. The initial direct-script header import failure was fixed before this
measurement. All 12 latest-head PR workflows passed. After integration,
[main CI_build](https://github.com/deser1/NppAI/actions/runs/38583428878) passed
all eight jobs for merge commit `7a58453e9f12125365d49e651af3703dc9cdc58d`.

Complete staged responses exactly matching their references on the same 28 test
tasks (full-reference AST counts match these counts):

| Prompt form | Four-form control | Eight-form training |
| --- | ---: | ---: |
| Original | 13/28 | 0/28 |
| Please prefix | 9/28 | 0/28 |
| Write Python code prefix | 10/28 | 0/28 |
| Code-only suffix | 11/28 | 0/28 |
| Novel polite/Python prefix | 0/28 | 0/28 |
| Novel implementation prefix | 0/28 | 0/28 |

Validation exact responses on the four previously trained forms are respectively
10/16, 9/16, 11/16, 12/16 for the control; eight-form training scores zero on
all forms. Both novel validation forms are zero in both arms. Free generation
from the body model scores zero exact responses on all validation/test forms in
both arms. Original test header exactness is 20/28 control versus 0/28 eight-form.
Eight-form staged syntax on the four previously trained test forms is 26/28,
27/28, 28/28, 25/28; this does not establish requested-name correctness.

Both actual-output audits verified raw prompts, references, case IDs and
recomputed metrics, but both record `wording_smoke_passed: false`. The eight-form
policy did not repair either novel wording probe and harmed known-form reference
matching under this fixed seed and budget. It is not promoted. This does not
establish that broader wording training fails universally; changed exposure,
sampling and token budgets, one seed, correlated/exposed synthetic tasks and
two-model inference remain limits. No generated code was executed.

Replay artifacts preserve raw outputs/references, paired metrics, audits and
verified source checkpoints/reports:

| Policy | Replay artifact | Source report SHA-256 |
| --- | --- | --- |
| four_forms | [11903357806](https://github.com/deser1/NppAI/actions/runs/38582420228/artifacts/11903357806) | `ff970ff5d19dea784153a8ed64fe9eeb35e19a91df861a0fc53478f00aae965b` |
| eight_forms | [11904027489](https://github.com/deser1/NppAI/actions/runs/38582420228/artifacts/11904027489) | `ee572c2db3eab6caa34891f6deec36f596b316218ad8be220bbfcb77a45fdce4` |

The archive digests reported by GitHub are respectively
`7c740c4e86b4280b84543d788752b8f011c1c61126bc3a97fbc798499abff2fc`
and `b1b6af42e82231c2ef72ae34e4174e6234f50af5c5f21183d405943f2107c173`.
Workflow artifact retention is 14 days; this committed summary preserves
measurements and provenance, not the raw artifact contents.
