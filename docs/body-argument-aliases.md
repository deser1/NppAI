# Full-response argument alias continuation

After #170 supplied correct headers but obtained0/336 complete reference matches
on new argument aliases, this experiment tests whether training full responses
with renamed arguments changes body generation. It is a preselected two-policy
comparison, not a checkpoint promotion or independent benchmark.

Both policies initialize every dense weight from the same four-form20k **body**
checkpoint in run38618406741. Its10k origin38582420228, preserved source pair,
model/report hashes, dataset manifest and cached continuation report are verified.
The historically named `initialize_header_weights` helper is a generic dense
all-weight loader; it does not replace a body with a header checkpoint.

Each policy uses6000 additional updates, batch4, seed42, AdamW0.001 with fresh
optimizer, gradient clip1, context256 and two CPU threads. Each batch contains
two original training examples and two function-renamed training examples. The
fixed policy keeps original argument names in the renamed half; the varied policy
uses #169's deterministic4/6-letter aliases. Pools contain6144 and18432 full
responses. No validation/test example is replayed; aliases and function names are
disjoint between training and development. The replay proportion is by examples,
not bytes or loss. Full response bytes including headers, bodies and EOS are
supervised; prompt tokens are masked, matching the source body's training method.

Reference bodies are reconstructed from frozen instruction-v3 by renaming Python
NAME tokens only. Strings and identifier substrings are preserved. Every training
frame fits256 including EOS: maxima184(original),196(fixed renamed),200(varied
renamed). All1100 development views also fit, maximum233. There is no truncation,
length-based filtering or fresh-probe access.

Evaluation before and after training uses the same1100 development views:
44 original,528 function-renamed,528 argument-aliased. They are correlated views
of exposed synthetic task families, not1100 independent tasks. Two arms are scored
separately: autonomous single-model output and complete output with a signature
supplied by the prompt-only parser. Supplied signatures are not learned copying.
The model generates at most160 new bytes per body call in either arm. Raw outputs,
syntax/name/full-signature metrics, reference AST/exact matches, per-name-length
counts and paired retained/lost/gained outcomes are retained. AST/exact matches
measure reference structure, not semantic correctness or execution.

After export, both arms on every development view must reproduce the in-memory
model's outputs. Independent evidence auditing rebuilds full references and pools,
recomputes scores from raw output, rejects missing/duplicate cases and checks
budgets, initialization and paired outcomes. Fixed seeds/data do not guarantee
bit-identical retraining; reload identity is a distinct within-run check.

Workflow artifacts contain model, full datasets/manifest, before/after raw report,
audit and preserved source models/reports. No generated source is executed, no
native model conversion or default checkpoint replacement occurs.


## Measured15 October2026

First matched run[38696129410](https://github.com/deser1/NppAI/actions/runs/38696129410)
at implementation`c02babd2fbafff0b56e45569a97bb2306f966dff` passed both policy
jobs. Each ran all18 tests, including actual PyTorch all-weight initialization and
full-body/EOS supervision checks. Both evidence audits and every export/reload
output check passed. Standard CI_build38696129295 also passed all8 jobs.

Both policies reproduced **identical raw before outputs**, matching #170's assisted
baseline. Complete assisted reference scores below are **AST/exact**, with a
signature supplied from the prompt. They are not autonomous model accuracy.

| Development group | Views | Same20k source | Fixed arguments after26k | Varied arguments after26k |
|---|---:|---:|---:|---:|
| Validation/original |16|14/14|15/15|13/13|
| Test/original |28|15/15|16/16|12/12|
| Validation/function-renamed |192|84/82|164/159|76/76|
| Test/function-renamed |336|74/72|153/150|71/71|
| Validation/argument-aliases |192|0/0|0/0|0/0|
| Test/argument-aliases |336|0/0|0/0|0/0|

All autonomous requested function/full-signature/reference-AST/exact counts are
zero before and after in every group and both policies. The body checkpoint does
not become an autonomous name-copying model through this continuation.

Fixed training retains all15 prior original **test** exact hits and gains one
(`test-length-13-2`); varied retains12/15, loses3, gains0. On original validation,
fixed retains13/14, loses1 and gains2, so full original-view retention is **not**
achieved. Varied retains11/14, loses3 and gains2. On function-renamed test views,
fixed retains69/72, loses3, gains81; varied retains46/72, loses26, gains25.

Assisted exact matches by requested function-name length on renamed test views:

| Name bytes | Views | Source20k | Fixed26k | Varied26k |
|---|---:|---:|---:|---:|
|9|112|52|62|47|
|12|112|10|44|11|
|15|112|10|44|13|

New-argument assisted full-output syntax/signature rises141→219/336(fixed) and
141→270/336(varied), but AST/exact stays zero. Syntactic validity and a supplied
signature are insufficient for body binding. For the explicit addition task
`cp_bo9wlp(ymci, xyvvel)`, the frozen reference body is
`return ymci + xyvvel`. The source instead emits`return len(values)`; fixed training
emits`return a + b`; varied emits`return len(dum)`. The latter outputs retain a
correct supplied header but refer to unrelated names or operations. They were
parsed/scored, never executed. Varied training also loses original square
`test-square-13-2`: correct`return x * x` becomes`return x -x`.

This run supports a narrow improvement in assisted body generation for function
renames under fixed arguments. It does not resolve new-argument binding, original
validation retention or autonomous copying, and does not justify promotion.
Further work should isolate variable binding rather than treating more alias
continuation as demonstrated progress. No checkpoint or quality gate changes.

### Provenance

- Body20k SHA256:`54905a910718248f5eadc74a789aeacc276c3455150a7da479fb48e5e60761e2`.
- Body source report SHA256:`93dbb8779537228a1e3fa162bde6344b8a7375e9e148798090d1ab54abbaf3a4`.
- Shared full development SHA256:`9764fd86e918de48a8c094bb2611db57f3f0dbe60527cb3e26bd027ad4aaae2a`.
- Fixed artifact11954246113, ZIP SHA256:`26fbb9f91bfb690f89a1a63a17d9a554648206f901d7e159f0be0ab41f040601`.
- Fixed raw report SHA256:`249a9fd8edf2b99da62c06a8fcb27c332921e233f180c8a14dbef2a2d26de4d9`.
- Varied artifact11954565376, ZIP SHA256:`50fb253f11089c73e2a8ce2c00e6d37ce7909b23463249fb4acb151521422eb2`.
- Varied raw report SHA256:`75f42a27d09c4d556fb9fd10026691bab4a46997bb954443cb787b67a45bf481`.

Both ZIP digests, exported checkpoint hashes and preserved source model/report
hashes were independently checked after download. Raw reports were re-audited
against frozen full references outside the training jobs. These are first-run
measurements; a later docs-triggered training run is separate evidence, not an
assertion of bit-identical training.
