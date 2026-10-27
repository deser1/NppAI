# Neutral names and prompt-position continuation

This is a predeclared development experiment following the rejected
[body-training pilot](function-body-training.md). Its exposed grid passed 10/12
new descriptions with familiar signatures, but zero views with new names or
arguments; the candidate also lost two prior factorial hits. Those results
motivate positional controls, not a proven positional cause or another unchanged
training scale-up.

## Frozen inputs and paired intervention

The initialization is the permanent `model-factorial-v1` asset, SHA-256
`f8ef88dccbbd3280c21e5acd51c41c6744e6ba597aa63686f3f43fd96beac229`.
The [manifest](../evaluation/neutral-binding-v1.json) hashes all generated
training/development rows, including controls. The
[data builder](../scripts/neutral_binding_data.py) reads only the prior authored
**training** bank. No prior development/grid/released-evaluation row becomes an
optimizer target. Authored control ASTs are renamed; model responses are never
rewritten, normalized or repaired for scoring.

There are 144 reviewed training views: six existing operations × three neutral
names × two argument spellings × four English/Polish descriptions. Each neutral
name (`task_aa`, `task_bb`, `task_cc`) occurs with **all six operations**. The same
argument spellings are shared across families of the same arity. Training names
cannot identify an operation without the request, although correlations beyond
this small synthetic bank remain possible. This tests within-family transfer,
not new algorithm learning or general-purpose competence.

Both arms use the same seed 97, 1024 updates, batch eight, learning rate .0003,
AdamW (.9/.999, epsilon 1e-8, decay .01), gradient norm 1 and completion-only
per-example mean cross entropy including EOS. They start from identical weights
and use identical draws: two original direct records, two factorial training
records and four neutral-body records. Original replay retains its legacy prompt
format; other samples use the existing 256-byte prompt boundary.

- **fixed_position** uses the unshifted neutral requests.
- **position_jitter** adds 0, 8 or 16 leading ASCII spaces to each sampled neutral
  request using a separate deterministic RNG (seed 9701). The semantic bytes and
  response targets are identical; original/factorial replay is unchanged.

Padding before `[AI]` shrinks by the corresponding amount. Thus instruction and
signature bytes move, while the completion starts at byte 256 in both arms.
This changes whitespace as well as absolute input positions, so even a measured
gain would not prove a unique internal cause. Both candidates are exported and
hashed before any readout. There is no checkpoint/seed search or early stopping.
Schedules, source/data hashes, all losses and candidate identities are retained.

## Readout and acceptance boundary

The unchanged production profile (`--prompt-bytes 256 --python-functions`)
measures source and both candidates. Its mechanical requested-header assistance
is not learned name copying. No reference or expected answer enters inference.

There are **120 correlated development views**, not 120 independent problems:

- Eight cells cross known/new equal-length function name, known/new equal-length
  arguments and 0/8 leading spaces, for six families × two languages: 96 views.
  `task_dd` and `ee`/`ff` never occur in training. Equal byte lengths keep semantic
  field offsets fixed before the explicit whitespace shift.
- Two additional longer-field cells per family/language: 24 views. Longer names
  or arguments change both identity and byte positions; report these separately.
- The all-known unshifted cell reproduces 12 training requests and is a control.
  Wording is held familiar throughout this grid; no fresh wording claim follows.

The **primary gate** uses only the 24 both-new-name-and-argument views, combining
unshifted and shifted cells. Eligibility for a later independently authored
measurement requires at least **12/24 functional passes**, strict improvement
over both source and fixed-position control, every source original exact hit
retained (all 96 measured), and every source factorial functional hit retained
(all 22 measured, including both `policz_silnie` requests). Equal totals cannot
hide a lost prior row. Other cells are secondary diagnostics, not alternate gates.
All cell counts, raw requests/responses, errors, hashes and generation times are
reported, including failures. Twelve training probes are not all-bank accuracy.

All 264 authored training/development references must pass restricted Docker;
all 120 constant-None negative controls must fail. Actual development/factorial
outputs execute only in restricted Docker. Infrastructure errors abort; missing
Docker yields `not_measured`, never a functional pass. Model/source weights must
remain unchanged during readout and the audit reconstructs data, schedules,
metrics, cell summaries and the per-case decision.

A quality failure records `rejected` while sound infrastructure may pass CI.
Even a development gate pass does not promote weights. A later independent
suite is reserved until an eligible frozen candidate exists. Production weights,
decoder, default routing and release assets are unchanged by this experiment.

## Execution

The `Neutral binding and position training` workflow runs real CPU optimization,
controls, actual generation, Docker tests and the evidence audit. It retains the
complete `neutral-binding-v1` artifact for 90 days. Rejected candidate binaries
are research artifacts, not default model assets.

```bash
python scripts/neutral_binding_training.py --phase train --source source.nppai --output measured
python scripts/neutral_binding_training.py --phase measure --source source.nppai --output measured --image "$NPPAI_VALIDATION_IMAGE"
python scripts/neutral_binding_training.py --phase audit --source source.nppai --output measured
```

The image must be a locally available immutable `sha256:` Docker ID. The
protocol and implementation are committed before the first training run;
measured results, including rejection, will be recorded separately afterwards.

## Measured result: 27 October 2026

[Run 40459046953](https://github.com/deser1/NppAI/actions/runs/40459046953)
completed both fixed 1024-update arms, all **264 positive** and **120 negative**
Docker controls, actual model execution, the full evidence audit and cleanup.
Every positive passed and every negative failed. All source original/factorial
hits were retained by both candidates. All 24 completed PR checks passed, with
one optional review skipped, before [PR #218](https://github.com/deser1/NppAI/pull/218)
was merged. This is a measured development result, not a model release.

Each cell below has 12 correlated views, six families × English/Polish. Zero
means a familiar field, one a changed field; `shift1` adds eight leading spaces.
All fields have matched byte lengths except the separately labeled long cells.

| Cell | Source | Fixed position | Position jitter |
| --- | --- | --- | --- |
| name0-args0-shift0 (training control) | 0/12 | 12/12 | 11/12 |
| name0-args0-shift1 | 0/12 | 2/12 | 12/12 |
| name0-args1-shift0 | 0/12 | 0/12 | 0/12 |
| name0-args1-shift1 | 0/12 | 0/12 | 0/12 |
| name1-args0-shift0 | 0/12 | 12/12 | 12/12 |
| name1-args0-shift1 | 0/12 | 2/12 | 12/12 |
| name1-args1-shift0 | 0/12 | 0/12 | 0/12 |
| name1-args1-shift1 | 0/12 | 0/12 | 0/12 |
| Longer name, familiar arguments | 0/12 | 0/12 | 0/12 |
| Longer arguments, familiar name | 0/12 | 0/12 | 0/12 |

| Gate/retention | Source | Fixed position | Position jitter |
| --- | --- | --- | --- |
| Primary both-new identifier views, functional | 0/24 | 0/24 | 0/24 |
| Twelve training probes, exact | 0/12 | 12/12 | 11/12 |
| Original direct replies, exact | 96/96 | 96/96 | 96/96 |
| Factorial exposed probes, functional | 20/22 | 20/22 | 20/22 |

The paired intervention improves the two shifted, familiar-argument cells from
2/12 to 12/12. Equal-length new function names also pass with familiar arguments.
This is a local improvement in input-position robustness; it does not establish
arbitrary-name or arbitrary-argument competence. Header assistance remains
mechanical. Both candidates fail the longer-name cell and every new-argument
cell. The jitter candidate also loses one new training-control success relative
to fixed position: Polish sum of squares returns the count of negative values.
Earlier source original/factorial retention is nevertheless complete.

The predeclared primary gate remains **0/24**, so the candidate is **rejected**
for fresh evaluation. No released weights, decoder, plugin routing, tag or release
asset is changed. Do not aggregate the cells into independent problem-solving
accuracy; one cell repeats training requests and all views reuse six operations.
The longer-name failure remains confounded by name identity and byte length.

Raw failed outputs include `def task_dd(ee, ff): return abs(aa - bb)` and
`def task_dd(ee): return max(0, aa)`, while other rows return the `sum` callable.
These illustrate remaining body binding and operation/output failures, not a
complete causal diagnosis. The next step is a frozen read-only categorization
of all new-argument failures, followed by one predeclared binding intervention
with unchanged retention gates. An unchanged positional-training scale-up is
not justified by these results.

[Manifest](evidence/neutral-binding-v1/manifest.json) and
[complete raw JSON/schedules/traces](evidence/neutral-binding-v1/raw.zip) retain
unmodified CI bytes in git. The original
[artifact 12713922590](https://github.com/deser1/NppAI/actions/runs/40459046953/artifacts/12713922590)
has SHA-256 `ee7ec18378d7c3b34965a965230588a4469cc56edb919f2d192c40768d06a3ba`.
It also contains both candidate binaries, omitted from the git archive and
subject to 90-day Actions retention. Their identities are retained permanently:

- Fixed position: `6de739e62087033338717507ff7cf880026705db105d0b678847d5eec78d6919`.
- Position jitter: `b0bfd7fb5102dd927eaf21bdec78d4944568745d96519d2c90bcb5eff713e25a`.

The retained-evidence test reconstructs all groups, raw scores, cell counts,
control outcomes, schedules, model/source identities and the failed gate. A
separate local audit of the downloaded full artifact also passed. No cross-host
training repeat or bitwise reproducibility is claimed.
