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
