# Source-start selection: fixed paired continuation

Status: fixed protocol measured and audited; margin intervention rejected for promotion.

The frozen oracle readout found 30/88 exact exposed alias replies with supplied
start positions versus 0/88 autonomous, and 36/88 with both interventions; all
variants remained 0/6 functional. This motivates one trainable intervention,
not a claim that identifier fixes establish semantic correctness.

## Fixed comparison

Start from the exact PR #198 sequential checkpoint in original artifact
12102426032, archive SHA-256
`35d8a994d589fbe7776013e20a8d38a3c7e5281195e2f1fb6c4387f1f1d4e074`.
Require byte/event replay of all 228 retained autonomous initialization replies.
Only `copy_q.weight` and `copy_k.weight` receive gradients. Vocabulary/hidden
weights and learned exit parameters remain frozen and are hashed every update.

Both arms use the first 256 batches of the existing diverse TRAIN schedule:
512 original and 512 resampled alias examples. Labels come from the first name
byte in TRAIN responses and the matching prompt-only source byte. Development,
train-family probes and functional tasks never enter optimization.

- Control: mean cross entropy for source start across eligible prompt bytes.
- Treatment: identical control plus a coefficient-1 hinge penalty
  `relu(1 + largest non-gold eligible source score - gold source score)`.

Margin, coefficient, optimizer, seed/profile, data and duration are fixed in
`evaluation/start-selection-v1.json` before the run. Both arms use AdamW
lr .001, betas .9/.999, eps 1e-8, weight decay .01 and norm clipping 1.
Hard negatives exclude gold, padding and generated response bytes. The highest
competing position is chosen within each TRAIN forward pass; no evaluation
result changes the objective. This compares a loss factor, with equal updates
and examples, not guaranteed equal compute.

## Evidence and gates

Repeat each complete training twice and require identical every-update traces,
exported bundle bytes and autonomous responses. Retain loss parts, batch hashes,
protected/base/exit identities and parameter hashes. Verify export/load equality.
Evaluate complete replies using the unchanged sequential decoder, plus existing
train-family probes and six Docker functional regressions. No oracle starts,
supplied response prefixes or output rewriting are used in these measurements.

Report denominators for every group and original-hit retention against the exact
initialization. The four historically protected hits are contained in the initial
14/16 original TRAIN exact replies; report both original-set retention and gains.
A candidate must preserve prior exact hits, improve autonomous alias/full-code
results over the matched continuation and pass separate functional gates before
any later independent evaluation. No model promotion in this pilot.

One seed, one predeclared coefficient and previously exposed inputs limit the
conclusion. Native FP32 pointer heads have no cross-CPU reproducibility certificate.
Do not tune based on development results; retain failures. The recipe changes
start supervision only; any residual signature/body errors remain visible.

```sh
python -m unittest discover -s tests -p test_start_selection_training.py -v
python scripts/start_selection_training.py --archive training.zip --output measured --image sha256:LOCAL_IMAGE_ID
```

Dedicated workflow: `.github/workflows/start-selection.yml`. Measured results follow.

## Measured results

Run [39133059499](https://github.com/deser1/NppAI/actions/runs/39133059499)
on source `70e74edd32a14c8d3cc390cfa31f4da91b403987` passed initialization
replay, both full training repetitions, exported checkpoint equality,
autonomous response repeats and unchanged base/exit/evaluation checks.

| Arm | Exact development | Exact aliases | Original TRAIN retained | Docker |
| --- | ---: | ---: | ---: | ---: |
| Initialization | 27/132 | 0/88 | 14/14 | 0/6 |
| Source cross entropy | 28/132 | 1/88 | 14/14, +1 new | 0/6 |
| Additional hard-negative margin | 28/132 | 1/88 | 14/14, +1 new | 0/6 |

Both arms' only exact development alias is `test-smallest-13-2` in test/renamed.
Both retain 11/64 exact new TRAIN-family probe replies. Wrong source-start
observations fall from 86/88 to 82/88 exposed alias replies in both arms;
two replies still diverge before the name. These observations are from raw
retained events; corrected source starts alone do not establish exact code.
The treatment changes some outputs but does not improve the declared exact
reply or functional outcome against control. No tuning or promotion follows.

The functional outputs still include missing callables, invalid syntax and an
empty reply. Full source/diagnostics remain in the retained report. The margin
arm additionally emits `def t below_31(values): ...`, a syntax failure, so a
small aggregate change must not be described as reliable code generation.

## Retained evidence

`docs/evidence/start-selection/raw-records.zip` preserves all original JSON and
JSONL members: complete replies/events, every-update traces for both runs of
both arms, complete report and original file manifest. `retention.json` pins the
original artifact and retained ZIP SHA-256 identities and excluded checkpoint
hashes. Original artifact 12139425244 has archive SHA-256
`3aaa698c87ab4e28967151b5100e4ef489509f33155c3dabe0adf28a383d3784`.

The importer verifies original ZIP CRC and every file digest, byte equality of
repeated exported bundles/traces, and actual frozen base/exit bytes inside the
exported bundles. Binary weight bytes remain in the original expiring Actions
artifact; retained hashes cannot recreate expired weights. Offline audit checks
all train batch hashes and loss parts, frozen parameter identities, both trace
repeats, all response metrics, original-hit retention and functional totals.
It does not rerun inference, optimizer updates or Docker.

```sh
python scripts/audit_start_selection.py
python -m unittest discover -s tests -p test_start_selection_evidence.py -v
```

Conclusion: retain the fixed negative comparison and diagnose residual start
transfer/termination failures on frozen traces before another mechanism change.
Independent broader evaluation remains reserved for a frozen qualifying candidate.
