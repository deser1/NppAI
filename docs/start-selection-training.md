# Source-start selection: fixed paired continuation

Status: protocol and implementation frozen before measurement. No new results.

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

Dedicated workflow: `.github/workflows/start-selection.yml`. Results pending.
