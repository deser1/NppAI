# Frozen copy-transition and exact-hit drift diagnostic

Status: fixed readout measured. No training or weight promotion.

The namespace/retention pilot copied none of 64 new train-family aliases and lost
all four initial original-training exact hits in every arm. This readout localizes
observed prediction changes before choosing another mechanism or training recipe.
Its specification and 68-case hash are frozen in `evaluation/copy-transition-v1.json`.

Read the exact initial model and four final Python-only bundles from
[run 39019625587](https://github.com/deser1/NppAI/actions/runs/39019625587).
Pin original ZIP/report SHA256, repeated checkpoint/trace bytes and loaded
parameter hashes. Recreate the initial seed-4242 readout from the pinned portable
base and require its recorded initial parameter identity. Use the existing
portable CPU environment with native FP32 arithmetic and pinned runtime.

Reference files opened by case construction are **training only**. The 64 alias
views are the existing 16 train-family views under four correlated prefixes.
The four original cases are indices 2, 12, 13 and 14 of the frozen original
training snapshot, selected by exact initial replies before this readout.
Validation/test references and fresh functional probes remain unused. There is
no optimizer, repair, code execution, checkpoint selection or coefficient search.

## Separate decoding modes

1. Autonomous full replies: standard 160-byte greedy/context/EOS semantics,
   recording raw bytes, first wrong byte relative to the training reference and
   its phase (scaffold, name, name termination, signature, body or EOS). Require
   exact replay of previously retained outputs for all 340 model/case pairs.
2. Gold-prefix diagnosis: predict each name byte plus `(` given the correct
   preceding `def `/name bytes. Record gate, vocabulary/pointer/mixture target
   probabilities, their argmax bytes and pointer peak/aligned source positions.
   Recompute each truncated causal prefix and require matching next-byte scores.
3. `def `-scaffolded name decoding: only the fixed scaffold is supplied, followed
   by up to 32 greedy bytes until `(`, newline or EOS. Record pointer peaks and
   transitions, marking whether the reference prefix still matches. After the
   first mismatch no expected aligned source position is assigned.

For each lost original exact hit, read the initial model on the same shared
prefix at the final model's first error. Compare initial/final vocabulary and
mixture predictions. A final correct vocabulary argmax with a wrong mixture
argmax is an observed readout override; a vocabulary argmax changing from correct
to wrong is observed base drift. These comparisons are observational, not a
causal ablation or a proof that either component alone explains the failure.

Pointer peaks summarize a **soft distribution**, not a discrete copy cursor.
Duplicate source bytes may produce correct aggregate byte mass even when peak
positions jump. UTF-8 is mapped by bytes, and source `(` after whitespace is
located separately. Conditional accuracy and scaffolded decoding are not
autonomous complete-function correctness. No broad task-generalization claim.

Retain all raw per-case readouts and recomputable summaries. Check loaded and
post-readout parameter hashes, exact cached replies and causal-prefix agreement.
The original models remain in 14-day Actions artifacts; retained hashes cannot
reconstruct weights after expiry. The diagnostic does not change production
inference, architecture or model defaults.

## Measured copy readout

[Run 39021877939](https://github.com/deser1/NppAI/actions/runs/39021877939)
completed on AMD EPYC 7763 with pinned Python 3.12.15, Torch 2.14.1+cpu and NumPy
2.5.2. All 340 autonomous replies replay the frozen source outputs exactly.
Loaded parameter hashes match the recorded initialization/final traces and remain
unchanged after diagnosis. All conditional causal-prefix checks passed.

| Readout | Initial | Fixed | Diverse | Retained | Diverse + retained |
|---|---:|---:|---:|---:|---:|
| Autonomous new-alias full replies | 0/64 | 0/64 | 0/64 | 0/64 | 0/64 |
| `def `-scaffolded exact names through `(` | 0/64 | 0/64 | 0/64 | 0/64 | 0/64 |
| Gold-prefix correct name bytes | 171/752 | 297/752 | 335/752 | 282/752 | 320/752 |
| Gold-prefix peaks at aligned name positions | 19/752 | 318/752 | 356/752 | 325/752 | 352/752 |
| Gold-prefix next-position peak pairs | 7/688 | 206/688 | 240/688 | 212/688 | 231/688 |
| Gold-prefix correct `(` predictions | 46/64 | 37/64 | 40/64 | 37/64 | 42/64 |
| Scaffolded first error at name byte 0 or 1 | 48/64 | 48/64 | 45/64 | 48/64 | 43/64 |

Training improves some conditional position/byte observations, but conditional
name bytes still fail frequently and no scaffolded name completes. Pointer peaks
are often not sequential; duplicate-byte aggregation limits interpretation of
individual peak jumps. Most scaffolded failures already occur in the first two
name bytes, before termination can be the first error. Supplying correct prior
name bytes therefore does not represent autonomous copying success.

At the gold-prefix `(` query the trained models' mean gate is 0.544, 0.507,
0.505 and 0.472, versus initial 0.018. Their vocabulary argmax is correct while
the mixture is wrong in 27, 24, 26 and 22 of 64 termination queries respectively.
This is an additional observed exit/gate problem under supplied name prefixes;
it is not proof that termination explains the earlier free-decoding failures.

## Lost original exact hits

All four original hits remain exact for the initial model and fail in all four
trained models. Of those 16 model/case losses, **15 first differ inside the
function name** and one inside the body. The initial model predicts the correct
byte on every shared prefix at those errors.

| First-error observation | Fixed | Diverse | Retained | Diverse + retained |
|---|---:|---:|---:|---:|
| First error in name / body | 4 / 0 | 3 / 1 | 4 / 0 | 4 / 0 |
| Final vocabulary correct, mixture wrong | 3 | 0 | 1 | 0 |
| Initial vocabulary correct, final vocabulary wrong | 1 | 4 | 3 | 4 |

Four losses have an observed override of a still-correct vocabulary prediction;
their gate values are about 0.935, 0.933, 0.950 and 0.852. The remaining twelve
have vocabulary argmax drift from correct to wrong. The one body error has gate
about 0.004 and a changed vocabulary prediction. A head-only explanation or a
base-only explanation would miss part of this evidence. These local comparisons
do not constitute full-trajectory ablations or guarantee a repair.

Original artifact ZIP SHA256/CRC and all five weight-file hashes were checked.
`docs/evidence/copy-transition/manifest.json` reconstructs the exact report,
including every conditional/scaffolded point, raw reply and shared-prefix drift
comparison. Replay, probability mixture identities, source alignment, causal
deltas, first-error positions and summaries are independently audited. Tests
reject altered reference alignment, probabilities, replay/parameter claims and
summary counts. Weights remain in 14-day Actions artifacts; hashes alone do not
reconstruct them. No independent functional benchmark was rerun by this diagnostic.

The next controlled protocol should separate a frozen/protected vocabulary path
from a prompt-only sequential pointer and an explicit learned name-exit decision.
Predeclare baseline, causal/source-start/termination safeguards and autonomous
full-code plus exact-hit retention gates. Any monotonic cursor or protection rule
must be disclosed as a mechanism change and measured end to end before fresh
functional evaluation or promotion. Neither mechanism is implemented here.
