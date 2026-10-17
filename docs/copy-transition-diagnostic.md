# Frozen copy-transition and exact-hit drift diagnostic

Status: fixed readout, measurement pending. No training or weight promotion.

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
