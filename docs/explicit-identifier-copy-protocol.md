# Explicit source-byte identifier copy: next controlled experiment

Status: **experimental Python implementation measured; acceptance gates not met, no promotion**.
See [implementation and measured status](gated-copy-readout.md).

## Motivation
The matched source-position attention auxiliary objective in PR #185 reduced its
training loss without improving autonomous complete development function names
(0/132). The next intervention must alter the **output distribution** rather than
only supervise an attention matrix.

## Candidate mechanism
At each autoregressive step compute normal vocabulary logits and a causal
source-pointer distribution over prompt byte positions. Map each source position
to its byte ID and sum pointer probabilities for duplicate bytes. Mix the two
distributions with a learned, bounded sigmoid gate:

`P(next_byte=b) = (1-g) * P_vocab(b) + g * sum_{i: prompt[i]=b} P_ptr(i)`.

The source pointer may attend **only to prompt positions**, never target tokens
or held-out reference answers. Keep EOS and special tokens on the vocabulary
path. Train using response next-byte negative log likelihood; optionally
supervise source positions for unambiguous identifier spans with a *predeclared*
coefficient, without using validation/test to choose it.

Do not replace arbitrary tokens by string substitution: the model must learn
when to copy and when to generate. UTF-8 multibyte characters and repeated
source bytes need explicit unit tests.

## Controlled comparison
1. Pin the same original checkpoint, training split, source examples, alias
   schedule, number of updates, seed, batch size, optimizer and context as the
   existing aligned-copy experiment.
2. Compare unchanged control against pointer-gated readout; use one predefined
   setting, no coefficient search or checkpoint cherry-picking.
3. Record initial predictions, raw complete outputs, training schedules,
   checkpoint hashes, runtime, and repeatability. Re-run the control on the
   same host and require byte-identical full reports.
4. Evaluate autonomous **full responses** separately from conditional
   next-byte and `def `-scaffolded diagnostics. Report exact requested name,
   full signature, parseable body, AST match, and original-task regression.
5. Keep exposed development views separate from a genuinely independent
   held-out evaluation. Never train on or tune to the independent holdout.
6. Use Docker-restricted functional checks on newly authored tasks before
   considering promotion; compare local/online backends only with equivalent
   tasks and explicit model provenance.

## Acceptance gates
- No prompt/reference leakage or future-token attention; deterministic
  duplicate-byte aggregation and correct UTF-8 handling.
- Existing inference/export compatibility is either preserved or a new
  versioned model format with safe fallback and tests is documented.
- Measured nonzero autonomous requested-name/signature improvements on
  previously unseen aliases **and** no unacceptable regression on original
  tasks; publish all numerator/denominator counts.
- Independent functional pass-rate and safety tests must be run before
  promotion. A better auxiliary loss or assisted decoding alone is insufficient.
- Do not merge a production-default change or promote weights based only on
  this design document.

## Implementation checklist
- [x] Implement experimental pointer/gate model and versioned Python-only export/load compatibility.
- [x] Add causal masking, duplicate-byte, UTF-8, EOS and gradient unit tests (actual execution in experiment CI).
- [x] Add paired pinned-control experiment and evidence audits.
- [x] Run actual PyTorch training and autonomous complete-code evaluation (0/132 complete development names).
- [x] Run separately authored Docker functional checks and publish failures/regressions (0/6 per arm; small correlated probes).
- [x] Reject promotion for this experiment: complete-name and functional gates remain unmet.
