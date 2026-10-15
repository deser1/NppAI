# Body-only loss versus full-response loss

#171's full-response varied-argument training did not obtain a complete reference
match on new argument names. #172's prompt-only normalization recovered some
matches but damaged existing tasks and remained assisted. This experiment tests
a different training loss with no canonicalization or postprocessing.

Both arms start from the same verified20k four-form body checkpoint from
run38618406741 and its10k origin38582420228. All dense weights are loaded; each
arm uses a fresh AdamW optimizer, learning rate0.001, clip1, batch4, seed42,
context256, two CPU threads and6000 additional updates. The full-response control
is retrained here rather than borrowing a previous numerical result.

Use #171's **same varied-argument data** in both arms:6144 original and18432
renamed/argument-aliased full training responses. Each batch draws two original
and two renamed examples. Preserve the entire input sequence, including the
teacher-forced declared header. Full-response loss supervises header, body and
EOS; body-only loss masks header targets as well as prompt targets, while keeping
every body byte, indentation and EOS supervised. All model weights still train;
header embeddings can receive gradients through later body predictions.

Both arms use the same example order. Record a hash of all6000 ordered batches
and actual supervised header/body+EOS counts from the tensors. An independent
stdlib audit reconstructs the planned schedule and byte counts from frozen
training records. Full-response and body-only arms must have identical draw
hashes and body+EOS exposure; only header target exposure differs. The50% replay
ratio is by examples, not bytes. Mean cross-entropy averages active targets, so
masking also changes loss normalization and gradient allocation; this is not an
isolated proof about one internal mechanism.

All full frames fit256 with no truncation (training max200, development max233).
Training lookups use only train records; development aliases and function names
are disjoint. No test/validation replay or fresh-probe access occurs.

Evaluate before/after on the shared1100 correlated exposed views:44 original,
528 function-renamed and528 argument-aliased, not1100 independent tasks. Report
prompt-signature-assisted and autonomous complete output separately,160 greedy
new bytes per call. Supplied signatures are not learned copying. No canonical
normalization/restoration from #172 is used. Keep raw outputs, full signature,
reference AST/exact counts, per-length scores and paired retained/lost/gained
outcomes. AST/exact are structural reference metrics, not semantic correctness.

Every exported/reloaded output must match in-memory output on both inference
arms and all development cases. Evidence audits rebuild complete references,
recompute raw metrics and verify the training data,20k initialization,26k budget,
actual draw order and mask counts. Tests check UTF-8 byte boundaries, identical
inputs, prompt/header exclusion, body/EOS inclusion and actual loss gradients.

Artifacts contain trained checkpoint, full pools/development/manifest, raw report,
audit and preserved source models/reports. One seed; fixed data/seeds do not
promise bit-identical retraining. No generated-source execution, native conversion,
model selection or promotion. Results pending.
