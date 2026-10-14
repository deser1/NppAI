# Training with variable whitespace positions

Four fixed wording forms repaired some regression tasks but both novel wording
prefixes remained 0/28. Compare fixed-form augmentation with the same augmented
corpus plus independent whitespace offsets sampled per training view. This is
one preselected experiment, not a proof that positional embeddings caused the
earlier failures.

Both arms train separate header and body models from scratch for 10000 updates
each, with seed42, context256, dense64/128 two layers, batch4, AdamW0.001, clip1
and name weight1. Both use the same6144 base augmented records and frozen name
splits. The shifted arm prepends0–32 spaces and appends0–16 LF characters with
a separate RNG seeded4201; it preserves task text, names and target responses.
The source batch RNG/order is unchanged. Sequences change length and token
position, so equal updates do not imply equal token budgets or compute.

Maximum full augmented training length184 plus maximum whitespace48 is232,
within context256. Oversized samples fail rather than truncate. Record source
hashes and augmentation metadata plus jitter seed/ranges, view count and the
SHA-256 of ordered (source ID, actual prompt, target response) views. Each model
has40000 sampled views; reproducibility uses frozen records and both seeds.
Default training has no offsets.

After optimization, replay each arm's two exported checkpoints with the same
uncorrected-header/rejection policy on all16 validation and28 test tasks under
four known forms and the two novel wording prefixes. Those novel words are not
inserted into training. Raw header/full-code metrics, paired outcomes and source
checkpoints remain in artifacts. Previously exposed synthetic tasks and one
seed are limits; word changes also change content, so the result cannot isolate
position from instruction understanding. No test-guided tuning, checkpoint
promotion, nativeINT8 or generated-source execution.
