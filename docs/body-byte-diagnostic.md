# Frozen body argument byte diagnostic

#173 obtained zero full new-argument references with either loss policy. This
inference-only diagnostic measures conditional next-byte predictions inside the
reference body, separately from the already observed free/assisted generations.
It does not train or promote a checkpoint.

Pin #173's final run38702282716, full-response report
`bdac050d83b2efd9b3ad1bd12d6b991142120d4e0e2ffdc2994fb78b4bba4296`
and body-only report
`4a9becc992979d49a4e1918e5ad9678a7857943dc50af3821f773084c473a645`.
Re-audit both source reports, model hashes, preserved20k/10k body source hashes,
matched data/schedules and identical raw baseline outputs. Compare three frozen
models: their common source20k, full-response26k and body-only26k.

For every one of the shared1100 correlated exposed development views, supply
the entire preceding **reference** prefix. A causal full-sequence forward pass
provides teacher-forced argmax bytes and reference-byte probabilities for the
body plus EOS. The next-token offset is prompt-byte-count + header-byte-count -1;
all frames fit256 without truncation. Reference AST Name spans identify argument
uses, while UTF-8-aware token spans keep literals, attribute names and keyword
names out of argument-use spans. Header argument declarations are excluded.

Report body argument first bytes and suffix bytes separately, non-argument NAME
tokens excluding Python keywords, OP tokens including punctuation, remaining body
bytes and EOS. The NAME/OP categories describe token spelling, not correct
operation choice. Keep all byte predictions/probabilities, target probability
sums and negative-log-probability sums, argument occurrence counts, completely
correct conditional occurrences, and cases with all conditional argument bytes
correct. NLL uses a probability floor1e-30 for underflow/zero values, so these
sums are bounded diagnostic values rather than unbounded cross-entropy. Suffixes are scored only for occurrences longer than one byte; an empty
suffix must not inflate the result.

At **every argument occurrence** in each model/case, run another forward pass
truncated immediately before its first byte. Require matching first-byte argmax
and numerically consistent logits with the causal full-reference prefill
(atol5e-5,rtol1e-4). Record check count and maximum logit delta. This verifies
alignment/causal prefill consistency; it does not turn teacher forcing into a
rollout. In particular, correct suffixes can be conditioned on a first byte that
the model would never generate itself.

Copy the original cached assisted/free outputs from #173's reports alongside
these conditional metrics, with their existing scores. They are **not fresh
inference calls** and no reference/body prefix is supplied to a free-generation
metric. All previous reference bytes, including callee names/operators and prior
argument uses, are supplied only to the explicitly teacher-forced diagnostic.
Do not infer autonomous copying or causal operation-selection ability from it.

Evidence auditing rebuilds full references, target offsets, categories and spans,
recomputes counts/NLL from raw predictions/probabilities, checks pinned source
cache equality, and rejects missing/duplicate cases and tampering. Tests cover
UTF-8 offsets, literals/attributes/keyword names, EOS shifts, first-byte failure
with apparently perfect gold-prefix suffixes, and real causal model logits.

No training, generated-source execution, canonical restoration, fresh-probe use,
native conversion or promotion. These are descriptive reference-conditional
measurements on exposed synthetic views, not independent task accuracy or a
unique causal explanation.


## Measured15 October2026

Run[38732335293](https://github.com/deser1/NppAI/actions/runs/38732335293),
implementation`9c623edfc6ca6bcebd521103734e35e572d48060`:SUCCESS.
All13 tests passed, including actual causal logits, loss gradients and dense
initialization checks. All1100 raw cases pass the evidence audit.
CI_build38732335201 passed all8 jobs.

There are1450 argument occurrences across the1100 views. Each frozen model
passes1450 truncated-prefix checks:4350 checks total, with identical first-byte
argmax. Maximum absolute logit deltas are1.10e-5(source20k),9.54e-6(full-response),
8.58e-6(body-only). This validates shifted causal prefill consistency for the
specified checkpoints; it is not an autonomous-generation score.

### New-argument test views

All336 test views are kept. Their456 reference argument occurrences contain456
first bytes and1512 suffix bytes. An occurrence can repeat a parameter inside a
body, and the views are correlated; neither denominator is an independent task
count. All three models still have0/336 cached assisted exact replies.

| Frozen model | Conditional first-byte top1 | Conditional suffix-byte top1 | All conditional bytes correct per occurrence | Cases with all conditional argument bytes correct |
|---|---:|---:|---:|---:|
| Source20k |9/456(2.0%)|37/1512(2.4%)|0/456|0/336|
| Full-response26k |26/456(5.7%)|130/1512(8.6%)|0/456|0/336|
| Body-only26k |38/456(8.3%)|141/1512(9.3%)|0/456|0/336|

The difficulty includes both starting an arbitrary argument name and its later
bytes, even with all preceding reference bytes supplied. It is not supported to
attribute the zero matches only to the first letter or an earlier rollout error.
This remains a descriptive conditional observation, not a unique internal cause.

Even complete suffixes are rare:0/456(source),1/456(full-response),0/456(body-only),
while the initial byte is supplied correctly before each scored suffix byte.
The one complete full-response suffix is not a complete argument occurrence.
For the first addition view, target`ymci` gets conditional argmax byte vectors
`[108,114,40,40]`(source),`[108,105,97,106]`(full),`[109,109,118,118]`(body-only).
Each position uses a successively supplied **gold prefix**; these vectors must not
be presented as a name emitted by one greedy rollout.

Validation aliases show the same complete-name failure:

| Frozen model | First bytes | Suffix bytes | Complete conditional occurrences | All-argument cases |
|---|---:|---:|---:|---:|
| Source20k |12/240|27/792|0/240|0/192|
| Full-response26k |24/240|95/792|0/240|0/192|
| Body-only26k |46/240|91/792|0/240|0/192|

### Familiar argument spellings

On function-renamed test views with original argument spellings, conditional
argument predictions are much stronger, but still differ from full generation:

| Frozen model | First bytes | Suffix bytes | Complete conditional occurrences | All-argument cases | Cached assisted exact replies |
|---|---:|---:|---:|---:|---:|
| Source20k |310/456|813/1104|241/456|194/336|72/336|
| Full-response26k |239/456|646/1104|196/456|148/336|65/336|
| Body-only26k |222/456|516/1104|201/456|155/336|84/336|

These compare different questions. Reference-prefix success is not a new complete
reply, and supplying gold operations/previous uses can conceal generation errors.
No new free/assisted generations were run; all displayed reply counts are cached
#173 outputs from the verified reports.

As an additional descriptive check of the stored raw replies, count the longest
matching **byte** prefix of each cached assisted alias reply and its reference,
then label the first differing reference byte using the stored category spans.
Body-only has153 first differences at an argument initial byte,15 at a suffix,
94 at a non-argument NAME token,71 at other body bytes,3 at OP/punctuation.
All336 replies remain failures. These labels describe expected reference
positions, not semantic error causes: a callee generated where an argument was
expected can be labeled an argument-first-byte difference.

The measurement directs further work toward arbitrary-name copying, while also
retaining the operation/rollout failures already documented in #172/#173. It does
not justify checkpoint promotion, automatic reference repair or treating higher
conditional byte accuracy as autonomous improvement. No model weights change.

### Provenance

- Pinned full/body source report hashes are listed above and verified before inference.
- Diagnostic report SHA256:`3b85faf6ed2a635c0f357507f44c819c122d8db4b556ed7bf97e55782941200f`.
- Artifact11970567427 ZIP SHA256:`1e583bfd5fed10e3fee63d76928eea2579d5c80da34d936177438b37505f75b8`.

The ZIP digest, three model hashes and both preserved source report hashes were
checked independently after download. The full raw diagnostic was re-audited
against frozen references, UTF-8 categories/spans, raw predictions/probabilities
and cached generation evidence outside the inference job.
