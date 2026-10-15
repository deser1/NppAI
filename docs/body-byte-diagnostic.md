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
uses, while UTF-8-aware token spans preserve literals, attributes and keyword
names as separate categories. Header argument declarations are excluded.

Report body argument first bytes and suffix bytes separately, non-argument NAME
tokens excluding Python keywords, OP tokens including punctuation, remaining body
bytes and EOS. The NAME/OP categories describe token spelling, not correct
operation choice. Keep all byte predictions/probabilities, target probability
sums and negative-log-probability sums, argument occurrence counts, completely
correct conditional occurrences, and cases with all conditional argument bytes
correct. Suffixes are scored only for occurrences longer than one byte; an empty
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
unique causal explanation. Results pending.
