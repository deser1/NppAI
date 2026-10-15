# Frozen function-name byte diagnostic

The matched 1024-update name-continuation experiment obtained zero requested
names/signatures in 88 complete replies. This inference-only diagnostic measures
where the frozen models fail under stronger, explicitly disclosed assistance.
It does not train, modify defaults, select a checkpoint or execute generated code.

The model triple is the pinned initial full-response checkpoint and the fixed /
resampled continuations from [run 38804571444](https://github.com/deser1/NppAI/actions/runs/38804571444).
The initial training report, both continuation reports and all checkpoint digests
must match. Source training reports are independently audited again.

Two separate measurements are made:

1. **Conditional name bytes:** supply the complete task prompt, fixed `def `
   scaffold and every preceding correct reference-name byte. Record each next-byte
   prediction and target probability, namespace (first three bytes) and suffix
   accuracy, full conditional-name hits and negative log probability. This is
   teacher forcing, not free copying. Every byte's full-prefill logits must match
   inference on its individually truncated prefix; an intentionally noncausal
   test model must be rejected.
2. **Scaffolded name decoding:** restart from the prompt plus only `def `, supply
   no reference-name bytes and decode greedily for at most 32 bytes, stopping at
   `(`, newline or EOS. A hit requires exactly the requested name followed by `(`.
   This still supplies a declaration scaffold; it does not measure full signature,
   function body or autonomous complete output.

The original complete free responses from #183 are copied verbatim into the
applicable diagnostic cases, with separate labels. They are cached observations,
not newly generated full replies. Their scores remain zero for requested names;
assisted byte statistics must not replace those results.

The fixed groups contain 180 correlated views, evaluated by each of three models:

| Group | Views | Exposure |
|---|---:|---|
| Original training draws | 16 | First 16 original draws of the fixed schedule |
| Fixed-alias training draws | 16 | First 16 renamed draws actually used by fixed training |
| Resampled-alias training draws | 16 | Matched source draws with names used by resampled training |
| Original validation / test | 16 / 28 | Already exposed development source records |
| `ev_` renamed validation / test | 16 / 28 | Existing renamed development views |
| `nc_` matched-prefix validation / test | 16 / 28 | Same suffix and byte length as `ev_`, training-like namespace |

Training views are selected by draw order, not outcome, and are not a new training
set or independent sample. Both models are evaluated on both alias training groups;
a group seen by one arm need not have been seen under that name by the other.
Matched-prefix development names are checked against the complete alias schedule
of both arms and must not overlap. The namespace comparison changes prompt and
reference bytes together and cannot isolate an internal causal mechanism.

The report retains raw predictions, probabilities, scaffolded bytes, causal-prefix
checks, case identities and cached complete replies. The audit reconstructs every
record and score, rejecting altered caches, case order, model identities or
summary counts. The experiment reads no fresh-probe or local-smoke tasks. Reports
are uploaded by **Frozen function-name byte diagnostic** for 14 days. Checkpoint
availability is likewise bounded by the source artifact retention period.

Implementation: `scripts/diagnose_name_bytes.py`; tests:
`tests/test_name_byte_diagnostic.py`.

## Observed outcome

Pending the frozen inference run. Neither a conditional-byte gain nor a
scaffolded-name hit alone establishes autonomous signature/body correctness.
