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

Measured run [38807119553](https://github.com/deser1/NppAI/actions/runs/38807119553)
completed on 2026-10-15. All six tests passed with actual PyTorch inference,
including rejection of future-dependent logits. All **5580 per-byte causal-prefix
checks** passed. The artifact ZIP digest was verified and raw evidence audited
again locally; the preserved chunks reconstruct the original report byte-for-byte.

For all three models, both the complete conditional-name count and scaffolded
exact-name count are **0/132 development views**, including original, `ev_` and
matched `nc_` views. The result is not explained solely by syntax/body failure
in a complete reply or by the unseen `ev_` namespace.

Conditional test-byte counts below supply each correct preceding name byte.
Namespace means the first three bytes; suffix excludes those bytes.

| Test condition | Model | Namespace correct | Suffix correct |
|---|---|---:|---:|
| `ev_` | Initial | 0/84 | 43/224 |
| `ev_` | Fixed | 3/84 | 68/224 |
| `ev_` | Resampled | 2/84 | 67/224 |
| Matched `nc_` | Initial | 0/84 | 47/224 |
| Matched `nc_` | Fixed | 83/84 | 75/224 |
| Matched `nc_` | Resampled | 84/84 | 82/224 |

The trained models predict the familiar namespace substantially more often under
this matched-prefix condition, while most suffix bytes are still wrong. Neither
trained arm produces a complete correct development name even with gold preceding
name bytes. Thus namespace familiarity is one observed sensitivity; changing the
prefix alone does not resolve complete-name failure. This does not identify an
internal attention defect or prove a unique causal mechanism.

| Training-draw group | Views | Initial scaffolded names | Fixed | Resampled |
|---|---:|---:|---:|---:|
| Original | 16 | 3 | 3 | 5 |
| Fixed-alias | 16 | 0 | 0 | 0 |
| Resampled-alias | 16 | 0 | 0 | 0 |

The complete conditional-name counts are the same in these training groups.
These are the first 16 draws per pool, potentially correlated/repeated views,
not exhaustive training-set accuracy. The few original-training hits do not
generalize to the exposed development names. Cached complete replies from #183
remain unchanged and continue to have zero requested-name/signature matches.

The [evidence manifest](evidence/name-byte-diagnostic/manifest.json) retains
summaries, checkpoint/report identities and 24 raw case chunks with individual
SHA-256 values. Reassembling its report metadata and ordered group chunks with
the shared `encoded` serializer reproduces the original report SHA-256
`a0ea3c09fbe71f061b530bf148f6dca74bc6da3afc6166f6ad4df8459302a1c2`.
Verified artifact 12006767931 ZIP SHA-256:
`2b4039c1c9ad29048ced8e564c4cd6f427c2e561497194c1b746ca6e77a252a0`.

No model is promoted. Before scaling continuation, design a training-only
identifier-copy objective with explicit source/target byte alignment and compare
unseen suffixes under matched namespaces. Keep assisted copying, full signature,
complete-body and functional evaluations separate; reserve an independent later
holdout instead of tuning on these exposed diagnostics.
