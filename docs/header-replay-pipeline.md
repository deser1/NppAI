# Frozen header replay and complete-body diagnostic

The final #167 half-original checkpoint loses one previous exact header case:

| Case | Expected / earlier output | Replay output |
| --- | --- | --- |
| `test-sort-2-2` | `def fn_nta64g(values):` | `def fn_nta64g(x):` |

The name is copied correctly; the argument is changed. This evidence classifies
argument drift, not a failure to copy the name. It does not establish why the
model changed the argument. Raw LF-terminated headers and prompt are retained in
the diagnostic report. The final half-original artifact11927663333 from run
38635038956 was inspected and its archive SHA-256 verified as
`49a00e622e1f4cc2d4322589cf29fc98d6b981de2b5edfb31866ac71ce9d0412`.

This inference-only experiment evaluates the entire staged response, to avoid
treating a correct header as proof of correct complete code. It uses three frozen
header arms with the same four-form20k-update body checkpoint from38618406741:

- Pretrained20k header, before #167 continuation.
- No-replay26k header from38635038956.
- Half-original26k header from38635038956.

The free-body output is a shared single-model control. There is no training, ratio
tuning, fresh-probe access, native conversion or production-model change. All
headers are generated greedily; only valid/context-fitting raw headers are used
verbatim as body prefixes. There is no signature repair, reference fallback or
argument substitution. Each generation call uses context256 and up to160 new
bytes. All full reference sequences fit context (maximum209 bytes including EOS).
Generated source is parsed, never executed.

Both source replay-report hashes are pinned, then audited against rebuilt training
and development data. Checkpoint hashes, full-weight initialization,26k budgets,
source copies and common body provenance are verified before inference. The
existing body initialization and continuation metadata are checked independently.
Newly generated headers are compared to their archived before/after outputs;
mismatches are counted explicitly and are not corrected or silently treated as
identical inference. All checkpoints must have the compatible dense byte64/128,
two-layer,context256 architecture.

Full references come only from frozen instruction-v3 validation/test, with the
same name substitutions and wording views as #167. There are44 original prompts
and528 correlated renamed views, not572 independent tasks. These are exposed
synthetic development tasks, not a fresh benchmark. Argument names and expression
bodies are preserved in references. Metrics distinguish syntax, requested name,
full requested signature, whole reference AST and exact response; AST/exact
matching alone is not functional correctness.

Reports retain all raw headers/body outputs, archived lost-header diagnoses,
per-length summaries, paired complete-output changes relative to the pretrained
header, source reports/checkpoints and an independent evidence audit. The audit
rebuilds complete references, recomputes metrics and rejects reference, prefix,
cache-match or aggregate tampering, missing/duplicate cases and changed regression
diagnoses. Infrastructure success is separate from model-quality success.

Implementation: `scripts/diagnose_header_replay.py`, tests
`tests/test_replay_pipeline.py`, workflow `.github/workflows/header-replay-pipeline.yml`.
Complete-body measured results are pending Actions.
