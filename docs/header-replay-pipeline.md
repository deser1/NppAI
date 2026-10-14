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
Complete-body results are recorded below.

## Measured complete-output results (2026-10-14)

Measured workflow [38638747173](https://github.com/deser1/NppAI/actions/runs/38638747173), implementation commit `b7ea8af15e4509e600eeec83d2680d0711efacc0`. All19 tests passed with the fixed CPU runtime. Source checks and independent evidence audit passed; newly generated headers have zero deviations from the pinned cached headers in every arm/group. No training or checkpoint promotion occurred.

Cells show whole-reference AST matches / exact complete responses. Name/header accuracy is not substituted for these values.

| Group | Views | Pretrained header | No-replay header | Half-original header | Free body |
| --- | ---: | --- | --- | --- | --- |
| Original validation | 16 | 12 / 12 | 0 / 0 | 14 / 14 | 0 / 0 |
| Original test | 28 | 13 / 13 | 0 / 0 | 15 / 15 | 0 / 0 |
| Renamed validation | 192 | 0 / 0 | 69 / 67 | 67 / 65 | 0 / 0 |
| Renamed test | 336 | 0 / 0 | 64 / 64 | 60 / 59 | 0 / 0 |

Half-original retains all13 earlier exact/AST original test hits and gains2; original validation retains all12 and gains2. The test improvements are `test-total-13-2` (fn_axd8dh→requested fn_axe8dh) and `test-even-13-2` (fn_e9e9d1→requested fn_e9evd1); the frozen body already emitted the reference expressions in both prior outputs. This is evidence of improved identifier handling in these two exposed cases, not general new-body learning.

| Renamed test name length | Views | No replay AST / exact | Half-original AST / exact |
| --- | ---: | --- | --- |
| 9 | 112 | 50 / 50 | 51 / 51 |
| 12 | 112 | 7 / 7 | 6 / 5 |
| 15 | 112 | 7 / 7 | 3 / 3 |

The half-original renamed test has155 exact reference headers but only59 exact complete outputs (60 whole-reference AST matches). Correct headers alone therefore overstate complete-reference success. Most complete hits are nine-byte names; the small12/15-byte groups remain weak. No-replay has more renamed full matches here but zero original full matches, preserving the previously observed namespace tradeoff. Single-model full matching is zero in all groups.

### The remaining argument regression

For `test-sort-2-2`, the reference is `def fn_nta64g(values):` followed by `return sorted(values)`. Pretrained generation has the correct header but body `return s(vab)`. Half-original generation changes the header argument to `x` and emits `return x * s`. Both are syntax-parsable functions and both fail whole-reference AST/exact matching. Half-original also fails the requested signature. The header regression is real, but this case was not one of the13 previous complete-output matches. This explains why the complete-output retained-hit count stays13 while header/signature retention still has a regression. No generated expression was executed and no functional-accuracy conclusion is drawn.

This is a useful limited development result: replay improves original complete-reference matching while enabling some remapped-name responses with a frozen body. It does not establish unseen-wording robustness, fresh-probe results, native equivalence, functional correctness or a production-ready coding model. Alternative semantically correct code can differ from reference AST/text. The remaining argument issue and poor long-name complete responses require separate development evidence; the default model remains unchanged.

Artifact: [11929132998](https://github.com/deser1/NppAI/actions/runs/38638747173/artifacts/11929132998), `nppai-header-replay-pipeline`.
Archive SHA-256 (also verified after download): `4b04dc5c4399e2898c7a0ad6eb86a84db8fd4699d53d4b1bc4bfd0928cc7dc2f`.
Report SHA-256: `0d0d4fbdd7488ed2b021f770ac29ef90ea57b7e8a437d0816e10acf78bbb147b`.
