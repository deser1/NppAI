# Alias copying and Docker failure diagnosis — next experiment

Status: **retained extractor and frozen oracle interventions measured, including Docker reruns; no promotion**.

## Baseline and question
PR #198 measured 27/132 exact replies on exposed development natural-name
views and retained 4/4 original exact hits, but obtained 0/88 renamed and
matched-prefix development replies and 0/6 Docker functional probes.
Determine whether failure starts at source-span selection, name-start gating,
cursor advancement, identifier termination, signature generation, or function
body semantics. Do not infer that a valid name implies functional correctness.

## Frozen, causal protocol
1. Pin the PR #198 sequential checkpoint, exact source and suite hashes, Python,
   PyTorch and hardware metadata. Do not retrain or modify weights.
2. For each failed alias case, record the prompt-only source span candidates,
   chosen start index, per-step cursor index and byte, start/exit probabilities,
   vocabulary byte and actual output byte. Include repeated source bytes,
   underscores, digits, UTF-8 and names near context boundaries.
3. Label the **first autonomous divergence** from the requested declaration:
   no declaration; wrong source span; wrong first identifier byte; cursor
   discontinuity; premature/late exit; incorrect signature; body/semantic error.
   Report overlapping downstream errors separately, not as the first cause.
4. Record a frozen counterfactual with an **oracle source start** and separately
   an **oracle exit boundary**, clearly labeled assisted upper bounds. Neither
   may be counted as autonomous success or used to tune on validation/test.
5. For each of six existing Docker probes, preserve the full generated output,
   parse result, requested callable lookup, and first failing functional check.
   Execute only inside the existing restricted Docker validation environment.
6. Reproduce the frozen baseline output hashes before comparisons; retain
   raw records, SHA-256 manifests, and a deterministic audit/reconstruction test.

## Output and gates
Publish counts with denominators per natural, renamed, matched-prefix and
training-family group. Compare 27/132, 0/88, 4/4 and 0/6 as historical
reference points, not new results. Explain every divergence category and
provide representative raw examples without exposing a fresh independent
holdout. A future mechanism PR must predeclare one intervention and matched
control, demonstrate autonomous gains, preserve original hits and separately
pass independent Docker functional evaluation before checkpoint promotion.

## Follow-up work
- [x] Implement read-only retained trace extractor and deterministic unit tests (partial protocol coverage; see limits below).
- [x] Run frozen diagnostics with audited artifacts and Docker failure taxonomy.
- [ ] Choose a single causal intervention from observed first-error counts.
- [ ] Evaluate a paired training experiment without holdout leakage.

## Retained evidence extractor

Run from the repository root:

```sh
python scripts/alias_copy_failure_diagnostic.py --output alias-diagnostic.json
python -m unittest discover -s tests -p test_alias_copy_failure_diagnostic.py -v
```

The extractor verifies the original metadata/chunk/report SHA-256 hashes and
existing pilot audit before joining every retained sequential after-training
reply to the pinned input groups. JSON includes the source manifest and
checkpoint hashes, full output bytes/events, all boundary-matched requested-name
source positions, and the first differing output byte. Source positions are
post-hoc labels, never decoding inputs. Repeated candidates stay ambiguous.

First output divergence and overlapping mechanism observations are separate.
A wrong start can initially emit the right byte; copying beyond a name can emit
a correct opening parenthesis. Neither observation alone proves a causal
functional failure. Cursor discontinuities or corrupted bytes fail the audit.

### Observations from existing PR #198 evidence

| Exposed group | Replies | Wrong source start | Declaration-prefix divergence | Exact replies |
| --- | ---: | ---: | ---: | ---: |
| validation renamed | 16 | 15 | 1 | 0 |
| test renamed | 28 | 27 | 1 | 0 |
| validation matched-prefix | 16 | 16 | 0 | 0 |
| test matched-prefix | 28 | 28 | 0 | 0 |

All 86 wrong-start cases first differ inside the requested name. The other two
replies differ before the name. Across all 132 development replies, the existing
27 exact matches are reconstructed; these are not new model results.

Historical Docker outputs contain three syntax failures (`gap_19`, `below_31`,
`first_43`), one empty reply (`edges_59`), and two missing requested callables
(`lift_7`, `limit_71`, each a recorded case-1 KeyError). Static parsing finds
`ft_7` and `imit_71` in those last outputs. The extractor preserves complete
historical diagnostics and source text; static declarations are not runtime
callable lookup or a new Docker run. Functional success remains 0/6.

### Remaining protocol work

This first implementation uses retained events only. Full per-step vocabulary
readouts, start probabilities, frozen oracle-start/oracle-exit interventions,
and additional UTF-8/context-boundary inference probes require the pinned
checkpoint and a separate run. Unit fixtures cover UTF-8 byte offsets, repeated
bytes/names, underscores/digits, wrong starts, early/late exits and tampering;
they do not establish model performance on those cases. No weights changed,
no generated code executed, no fresh holdout consumed, no promotion.

## Frozen oracle runner

`scripts/alias_oracle_diagnostic.py` accepts only the SHA-256-pinned PR #198
archive and repeated checkpoint. Four fixed modes are evaluated without tuning:
autonomous; oracle start only; oracle exit only; both. A start oracle selects the
unique requested-name prompt byte position. The exit oracle forces copying until
`4 + UTF-8 name length` response bytes, then delegates to the unchanged vocabulary
head. Exit-only does not correct a wrong start. Neither supplies `def ` or rewrites
outputs. A missing/ambiguous requested source span fails rather than being guessed.

The original decoder and model remain unchanged. A diagnostic proxy records
learned start position/probability, vocabulary byte and learned/effective exit
probabilities. A modified start readout is only consumed if the original decoder
reaches its autonomous `def ` trigger; `triggered` and events make this visible.

Before assisted comparisons, every autonomous output byte and cursor event must
match the retained baseline. All four modes repeat exactly. The six exposed
functional probes also repeat, then run through the existing restricted Docker
validator. Complete raw records, source/checkpoint/suite hashes, runtime/hardware,
Docker image ID and a SHA-256 manifest are uploaded. The final `complete` marker
requires unchanged model state; partial reports do not establish success.

Workflow: `.github/workflows/alias-oracle.yml` (pinned CPU runtime, dedicated
experiment branch push or manual dispatch). Local non-Torch contracts run in
normal CI; the dedicated job additionally runs actual decoder intervention tests.
Results measured below. Assisted outcomes diagnose sensitivity and are not guaranteed
mathematical upper bounds, independent holdout accuracy, or model promotion.

## Measured frozen interventions

Run [39055914603](https://github.com/deser1/NppAI/actions/runs/39055914603)
completed on source `c024951430da4ad747088db8dc15f1101170153e`.
All 228 autonomous responses replayed retained bytes/events exactly, each of the
four modes generated twice identically, and model weights remained unchanged.

| Mode | Exact development replies | Exact renamed/matched-prefix | Docker passes |
| --- | ---: | ---: | ---: |
| Autonomous | 27/132 | 0/88 | 0/6 |
| Oracle start | 57/132 | 30/88 | 0/6 |
| Oracle exit | 29/132 | 0/88 | 0/6 |
| Both oracles | 65/132 | 36/88 | 0/6 |

Correcting source selection helps this exposed set much more than exit-only
intervention. This supports prioritizing a single controlled start-selection
training intervention next, but does not prove generalization. Both oracles
still fail 52/88 alias replies and every functional probe. Delegating to the
vocabulary head at the name boundary can append more identifier bytes (for
example `lift_7tlpue7`), so an exit oracle is not an output-name repair.
With start-only, `lift_7` returns `-x`, `below_31` returns `max(values)`, and
`limit_71` returns `abs(x)`; correct names do not resolve body semantics.

`docs/evidence/alias-oracle/raw-records.zip` retains every original JSON member,
including per-case bytes/events/readouts and the original manifest.
`report.json` retains runtime, hashes, all group summaries and full Docker
outputs/diagnostics. `retention.json` pins both original and retained ZIP hashes.
The checkpoint binary is excluded from the retained ZIP; its SHA-256 remains
in the original manifest, and its bytes remain in the original Actions artifact.

```sh
python scripts/audit_alias_oracle.py
```

The read-only auditor verifies ZIP/file hashes, input identities, all 912
response scores, cursor/exit event consistency, baseline reconstruction and
functional totals. It does not rerun Docker or reconstruct model logits from
weights. No independent holdout was exposed and no checkpoint promoted.
