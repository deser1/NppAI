# Alias copying and Docker failure diagnosis — next experiment

Status: **retained-evidence extractor implemented; fresh inference, oracle interventions and Docker reruns pending**.

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
- [ ] Run frozen diagnostics with audited artifacts and Docker failure taxonomy.
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
