# Alias copying and Docker failure diagnosis — next experiment

Status: **diagnostic protocol only; no new inference or training run**.

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
- [ ] Implement read-only trace extractor and deterministic unit tests.
- [ ] Run frozen diagnostics with audited artifacts and Docker failure taxonomy.
- [ ] Choose a single causal intervention from observed first-error counts.
- [ ] Evaluate a paired training experiment without holdout leakage.
