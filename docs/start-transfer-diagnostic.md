# Frozen source transfer and signature termination diagnosis

Status: measured reconstruction of existing PR #198 and PR #202 records.
No new training, inference or generated-code execution.

The classifier first audits the original evidence, then reconstructs 684 case
labels: 228 initialization replies and 228 for each continuation arm. Every
case retains its group/index through a deterministic file, source location,
first output divergence, termination observations and the actual recorded event
at that divergence. Existing raw outputs/events remain in the source evidence.

## Source selection

Absolute source cursors are mapped to UTF-8 prompt character boundaries and
identifier spans. Word-internal positions, requested-name internal positions,
other identifier starts, whitespace and punctuation are distinguished.
UTF-8 continuation bytes receive an explicit label. Repeated requested names
remain multiple valid reference starts; no convenient occurrence is guessed.

| Exposed alias source location | Initial | Source CE | Source CE + margin |
| --- | ---: | ---: | ---: |
| Requested start | 0 | 4 | 4 |
| Inside requested name | 5 | 10 | 8 |
| Inside another identifier | 58 | 50 | 53 |
| Start of another identifier | 13 | 17 | 16 |
| Whitespace | 2 | 1 | 2 |
| Punctuation | 8 | 4 | 3 |
| No copy trigger | 2 | 2 | 2 |
| Total | 88 | 88 | 88 |

Recorded starts inside or at `Define` occur in 33/88 initialization replies
and 28/88 replies in each trained arm. `function` occurs in 18/88, 19/88 and
22/88 respectively. These are selected-token observations, not attention
probabilities or experimentally demonstrated causes of transfer failure.
All 44 development natural-name replies select the requested start in all
three stages, but only 27/44 are exact. The naming contrast is observational;
no new independent samples or causal wording intervention were evaluated.

## Correct starts can still corrupt the signature

Both trained arms correct the same four exposed alias starts, all in test/renamed.
One (`test-smallest-13-2`) becomes exact code. The other three first differ in
the signature: `test-square-13-2`, `test-total-13-2`, `test-largest-13-2`.
At their first divergence, the recorded vocabulary byte is **58 (`:`)**, but
the copy path emits **32 (space)** from the prompt. Exit probabilities are
approximately .1241, .1767 and .2341, all below the .5 exit threshold. The
result continues copying wording such as `in Python to return ...`.

This is a directly observed override of a correct vocabulary byte on that
trajectory, not an oracle repair or a prediction of full-code success after
changing the exit gate. All four corrected starts have a late-exit observation,
including the one exact response: copying past the name can emit a correct
opening parenthesis, so late exit alone is not synonymous with output failure.
The classifier reports first divergence and overlapping observations separately.

## Consequences for the next experiment

Start selection remains a separate bottleneck (82/88 wrong starts and two
missing triggers in each continuation). Boundary-related hypotheses are worth
a controlled readout experiment, but complete candidate logits were not retained:
this report cannot re-rank alternatives or estimate gains from candidate masking.
Signature-boundary exit behavior warrants its own single-factor comparison.
Do not bundle a new source scorer and a new exit rule and attribute gains to one.
Predeclare the next isolated intervention and preserve complete-code, retention
and independent functional gates. Existing Docker results remain 0/6 for both
continuations. No model promotion follows from this diagnosis.

## Reproduction

```sh
python scripts/start_transfer_diagnostic.py --directory docs/evidence/start-transfer
python scripts/start_transfer_diagnostic.py --audit docs/evidence/start-transfer
python -m unittest discover -s tests -p test_start_transfer_diagnostic.py -v
```

The 37 generated JSON members have SHA-256 identities in `manifest.json`.
The audit regenerates every member from independently audited source data and
checks exact bytes, detecting even an invented count with updated local hashes.
Tests cover UTF-8 positions, repeated requested names, identifier interiors,
prompt boundaries and the recorded copy/colon override. These fixtures do not
expand the measured model's Unicode or boundary coverage. Body text mismatch
labels do not establish semantic failures; Docker evidence is separate.
