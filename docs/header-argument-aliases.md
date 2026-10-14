# Header argument alias continuation

#168 diagnosed a remaining header regression: `test-sort-2-2` copies the function
name but changes argument `values` to `x`. This experiment measures whether training
on varied argument names improves copying, while preserving original-example replay.
It does not repair generated headers or train complete bodies.

Both policies start from the original four-form20k-update header checkpoint in
38618406741, not from a checkpoint selected using #167/#168 scores. Sources and
initialization are verified against38582420228, frozen manifests, checkpoint/report
hashes, preserved copies and continuation metadata. All dense weights initialize;
AdamW resets. Every policy has6000 extra updates, batch4, AdamW0.001, clip norm1,
seed42, CPU FP32/two threads, context256 and26000 total header updates.

| Fixed exposure | Both policies |
| --- | --- |
| Original training pool | unchanged6144 views,512 names |
| Renamed training pool | 18432 views,1536 function names, balanced9/12/15-byte names |
| Batch sampling | 2 original + 2 renamed examples |
| Total draws | 12000 original + 12000 renamed |
| Fixed-argument control | unchanged source argument names in the renamed half |
| Varied-argument policy | replacement argument names in the renamed half only |

Aliases are deterministic4/6-letter lowercase identifiers, exclude Python keywords
and builtin names, preserve parameter order/arity, and consistently replace complete
argument tokens in the declared prompt and reference header. There are1824 distinct
training aliases and168 distinct development aliases, with no overlap. Original
replay prompts/headers are identical across policies. Only instruction-v3 train is
used for training; no validation/test examples or fresh-probe-v1 content is replayed.
Alias lengths change byte positions and supervised token counts; this is a matched
data-policy comparison, not an isolated mechanistic explanation of copying errors.

Both policies share1100 development views:44 unchanged original prompts,528
function-renamed views from #167, and528 further argument-aliased versions. These
are correlated views of previously exposed source tasks, not1100 independent or
fresh problems. The additional aliases are fixed before outcomes are measured,
not selected by model scores. All header reference frames fit context; maximum193
bytes including EOS. No ratio or checkpoint selection is performed.

Evaluate autonomous header outputs before/after training, with no supplied signature
prefix, fallback or correction. Keep valid-header, requested-name, full signature
and exact-header metrics separately, with grouped length/wording summaries and
paired original-hit retention. A correct function name does not prove its arguments
were copied. Neither header metric proves complete-code or functional correctness.
Generated source is never executed.

`scripts/header_replay_training.py` provides the fixed training/reload runner for
this experiment. Every development output must agree between trained memory weights
and the exported/reloaded checkpoint. Fixed data and seeds do not guarantee identical
retraining across runners, as previously observed. Artifacts preserve both pools,
all development records, raw before/after cases, hashes, source weights and a
recomputing evidence audit. Negative quality results remain valid infrastructure
evidence. Defaults and production checkpoints are unchanged.

Implementation: `scripts/header_argument_aliases.py`, tests
`tests/test_argument_aliases.py`, workflow `.github/workflows/header-argument-aliases.yml`.
Measured results are pending Actions.
