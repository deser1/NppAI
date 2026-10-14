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
First measured results are recorded below.

## First measured results (2026-10-14)

Measured workflow [38642020413](https://github.com/deser1/NppAI/actions/runs/38642020413), implementation commit `f0adb2efaba0cb1259d69410a84975e628e6e75b`. Both jobs passed all19 tests with PyTorch, source/provenance checks, actual12000/12000 draw counts, every within-run development export/reload comparison and raw evidence audit. Baseline summary counts agree across policies. These tables describe this specific training measurement; repeated training can vary.

Exact header and full requested-signature counts agree in this measurement.

| Development group | Before | Fixed arguments after | Varied arguments after |
| --- | ---: | ---: | ---: |
| Original validation | 14/16 | 16/16 | 15/16 |
| Original test | 21/28 | 22/28 | 17/28 |
| Function-renamed validation | 0/192 | 98/192 | 67/192 |
| Function-renamed test | 0/336 | 155/336 | 87/336 |
| New argument aliases validation | 0/192 | 0/192 | 0/192 |
| New argument aliases test | 0/336 | 0/336 | 0/336 |

| Original test retention | Fixed arguments | Varied arguments |
| --- | ---: | ---: |
| Earlier exact hits retained | 20/21 | 15/21 |
| Earlier hits lost | 1 | 6 |
| Newly correct | 2 | 2 |

Neither policy retains all prior hits. On original validation the fixed policy retains14/14 and gains2; varied retains13/14, loses1 and gains2. The varied policy has zero function-renamed exact headers at15-byte names in both validation/test. No checkpoint is promoted.

Both policies have zero exact/full signatures on unseen development argument aliases, despite requested-name hits on the test of164/336(fixed) and173/336(varied). Valid-header counts are277/336 and315/336 respectively. More parsable headers or correct function names do not establish argument copying. Training on varied argument aliases under this budget does not improve the complete-signature result and reduces earlier retention.

### Raw examples and remaining error

The varied model receives `Define cp_bo9wlp(ymci, xyvvel) in Python to return the sum of ymci and xyvvel.` Its reference is `def cp_bo9wlp(ymci, xyvvel):`, but it emits `def cp_bo9wlp(yuto):`. The name is correct; argument names and arity are wrong. Other wording views for that same task emit `yunt` or `yyxt` instead. These are correlated examples, not independent failures.

For original `test-sort-2-2`, the varied checkpoint emits `def fn_nta64g(virr, rvrrrv):` instead of `def fn_nta64g(values):`. This does not repair #168's argument issue and introduces an arity mismatch. The raw output was inspected directly from the downloaded varied artifact; its archive SHA matches the upload digest. No generated header or expression was executed.

This experiment rejects this preselected alias policy as an improvement under the measured conditions. It does not prove that alias augmentation can never work, that a specific architecture causes the failure, or that longer/multi-seed training would succeed. No tuning or checkpoint selection follows from the exposed scores here. Complete-function bodies, native inference and fresh-probe-v1 were not evaluated; the earlier full-output results from #168 remain separate. Preserve these failed development cases for diagnostics before proposing another learning or explicitly assisted inference method.

| Policy | Artifact | Archive SHA-256 | Report SHA-256 |
| --- | --- | --- | --- |
| fixed_arguments | [11931336747](https://github.com/deser1/NppAI/actions/runs/38642020413/artifacts/11931336747) | `cc2637531679b40cdbb7348ffb783dbf851690f4e6a30044932d9d34d16117c5` | `185a3403d16c673567602a23ce6a66e0882b642797d982b3de54ed2f9704d204` |
| varied_arguments | [11930447425](https://github.com/deser1/NppAI/actions/runs/38642020413/artifacts/11930447425) | `bb95a40fef4d006e0383f904b556e154894698921e8d101574c1608506c6e8bd` | `6511512cf84d3fbf18963f00b04535da0cedf4235474b83637de718397233a20` |

Development SHA-256: `7bd796f994b3972f4983954ed96e73ed6a169a6300c40b207fc1d467901b74fa`.
Varied renamed pool: `b928498057e8288742ffd0eec0e11d10b94fb30d2272135814c6f0b7ca9c131e`.
Fixed renamed pool: `71c7ded47937f1c7c483fa9bc74c8dfefd7d5dd7f79e66a6204973123adb720e`.
