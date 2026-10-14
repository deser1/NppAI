# Original-example header replay experiment

PR #166 improved copying of replacement names but lost all21 original test exact
headers and14 original validation exact headers. This development experiment asks
whether a fixed training replay mix can preserve those specific earlier hits while
learning variable-length replacement names. It does not train or evaluate bodies.

Both policies start from the same verified20k-update four-form header checkpoint
in38618406741, not from either regressed #166 checkpoint. Initial sources are checked
against38582420228, original frozen manifests, continuation metadata, preserved
copies and cached source reports. All dense weights initialize the model; AdamW
state resets. There is no fresh-probe-v1 access or checkpoint promotion.

| Setting | No replay control | Half-original replay |
| --- | --- | --- |
| Additional updates / total header updates | 6000 / 26000 | 6000 / 26000 |
| Batch size / total example draws | 4 / 24000 | 4 / 24000 |
| Original examples per batch | 0 | 2 |
| Renamed examples per batch | 4 | 2 |
| Total original / renamed draws | 0 / 24000 | 12000 / 12000 |
| Renamed training pool | same18432 records,1536 names | same18432 records,1536 names |
| Original replay pool | stored for provenance, not sampled | 6144 training-only wording views,512 names |

The renamed pool is exactly the balanced9/12/15-length cp_ pool from #166. Original
examples come only from instruction-v3 train, with four known wording variants;
validation/test records are never replayed. Both policies use seed42, batch4,
fresh AdamW0.001, clip norm1, context256, CPU FP32/two threads. Half-original means
50% of examples, not50% of supervised bytes or loss. Replay deliberately reduces
renamed exposure; record-pool sizes and exposure counts are reported separately.
The report's base dataset metadata describes the renamed pool, while its replay
metadata records the original pool and actual sampling policy/draw counts.

The unchanged development set contains16/28 original validation/test prompts and
192/336 renamed wording views. These are correlated, previously exposed development
tasks; remapped names do not make their task families independent or new. Before
and after training, greedy header generation supplies no reference prefix. Reports
keep valid-header, requested-name, requested-signature and exact-header counts,
grouped by length/wording, with paired before/after cases.

Retention tracks identities: a lost earlier hit is reported even if a different
case becomes correct and the overall count stays unchanged. The evidence audit
rebuilds train/development pools, checks their hashes and fixed replay/draw counts,
recomputes raw-output scores and paired retention. It also reports whether long
names have any exact matches in both development splits; this is a diagnostic,
not an automatic production gate. Negative quality evidence remains a successful
infrastructure measurement. Generated code is never executed.

Every development output is compared between the trained in-memory model and its
exported/reloaded checkpoint. Fixed seeds and data do not promise bit-identical
retraining across runners; #166 measured numerical variation. Artifacts preserve
both training pools, development data, raw before/after reports, audits, checkpoints
and initial-source evidence. No model selection is performed from these scores.

Implementation: `scripts/header_original_replay.py`, regression tests
`tests/test_header_original_replay.py`, workflow `.github/workflows/header-original-replay.yml`.
First measured results are recorded below.

## First measured results (2026-10-14)

Measured workflow [38634434415](https://github.com/deser1/NppAI/actions/runs/38634434415), implementation commit `b090bff66e3a61117347cc469fe5db4aaf9ec779`. Both jobs passed all14 tests with the fixed CPU runtime, source/draw verification, every within-run development export/reload comparison and raw-output audits. The source header SHA-256 is `c68c7908afb65f13ff113e15b6dfa97f73a066da97ec6c1ff40c1a24c16139bd`. Baseline summaries agree across policies. Counts below describe this specific training measurement; later retraining can vary.

| Development group | Before exact header | No replay after | Half-original after |
| --- | ---: | ---: | ---: |
| Original validation | 14/16 | 0/16 | 16/16 |
| Original test | 21/28 | 0/28 | 21/28 |
| Renamed validation | 0/192 | 125/192 | 97/192 |
| Renamed test | 0/336 | 198/336 | 159/336 |

| Original group | Policy | Baseline hits retained | Lost | Newly correct |
| --- | --- | ---: | ---: | ---: |
| Validation | No replay | 0/14 | 14 | 0 |
| Validation | Half-original | 14/14 | 0 | 2 |
| Test | No replay | 0/21 | 21 | 0 |
| Test | Half-original | 19/21 | 2 | 2 |

The half-original test total remains21/28, but two earlier correct cases are lost and two formerly incorrect cases become correct. Aggregate equality is not full retention. Validation retains all14 baseline hits and gains2. The no-replay policy loses all baseline hits in both splits. Both `all_baseline_exact_retained` diagnostics are false; neither checkpoint is promoted. Both have long-name exact matches in each development split, but that partial diagnostic does not override regression or prove complete-code quality.

| Renamed test length | Views | No replay exact/signature | Half-original exact/signature |
| --- | ---: | ---: | ---: |
| 9 | 112 | 83 | 87 |
| 12 | 112 | 73 | 52 |
| 15 | 112 | 42 | 20 |

| Renamed validation length | Views | No replay exact/signature | Half-original exact/signature |
| --- | ---: | ---: | ---: |
| 9 | 64 | 61 | 62 |
| 12 | 64 | 37 | 23 |
| 15 | 64 | 27 | 12 |

On renamed test the half-original arm has313/336 valid headers,188/336 requested-name hits and159/336 exact/full signatures. Control counts are306/336,232/336 and198/336. Validity/name-only metrics do not substitute for correct signatures. Actual sampled draws are12000 original plus12000 renamed in replay, versus0/24000 in control; reduced renamed exposure is part of this comparison.

This measurement supports using original training replay to reduce the observed namespace regression while retaining some variable-length copying. It also shows a tradeoff in renamed development accuracy and two remaining lost baseline test cases. It does not isolate a general causal mechanism or establish statistically reliable gains across seeds. No ratio was selected or tuned after looking at these scores, and no new fresh-probe/body/native measurement was made. Investigate the lost baseline cases using saved raw evidence before another preselected development experiment.

| Policy | Artifact | Archive SHA-256 | Report SHA-256 |
| --- | --- | --- | --- |
| no_replay | [11926764460](https://github.com/deser1/NppAI/actions/runs/38634434415/artifacts/11926764460) | `5a4f30d3f829eab4af1834f19866f2630f95c4f99b8b2e9b4e168dc46a6f0b8e` | `1e3e486f8fb804fabec9702e4152b84b43c3ff393c025f58651defb9de908673` |
| half_original | [11926414603](https://github.com/deser1/NppAI/actions/runs/38634434415/artifacts/11926414603) | `677c9d27721b9bd186a31fc95cdacf1d4bf37826afff780d2ca0fe14ba175828` | `b4ae689777c5c674fbf86e2719d22720d1b781c3be8470a61acead903403c1ba` |

Original pool SHA-256: `401e79702ad33dc867954e906e351296f2a73e53fb8130ac92d1ebe43c396bec`.
Renamed pool: `71c7ded47937f1c7c483fa9bc74c8dfefd7d5dd7f79e66a6204973123adb720e`.
Development: `2170a0e857629bb3f8f1caac3e6920750333ec72e577fa18c58f62478a039d23`.
