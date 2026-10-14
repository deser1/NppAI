# Matched header name length continuation

PR #165 found only 2/144 versus 3/144 full staged signature matches on a small
fresh probe, all with the shortest names. This experiment examines training on
variable name lengths using development data. It does not use fresh-probe-v1
for training, evaluation, checkpoint selection or task generation.

Both arms initialize all dense FP32 header weights from the four-form continuation
pair in run38618406741 (20k header updates). The workflow also downloads the initial
pair from38582420228, verifies original manifests/checkpoint hashes, continuation
metadata, cached source report, preserved initialization copies and reload evidence.
Only the header model is continued. The body model and default model are unchanged.

| Fixed setting | Value |
| --- | --- |
| Additional optimizer updates | 6000 per arm (26k total header updates) |
| Runtime | CPU FP32, torch2.14.1+cpu, two threads |
| Architecture | byte vocabulary256, dim64, hidden128, two layers, context256 |
| Optimizer | fresh AdamW, learning rate0.001, batch4, clip norm1, seed42 |
| Source | instruction-v3 train only; 1536 records, 512 source names |
| Training views | 3 replacement names × 4 known wording variants per source record =18432 |
| Training names | 1536 unique names per arm |
| Short control | all replacement names length9 |
| Mixed arm | equal record counts for lengths9,12,15 |

Deterministic generated names use `cp_` plus lowercase alphanumeric suffixes. Both
arms change the namespace and lexical identities; the first length9 candidate is
shared. Argument lists, meanings and response header format are preserved. This
is a matched data-policy comparison, not an isolated causal name-length ablation:
lexemes, response byte counts and token budgets differ. There is one seed.

Development uses the old instruction-v3 validation/test source tasks. Names are
remapped independently from training to the same three lengths for both arms, then
expanded into four known wording views. There are192 validation and336 test views,
plus the unchanged16/28 original prompts. The views are correlated. Families and
prompt templates were previously exposed; this is development evidence, not a new
independent benchmark. Remapping names does not make task families new.

Before and after training, the header model generates without supplied name or
signature prefixes. Each call uses greedy decoding and up to160 new bytes. Report
metrics distinguish valid single-line headers, requested names, full name/argument
signatures and exact reference headers. Header success does not establish complete
code or body correctness. No generated source is executed. Exported checkpoint
outputs must match the in-memory model on every development view.

Artifacts preserve the derived train/development data, manifest hashes, raw output
cases before/after, grouped length/wording metrics, paired changes, source checkpoints
and an evidence audit. The audit rebuilds records, recomputes metrics and rejects
changed prompts/references, duplicate/missing cases or inconsistent summaries. A
negative quality result remains valid evidence. Infrastructure CI is separate from
quality; there is no automatic model promotion.

Implementation: `scripts/header_name_length.py`, tests
`tests/test_header_name_length.py`, workflow `.github/workflows/header-name-length.yml`.
Measured results and limits are recorded below.

## Measured results (2026-10-14)

Measured workflow [38631515174](https://github.com/deser1/NppAI/actions/runs/38631515174), evaluator commit `ed165fdb06cd71826097cc72a09669a5dbb86c9a`. Both jobs passed all9 tests with the fixed CPU runtime, source verification, export/reload comparisons and evidence audits. Source header checkpoint SHA-256 `c68c7908afb65f13ff113e15b6dfa97f73a066da97ec6c1ff40c1a24c16139bd`. Baseline summary counts agree across arms.

| Development group | Before: exact header | Short control: after | Mixed lengths: after |
| --- | ---: | ---: | ---: |
| Original validation | 14/16 | 0/16 | 0/16 |
| Renamed validation | 0/192 | 60/192 | 118/192 |
| Original test | 21/28 | 0/28 | 0/28 |
| Renamed test | 0/336 | 81/336 | 206/336 |

Exact header and full signature counts agree in this measurement. Requested-name matches on renamed test are92/336(short) versus241/336(mixed), so correct names alone overstate full signature success. Valid headers are219/336 versus302/336; validity does not establish the requested name or arguments.

| Renamed test name length | Views | Short exact/signature | Mixed exact/signature |
| --- | ---: | ---: | ---: |
| 9 | 112 | 81 | 87 |
| 12 | 112 | 0 | 70 |
| 15 | 112 | 0 | 49 |

| Renamed validation name length | Views | Short exact/signature | Mixed exact/signature |
| --- | ---: | ---: | ---: |
| 9 | 64 | 60 | 58 |
| 12 | 64 | 0 | 39 |
| 15 | 64 | 0 | 21 |

Paired before/after comparison finds206 new exact hits on renamed test in the mixed arm and81 in the short arm, from a zero baseline. Both arms lose all21 original test exact hits and14 original validation exact hits. This is a strong observed regression on the original namespace. The mixed policy improves these development-copying counts but does not preserve prior behavior; neither checkpoint is promoted. Both arms replace all original training names with cp_ names, which changes namespace exposure. The observed loss is consistent with specialization/forgetting, but this experiment does not isolate its mechanism.

There is no new evidence here about complete-function bodies, native INT8 or fresh-probe-v1 quality. Further development should preselect a matched replay mix retaining original training examples, measure original and renamed cases together, and keep the exposed fresh probe separate. No next experiment or parameter tuning was included in this run.

| Arm | Artifact | Archive SHA-256 | Report SHA-256 |
| --- | --- | --- | --- |
| short_control | [11926128967](https://github.com/deser1/NppAI/actions/runs/38631515174/artifacts/11926128967) | `db7c3c139b9d5ae67038eeaebbb260a04ed4696d12362b2c193da2df540821c5` | `b46820751f18a722d89eb285d4a01e9d8bc66e51dda43edee019e4a52b0e83b0` |
| mixed_lengths | [11926256819](https://github.com/deser1/NppAI/actions/runs/38631515174/artifacts/11926256819) | `9fbcf2e8e4cfa06a71221cdb90a16d75faf4cbbd7c0ec2be4620566cbe6a3930` | `8d68a9a38bc95d59f793591de761d70805546b0562746b84c3465fa5587cd769` |

Development dataset SHA-256: `2170a0e857629bb3f8f1caac3e6920750333ec72e577fa18c58f62478a039d23`.
Training SHA-256: short `ddcc0c16a81aa6fa89b9190cd5d992ec0597cc61c69903d42b91787c7651d648`; mixed `71c7ded47937f1c7c483fa9bc74c8dfefd7d5dd7f79e66a6204973123adb720e`.

### Repeat-training variation

The unchanged experiment code was trained again in [38632091438](https://github.com/deser1/NppAI/actions/runs/38632091438), commit `5ee1492a35ab7db2c086e7a316cf8b0f142476bf`. Renamed test exact/signature counts were80/336(short) and198/336(mixed); validation62/192 and125/192. Mixed test per-length counts were83/112,73/112,42/112. Original exact headers remained0/16 and0/28 in both arms. Before-training original test counts remained21/28. Both audits and reload checks passed.

Fixed seed/runtime and reproducible data do not provide bit-identical retraining evidence here. The cause of the numerical variation has not been isolated. The first-run table above describes that specific measurement, not guaranteed repeat scores. Both observed runs support the same limited development conclusion and original-name regression. Export/reload identity is verified within each run; it is not an assertion that separately trained checkpoints are identical. Cross-seed/statistical confidence and reproducible training kernels remain unverified.
