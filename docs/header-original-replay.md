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
Measured results are pending Actions.
