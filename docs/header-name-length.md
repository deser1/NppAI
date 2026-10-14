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
Measured results are pending the experiment workflow.
