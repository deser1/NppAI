# Training with prompt wording variants

The frozen 6000-step pipeline lost all original exact successes after three
small wording changes. This next fixed experiment trains both components from
scratch in two arms: plain original prompts and four-form augmented prompts.
Each header/body model receives exactly 10000 updates with the same seed42,
context256, dense64/128 two-layer byte architecture, batch4, AdamW0.001 and clip1.
Name loss weight stays1. The plain and augmented pipelines each total20000
updates; their token budgets, per-record exposure and sampled batches differ.

Only frozen training records are expanded, in original record order followed
by original/Please-prefix/Python-prefix/code-only-suffix order. The source has
1536 records and512 names; augmentation has6144 records and the same names.
Keep original source IDs plus unique derivative IDs and a variant label.
Prompts change; target headers/full responses and name splits do not. All
augmented full-code training sequences fit context256 (maximum184 bytes).
Source files/manifests remain unchanged; actual optimization counts and hashes
are reported separately. Default training remains unaugmented.

After optimization, compose each arm's two exported checkpoints with the same
uncorrected-header/rejection policy. Evaluate all16 validation and28 test tasks
under the four trained forms and two forms excluded from augmentation:
`Please write Python code. ` and `Provide a Python implementation: ` prefixes.
No output selects budgets, variant weights, stopping or a checkpoint. The two
novel forms are wording probes of previously exposed synthetic tasks, not a
pristine competence test or independent new programming tasks.

Preserve both arm checkpoints/reports and raw per-form staged/single-model
outputs, full AST/exact-reference metrics, header metrics and paired outcomes.
Verify source hashes, manifests, dimensions and the explicit10000-step budget.
The old6000-step replay remains the default source configuration. Parsing
generated ASTs never executes source. No automatic promotion, nativeINT8 or
universal robustness claim. One seed and correlated templates remain limits.
