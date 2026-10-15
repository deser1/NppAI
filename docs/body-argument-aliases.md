# Full-response argument alias continuation

After #170 supplied correct headers but obtained0/336 complete reference matches
on new argument aliases, this experiment tests whether training full responses
with renamed arguments changes body generation. It is a preselected two-policy
comparison, not a checkpoint promotion or independent benchmark.

Both policies initialize every dense weight from the same four-form20k **body**
checkpoint in run38618406741. Its10k origin38582420228, preserved source pair,
model/report hashes, dataset manifest and cached continuation report are verified.
The historically named `initialize_header_weights` helper is a generic dense
all-weight loader; it does not replace a body with a header checkpoint.

Each policy uses6000 additional updates, batch4, seed42, AdamW0.001 with fresh
optimizer, gradient clip1, context256 and two CPU threads. Each batch contains
two original training examples and two function-renamed training examples. The
fixed policy keeps original argument names in the renamed half; the varied policy
uses #169's deterministic4/6-letter aliases. Pools contain6144 and18432 full
responses. No validation/test example is replayed; aliases and function names are
disjoint between training and development. The replay proportion is by examples,
not bytes or loss. Full response bytes including headers, bodies and EOS are
supervised; prompt tokens are masked, matching the source body's training method.

Reference bodies are reconstructed from frozen instruction-v3 by renaming Python
NAME tokens only. Strings and identifier substrings are preserved. Every training
frame fits256 including EOS: maxima184(original),196(fixed renamed),200(varied
renamed). All1100 development views also fit, maximum233. There is no truncation,
length-based filtering or fresh-probe access.

Evaluation before and after training uses the same1100 development views:
44 original,528 function-renamed,528 argument-aliased. They are correlated views
of exposed synthetic task families, not1100 independent tasks. Two arms are scored
separately: autonomous single-model output and complete output with a signature
supplied by the prompt-only parser. Supplied signatures are not learned copying.
The model generates at most160 new bytes per body call in either arm. Raw outputs,
syntax/name/full-signature metrics, reference AST/exact matches, per-name-length
counts and paired retained/lost/gained outcomes are retained. AST/exact matches
measure reference structure, not semantic correctness or execution.

After export, both arms on every development view must reproduce the in-memory
model's outputs. Independent evidence auditing rebuilds full references and pools,
recomputes scores from raw output, rejects missing/duplicate cases and checks
budgets, initialization and paired outcomes. Fixed seeds/data do not guarantee
bit-identical retraining; reload identity is a distinct within-run check.

Workflow artifacts contain model, full datasets/manifest, before/after raw report,
audit and preserved source models/reports. No generated source is executed, no
native model conversion or default checkpoint replacement occurs. Results pending.
