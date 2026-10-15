# Argument signature assistance diagnostic

#169 obtained zero complete signatures on unseen argument aliases, even after
varied-argument training. This inference-only diagnostic separates supplying a
correct declared signature from the frozen model's ability to generate its body.
It is an assisted method, not evidence of autonomous argument copying.

The existing restricted prompt parser reads exactly one explicit Python function
signature with distinct plain positional parameter names. Defaults, annotations,
variadic arguments, keywords and ambiguous/multiple signatures are rejected. Its
input is only the prompt string; references are used later for scoring. The parser
canonicalizes the LF-terminated declaration without body indentation, matching the
prefix format used by learned-header composition. No reference repair is applied.

Compare three prefixes with the same four-form20k body from38618406741:

- Cached fixed-argument26k header from #169 run38642891392.
- Cached varied-argument26k header from that same run.
- Signature extracted directly from the task prompt.

A fresh free-body output is the shared control. Learned header outputs come verbatim
from the pinned reports, not new header inference calls. Thus inference resource
budgets differ, and name/argument correctness supplied by the parser must not be
credited to the learned model. Checkpoint/report hashes, source initialization,
26k budgets, source pools and raw header-report audits are verified before body
inference. The common body continuation is checked against38582420228 and cached
source metadata independently.

The shared development records are #169's44 original prompts,528 function-renamed
views and528 argument-aliased views:1100 correlated views of exposed synthetic
source tasks, not1100 independent problems. Full code references are reconstructed
from frozen instruction-v3, renaming NAME tokens only so strings/identifier
substrings are unchanged. Existing #168 full references are byte-identical. All
reference frames fit context256 (maximum233 bytes including EOS). No training,
ratio tuning, checkpoint selection, fresh-probe access or native conversion occurs.

The body model generates greedily with up to160 new tokens per call, using each
accepted prefix verbatim. Unsupported or overflowing signatures fail the parser;
there is no learned fallback or automatic correction. Outputs are parsed, never
executed. Metrics distinguish syntax, function name, full requested signature,
whole reference AST and exact complete response. Supplied correct arguments do not
prove correct variable use or complete-code semantics. AST/exact matching can miss
semantically equivalent implementations and is not functional accuracy.

Reports preserve all raw prefixes/full outputs, learned source reports/checkpoints,
common body weights, per-length summaries and paired assisted-versus-learned cases.
The evidence audit rebuilds full references and prompt-derived prefixes, verifies
cached headers, recomputes scores and paired/group summaries, and rejects missing,
duplicate or tampered cases. Infrastructure success is separate from quality.
Defaults and production models are unchanged.

Implementation: `scripts/evaluate_argument_assistance.py`, tests
`tests/test_argument_assistance.py`, workflow `.github/workflows/argument-signature-assistance.yml`.
Measured results are pending Actions.
