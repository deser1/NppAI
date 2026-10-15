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
Measured results are recorded below.

## Measured results (2026-10-15)

Measured workflow [38680571234](https://github.com/deser1/NppAI/actions/runs/38680571234), implementation commit `e9537815e1201726b114e06503d00e6fe4e1a63e`. All18 tests passed with the fixed CPU runtime, source checks and independent raw-output audit. The parser supplies1100/1100 reference-equal headers from prompt text only. This is supplied scaffolding, not learned copying; no model was trained or promoted.

Cells show whole-reference AST / exact complete responses.

| Group | Views | Fixed learned header | Varied learned header | Task signature assisted | Free body |
| --- | ---: | --- | --- | --- | --- |
| Original validation | 16 | 14 / 14 | 13 / 13 | 14 / 14 | 0 / 0 |
| Original test | 28 | 14 / 14 | 13 / 13 | 15 / 15 | 0 / 0 |
| Function-renamed validation | 192 | 64 / 63 | 58 / 58 | 84 / 82 | 0 / 0 |
| Function-renamed test | 336 | 64 / 63 | 52 / 52 | 74 / 72 | 0 / 0 |
| Argument-aliased validation | 192 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Argument-aliased test | 336 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

On function-renamed test, assistance retains all63 fixed-control exact hits and gains9, reaching72/336; AST64→74. Relative to the varied control it retains52 and gains20. Original test exact responses improve14→15 relative to fixed and13→15 relative to varied, with no lost complete-reference hits in those comparisons. This measures assistance with a fixed body, not an improvement in the model's autonomous signature copying.

| Function-renamed test name length | Views | Fixed AST / exact | Assisted AST / exact |
| --- | ---: | --- | --- |
| 9 | 112 | 52 / 52 | 52 / 52 |
| 12 | 112 | 8 / 7 | 11 / 10 |
| 15 | 112 | 4 / 4 | 11 / 10 |

For new argument aliases, all336 test headers are supplied correctly but complete AST/exact matching stays0/336; validation is likewise0/192. Only141/336 assisted test outputs parse as functions with the requested signature, because later body text can make the whole response invalid. Prefix correctness is not full-response syntax or body correctness.

### Raw body failure example

Task: `Define cp_bo9wlp(ymci, xyvvel) in Python to return the sum of ymci and xyvvel.`
The assisted prefix is exactly `def cp_bo9wlp(ymci, xyvvel):` but the model emits body `return len(values)`, instead of reference `return ymci + xyvvel`. The Please wording view emits the same body; the Python-prefix view emits `return tex(values)`. These are correlated examples of the same task. Correct declared arguments do not force the model to use those names or choose the requested operation. The raw artifact was downloaded and its archive digest verified. No expression or generated code was executed.

This diagnostic shows that repairing/supplying the header alone does not repair the complete-reference result on this exposed argument-alias set. It does not prove a universal inability, isolate an internal architectural cause, or establish functional accuracy; equivalent non-reference code can fail AST/text matching. The parser also supports only one explicit restricted signature. No new data, body fine-tuning, native replay or fresh-probe measurement was introduced. Keep explicit assistance separate from autonomous-model milestones; the default model remains unchanged.

Artifact: [11948415533](https://github.com/deser1/NppAI/actions/runs/38680571234/artifacts/11948415533), `nppai-argument-signature-assistance`.
Archive SHA-256, independently verified after download: `a932497d2d675ecc1ebb075e0725a4cb9f2b31bcac144792adf96306e700c473`.
Report SHA-256: `062321a9336ccb1d74d0cd5e0c436941593c760c99ebb2d6a992509ace632437`.
