# Frozen canonical argument binding diagnostic

#171 improved assisted body generation for renamed functions but obtained zero
new-argument reference matches. This inference-only diagnostic asks whether a
frozen body produces better complete references when arbitrary argument names
are normalized to familiar training spellings and then restored.

The single frozen control is #171's final fixed-argument26k checkpoint, from
run38697039023 artifact`nppai-body-arguments-fixed_arguments`. Pin its full raw
report SHA256`e11680be1dc9da3ff056339cbe893ee544d897f8061d4c162dae065e05108a28`,
verify the checkpoint and preserved20k/10k source hashes, and re-audit its training
budgets, full references and raw before/after evidence. This is a controlled body
checkpoint, not a promoted autonomous model. No model retraining occurs.

Train-only inspection finds single-argument names`x`(480),`values`(384),`text`(384)
and two-argument`a,b`(288). Three **globally fixed** normalization policies use
respectively`x`, `values` or`text` for a single positional argument; all use`a,b`
for two arguments. Other arities are passed through unchanged, including the
75 three-argument test views. Selection uses only prompt-parsed arity/order,
never family labels, reference code or per-case best results. The three policies
are reported separately; there is no best-of-three/oracle aggregate.

For each case, parse the explicit task signature with the existing bounded
parser. Simultaneously substitute whole argument-name words in the prompt, infer
with a canonical signature prefix, and restore original argument spellings.
Restoration is limited to variable/parameter token spans identified by the AST
of a single function containing one return statement. It preserves literals,
attribute names, keyword names, operation tokens and formatting. Reject syntax
errors, changed signatures, nested binding/comprehensions, unsupported structure
and capture of a pre-existing free name by a restored argument. Rejected outputs
score as empty and remain in the denominator; raw normalized text and rejection
status are retained. Neither syntax nor unbound-name/operation errors are repaired.

Evaluate all1100 correlated exposed development views from #171:44 original,
528 function-renamed and528 argument-aliased. Reproduce every cached direct
assisted output with the same frozen model; require zero deviations. Identity
normalization must also preserve raw output. Deduplicate identical two-argument
and passthrough prompts across the three policies, without combining their scores.
Each body inference has context256, greedy160 new bytes and two CPU threads.

Retain raw canonical outputs, prompt transforms, restored outputs, rejection
statuses, syntax/name/signature/reference AST/exact scores, per-name-length counts
and paired outcomes relative to direct assistance. Rebuild references/transforms
and re-score all raw evidence in an independent audit. Tests cover simultaneous
swaps, Unicode token offsets, literal/attribute/keyword preservation, scope/capture
rejection and report tampering. Generated code is parsed, never executed.

Signature scaffolding, canonicalization and restoration are deterministic
assistance, not learned argument copying or autonomous code generation. View
counts are correlated development measurements, not independent problem accuracy.
There is no reference repair, fresh-probe access, native conversion, model
selection or promotion. Normalization also changes prompt/prefix bytes and lengths,
so this does not isolate a single causal mechanism inside the model.


## Measured15 October2026

Run[38698946264](https://github.com/deser1/NppAI/actions/runs/38698946264),
implementation`780a78b868be483dc7ea0a0137933824b150a595`:SUCCESS.
All12 tests passed including real PyTorch supervision/initialization checks.
All1100 direct-assisted outputs reproduce the pinned #171 cache with zero
deviations; all raw-output and transform audits pass. CI_build38698946221 passed
all8 jobs. This is frozen inference, not another training run.

Complete assisted reference matches below are **AST/exact**. Each column is one
fixed policy on the entire group, with no oracle aggregate or reference-driven
schema selection.

| Development group | Views | Direct assistance | Single x | Single values | Single text |
|---|---:|---:|---:|---:|---:|
| Validation/original |16|15/15|8/8|7/7|6/6|
| Test/original |28|16/16|8/8|7/7|7/7|
| Validation/function-renamed |192|170/170|84/84|78/78|78/78|
| Test/function-renamed |336|166/165|85/85|77/77|74/73|
| Validation/argument-aliases |192|0/0|84/84|78/78|78/78|
| Test/argument-aliases |336|0/0|85/85|77/77|74/73|

On new-argument test views, single-argument cases contribute50/264(x),42/264(values)
and38/264(text) exact matches. The shared two-argument schema contributes35/36
in each column. Three-argument passthrough cases remain0/36 in each. They are
retained in every denominator; the three policies share two-argument results,
not three independent confirmations.

The gain comes with regressions: on original test views, x retains8/16 previous
exact hits and loses8; values/text retain7/16 and lose9 each, with no gains.
On function-renamed test views, x retains85/165 and loses80; values retain77/165,
lose88; text retains73/165, loses92. None gains a new exact hit in that group.
Thus no global normalizer satisfies retention requirements or replaces direct
assistance as the default. None is promoted, including the policy with the
largest new-argument score.

Across all1100 views, restoration outcomes are:

| Policy | Accepted | Invalid syntax | Rejected shape | Rejected capture |
|---|---:|---:|---:|---:|
| x |824|259|7|10|
| values |997|91|9|3|
| text |928|160|7|5|

Rejected outputs remain failures, not removed cases. The identity/passthrough
raw-output checks also pass. Reporting parser/rejection outcomes does not turn
syntactic acceptance into correctness.

Raw addition example: for`cp_bo9wlp(ymci, xyvvel)`, direct assistance produces
`return a + b` under the requested header, which does not match the reference.
All policies use the same canonical prompt`cp_bo9wlp(a, b)` and generate
`return a + b`; variable-token restoration yields`return ymci + xyvvel`, exactly
the frozen reference. Only identifiers were restored; the operation was generated.

Raw regression: `test-length-13-2` directly generates the correct
`return len(values)`. Under x normalization the prompt says “return the number
of x”, but the model generates`return x * x`. Restoration gives
`return values * values`, still the wrong operation. The transform does not fix it.

This supports a narrow conclusion: deterministic name assistance can recover some
reference bodies on argument-alias views, but operation choice also changes with
argument spelling and failures remain substantial. It does not demonstrate
learned copying or semantic correctness. Prompt length/context changes are also
part of the treatment. Retain the exposed development status and require separate
held-out evidence for any proposed production path.

### Provenance

- Pinned #171 report SHA256:`e11680be1dc9da3ff056339cbe893ee544d897f8061d4c162dae065e05108a28`.
- Diagnostic report SHA256:`6aea17ff21adbb1587ac1da8ce97537dd4976e3e93ff8edb11ed09ac74d05a69`.
- Artifact11955600542, ZIP SHA256:`ac343752011b32240aa938aa020bb47f0ba59051a94662a9354f94fe46ebf06b`.

The archive digest, preserved source report and checkpoint hashes were checked
again after download. The raw1100-case report was independently re-audited against
frozen full references and all prompt transforms. No generated source execution.
