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
selection or promotion. Results pending.
