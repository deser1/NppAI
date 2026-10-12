# Reproducible demo and current evidence

Use the [v0.4.0 release](https://github.com/deser1/NppAI/releases/tag/v0.4.0)
package matching Notepad++'s architecture. Extract the `NppAI` directory into
`plugins` and restart Notepad++. A compatible `.nppai` model is required separately.

## Measured generation baseline

| Task | Recorded completion | Result |
| --- | --- | --- |
| Python fixture in `evaluation/code-quality/manifest.json` | Empty | 0/1 syntax-valid completions |

Evidence: [native evaluation run 38177545636](https://github.com/deser1/NppAI/actions/runs/38177545636).
This run predates the diagnostic fix in PR #140. Its green workflow status means
the measurement completed; it does not mean generation passed. The cause of the
empty completion is not yet established. Syntax validity does not prove functional
correctness.

## Repeat the measurement

Run `Native baseline model evaluation` with an explicit successful training run ID
(for example the 300-step experiment). Download `baseline-native-generations` and
inspect `generation-metadata.json`, `recorded-outputs.json`, and `syntax-score.json`.
Keep the model SHA-256, training run ID, engine revision, and diagnostics with any
published result. Generated code is not executed by this workflow.

## Record a Notepad++ demonstration

Once a checkpoint produces useful nonempty output, record installation, model
loading, a selected-code prompt, live output, the proposed diff, and the validation
result. Publish the actual recording with its checkpoint hash and task text.
No GUI recording or useful model-output claim is supplied by the current baseline.

## Package validation scope

`Published release ZIP smoke` downloads v0.4.0 into a fresh Windows runner,
checks all package layouts, and loads the production x64 DLL to validate exports,
name and Unicode contract. It does not validate the real Notepad++ GUI, updates,
ARM64 execution, or model deployment. Those require separate end-user checks.
