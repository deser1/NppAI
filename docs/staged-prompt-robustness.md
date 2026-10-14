# Prompt robustness of frozen two-stage generation

The original staged pipeline matched 11/28 complete test references and 10/16
validation references. Before changing production behavior, replay the same
two frozen checkpoints from source run 38325353585 under four preselected forms:

| Variant | Change to the original task |
| --- | --- |
| original | None |
| please_prefix | Add `Please ` before it |
| python_prefix | Add `Write Python code. ` before it |
| code_only_suffix | Append ` Return only Python code.` |

Source IDs, requested names, task text and complete reference answers remain
unchanged. These are paired wording perturbations of existing exposed synthetic
tasks, not fresh independent programming tasks. Prefix/suffix changes alter
formatting and byte positions; this experiment cannot isolate either effect.
No weights, decoding settings or acceptance decisions are tuned from results.

For each form, measure all 16 validation and 28 test records. Keep the original
learned-header acceptance policy: no reference-name correction, replacement
header or fallback. Report accepted/exact headers, complete-source syntax,
requested function, full reference AST and exact response separately for staged
and single-body-model generation. Generated source is parsed, never executed.

Pair exact outcomes by source ID and report retained, lost, gained and neither
successes relative to the original prompt. Do not pool four correlated forms
into a claim of 112 independent test tasks. Keep raw prompts, headers, references,
rejection decisions and both outputs plus a derived-group hash and copies of
the verified source checkpoints/reports. Default replay without
`--prompt-variants` is unchanged. Native INT8, general correctness and model
promotion are outside this measurement.

```sh
python -m unittest tests.test_prompt_robustness tests.test_staged_generation
python scripts/evaluate_staged_generation.py --headers header-source \
  --bodies body-source --dataset evaluation/instruction-v3 \
  --output robustness-experiment --source-run 38325353585 --prompt-variants
```
