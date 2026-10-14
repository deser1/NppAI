# Fixed-format identifier echo sanity check

The eight-name code experiment reconstructed 24/24 seen answers but copied
0/28 held-out names. This probe removes code synthesis: the input is exactly
`Copy identifier: fn_abcdef` and the response is exactly `fn_abcdef` plus LF.
It uses each V3 training name once (512 examples) and the same frozen held-out
name sets (16 validation, 28 test). Source code-task prompts/answers are not
optimized; derived echo records and their training hash are disclosed.

The two-layer dense byte model uses dimension 64/hidden 128, context 96, seed 42,
batch four, AdamW 0.001, gradient clip 1 and a preselected budget of 3000 steps.
Prompt targets are ignored; all response bytes, LF and EOS are supervised.
Held-out splits are loaded only after the last update. No test result selects
parameters, stopping or a checkpoint. Compared with prior code experiments,
the task, corpus construction and context change; this is not a paired ablation.

Evaluate autonomous exact responses separately from per-byte teacher forcing.
Preserve raw generated responses, source manifest, derived training hash, model
hash, runtime and losses. The exported FP32 checkpoint is reloaded and every
reported output must equal its pre-export model output. The first16 training
names are a seen-data sanity sample, not a held-out score.

```sh
python -m unittest tests.test_identifier_echo
python scripts/train_identifier_echo.py --dataset evaluation/instruction-v3 \
  --output echo-experiment --steps 3000
```

Every identifier has nine ASCII bytes and appears at a fixed input position.
A successful result therefore demonstrates only this narrow copying task.
It does not demonstrate variable lengths, shifted positions, instruction
understanding or code generation. A failure cannot by itself identify a
capacity or architecture defect. Native INT8 is not measured. No checkpoint
replaces the plugin's default model and no generated code is executed.
