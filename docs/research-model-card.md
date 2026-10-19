# NppAI research staged-mixture checkpoint

- Asset: `NppAI-research-staged-mix.nppai`
- SHA-256: `4859ad69e93ff2ca80d8951bca617476d118f8ef127abc4b313b8d2f0671c6fd`
- Architecture: own dense v2, 256-byte vocabulary, 1024-byte context.
- Source: artifact 12149182157, `staged_mix/experimental.nppai`, from
  [measured run 39162359982](https://github.com/deser1/NppAI/actions/runs/39162359982).
- Training: existing authored NppAI staged-protocol training records, 1024 updates;
  direct/specification/plan/code mixture. No new weights are introduced here.
- Purpose: inspect the custom format, reproduce service startup and study failures.
  It is not a recommended production coding model.
- Evaluation: **0/12 functional passes** on the six bilingual new-task records in
  all measured inference arms. Authored reference programs passed all 12 controls.
  These inspected probes must not subsequently be presented as a fresh holdout.
- Training diagnostic: direct 88/96, autonomous staged 2/96, code with authored
  specification/plan 16/96 exact responses. Those assisted training scores do not
  demonstrate generalization.
- Runtime: Python service uses NumPy inference without PyTorch. The native
  inference runtime has separate bounded parity/quantization tests. This asset
  does not certify cross-platform training reproducibility or every native mode.
- Safety/use: do not execute output outside the restricted validator. Passing a
  small authored test suite is not a general correctness guarantee.

The model is supplied as a research companion to an experimental plugin release.
Distribution does not change plugin defaults or imply successful model promotion.
Protocol and retained evidence:
[training](https://github.com/deser1/NppAI/blob/main/docs/staged-protocol-training.md),
[failure diagnosis](https://github.com/deser1/NppAI/blob/main/docs/stage-failure-diagnostic.md),
[latest measured continuation](https://github.com/deser1/NppAI/blob/main/docs/fallible-planner-training.md).
