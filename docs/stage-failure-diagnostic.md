# Frozen stage-failure diagnosis

This is a **training-only, read-only diagnostic**, not another training run or
proof of improved held-out code generation. It uses the unchanged direct and
staged-mixture checkpoints from the [stage-protocol pilot](staged-protocol-training.md).
It never executes generated source, updates parameters or selects a new default.

## Fixed inputs and measurements

Two original source archives are pinned by full SHA-256:

- Run `39162359982`, artifact `12149182157`: the two measured checkpoints used
  for all interventions. Their hashes are fixed in the runner and raw report.
- Run `39162685717`, artifact `12149930502`: retained scalar training traces and
  metadata for comparing the two historical allocations; these weights are not
  substituted into the generation experiment.

Use **all 96 existing training records**, including all wording variants, and
no new test references or new target annotations. They are correlated examples
already optimized by the pilot, not 96 independent or unseen tasks. Every model
uses its original 1024-byte context, 128-byte output ceiling, greedy FP32 decoding,
RMSNorm epsilon `1e-5` and two CPU threads. There is no assistant header prefix,
renaming, repair or hidden reference correction.

Compare four modes, recording all requests and raw responses:

| Mode | Inserted authored artifacts | Generated content |
| --- | --- | --- |
| Direct | None | Code from original task |
| Autonomous | None | Specification, plan, code |
| Gold specification | Training specification | Plan, then code |
| Gold specification and plan | Both training artifacts | Code |

The last two modes are **assisted interventions**. They bypass inserted stages;
all other stages retain normal empty/EOS/budget acceptance. Direct raw code is
scored even on length termination, matching the original control. Different call
counts/input contexts do not provide a compute-matched causal accuracy comparison.
Completing a stage says nothing about the meaning of its output.

For each training target, also measure next-byte predictions while supplying the
**correct target prefix** and, for plan/code, the correct prior artifacts. Retain
target/predicted byte IDs, target log probabilities and EOS. These conditional
readouts are explicitly separate from autonomous generation. Offline audits
recompute all scores, denominators, request provenance, conditional metrics and
first-error regions from raw records.

## Measured local results

Exact full reference responses on all 96 training records:

| Inference mode | Direct-trained model | Staged-mixture model |
| --- | ---: | ---: |
| Direct task → code | 96/96 | 88/96 |
| Autonomous stages | 0/96 | 2/96 |
| Gold specification → generated plan/code | 0/96 | 12/96 |
| Gold specification and plan → code | 0/96 | 16/96 |

The staged-mixture model's conditional target-byte results are:

| Target, with correct prefixes | Correct next bytes, including EOS | Entire conditional sequence exact |
| --- | ---: | ---: |
| Direct code | 4318/4326 | 88/96 |
| Specification | 6328/6882 | 12/96 |
| Plan | 1600/1662 | 56/96 |
| Staged code | 4017/4326 | 16/96 |

Among the **80 failed staged-code conditional sequences**, the first wrong byte
is inside the function name in **50**, later in the signature in **13**, in the
body in **15**, and in the `def ` prefix in **2**. Sixteen sequences have no wrong
byte. This is a gold-prefix readout, not the location of every error made during
free generation. All 16 retained training-probe direct and autonomous replies
replay exactly for each checkpoint in this local measurement.

Two conclusions are supported by these observations:

1. Bad generated specifications/plans contribute to failure: replacing them with
   authored training artifacts increases exact replies from 2 to 12 to 16.
2. Error propagation is not the only issue. Even correct prior artifacts leave
   80/96 code targets failing, while the same mixture checkpoint gives 88/96
   direct answers. It has not adequately fitted the staged response protocol.

The results do **not** identify a unique cause of the final-stage gap. Stage
instructions, input length/positions, artifact content and mixed training exposure
all differ from direct generation. They do not demonstrate general reasoning or
new-task improvement. Original pilot functional scores remain **0/12**; no new
functional execution or checkpoint promotion is performed here.

## Historical training divergence

The two source allocations have identical expanded initial-state hashes and
identical base-task schedules/stage assignments. Their first saved scalar-loss
differences occur at update **1** for the direct arm and update **2** for the
mixture. Final checkpoint bytes differ in both arms. This bounds the observable
training divergence; scalar logs do not identify the first parameter divergence
or a causal numerical operator because per-update tensors were not retained.
Same-host agreement from the pilot must not be generalized to every CPU/runner.

## Reproduce and next bounded experiment

```sh
python -m unittest discover -s tests -p 'test_stage_failure*.py' -v
python scripts/stage_failure_diagnostic.py \
  --archive original-pilot.zip --comparison original-second-allocation.zip \
  --output measured
python scripts/audit_stage_failure.py measured
```

The dedicated workflow downloads only the two pinned archives and retains its
raw records as `nppai-stage-failure-diagnostic`. Archive loss is an infrastructure
failure; it does not select a different model. Parameters and checkpoint bytes
are checked unchanged after measurement.

Next, isolate **staged-code fitting on training data with authored prior artifacts**
under a fixed budget and a direct-response retention control. First require exact
training-stage reconstruction and retention; then measure a completely autonomous
pipeline on frozen development tasks. Continue to keep assisted results separate.
Do not increase the training budget or tune prompts against the new-test results.
A separate numerical first-step investigation would need saved tensors/weights;
these scalar traces alone do not justify a portability fix.

Raw local requests, responses, conditional bytes, scalar source traces and both
repeats are retained in [failure-records.zip](evidence/staged-protocol/failure-records.zip),
with [source/member hashes](evidence/staged-protocol/failure-source.json). Both
complete same-host diagnostics matched after excluding elapsed seconds. Extract
the archive and audit `PATH/measured` and `PATH/repeat` separately. This observation
does not certify cross-runner equality.
