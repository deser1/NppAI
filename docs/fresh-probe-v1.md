# Fresh evaluation probe v1

The prior 28 test records have been repeatedly analyzed. This evaluation-only
probe was created after the fixed continuation checkpoints from run 38618406741
were trained; it is not added to training and contains no `train.json` split.
No model has yet been evaluated on these records in this change.

The frozen [manifest](../evaluation/fresh-probe-v1/manifest.json) records the
[tasks](../evaluation/fresh-probe-v1/tasks.json) SHA-256, seed 20261014, exact
legacy source snapshots and intended model cutoff. Preserve LF bytes on all
platforms; verification rejects changes rather than updating hashes silently.

| Dimension | Scope |
| --- | --- |
| Task families | 12 new labeled families, disjoint from instruction-v1/v2/v3 |
| Task instances | 36 distinct function-name/argument instances, three per family |
| Wording views | 144 correlated views, four per instance |
| Names | 9, 12, 15 bytes; no names shared with any prior train/validation/test split |
| Forms | One known-format control plus three new request/contract/brief forms |
| Budgets | Context 256 and generated response budget 160; all reference sequences fit |

New tasks cover occurrence counting, membership, filtering nonzero values,
sum of squares, first-or-None, mapping lookup/defaults, key/value inversion,
case-sensitive prefixes/suffixes, literal replacement, separator splitting and
pairing unequal-length inputs. Argument aliases vary. These remain small,
project-authored Python tasks; distinct family labels do not prove that every
underlying primitive is unseen. Names, argument vocabulary/lengths, task content
and wording change together, so this does not isolate one cause of failure.

Validation executes only the fixed trusted reference functions against explicit
edge-case examples. It does not run or score NppAI-generated code. Linux and
Windows CI check canonical bytes, generator reproduction, unique case/group
IDs, split/name/prompt isolation, signature preservation and context bounds.

For the next inference measurement, preselect the four-form and eight-form
continued pairs from run 38618406741. Report raw outputs and scores by wording,
family and name length, and pair wording outcomes by task instance. Distinguish
autonomous, staged and any explicitly assisted output. Do not select checkpoints,
decoding policies or thresholds from this probe's scores. After first use this
public set is exposed regression evidence, not an untouched benchmark.

There are 12 authored families and 36 instances, not 144 independent programming
problems. Exact response/AST matching can miss valid alternative implementations
and does not establish functional correctness of model outputs. No production
readiness or model-quality gain is claimed by the dataset/reference CI.

```sh
python scripts/build_fresh_probe.py --check
python -m unittest tests.test_fresh_probe -v
```

## Frozen model evaluation

`scripts/evaluate_fresh_probe.py` evaluates both preselected policies from training run
38618406741. The workflow downloads their final evaluation artifacts and the fixed
initial pair from 38582420228. Before inference it verifies checkpoint hashes, old
training manifests, continuation initialization, 10k additional/20k total update
provenance, reload checks and preserved initial copies. Dense architecture checks
reject incompatible weights. There is no training or checkpoint selection.

Each of the 144 views receives greedy FP32 learned-header generation, staged body
generation only for accepted raw headers, and free-body generation as a control.
Each call uses context256 and up to160 new byte tokens. No reference correction or
fallback is applied. Model outputs are parsed, never executed. Metrics distinguish
requested function name from the full name/argument signature and from whole
reference AST/exact response matching. These are structural/reference metrics,
not semantic correctness or real repository performance.

Artifacts retain all raw outputs, sources, report hashes, and an audit that
recomputes case/group metrics and rejects changed metadata or missing/duplicate
cases. Summaries group by wording, family and name length; wording changes are
paired within the same36 task instances. An all-zero quality result remains valid
evidence and does not fail the infrastructure workflow. CI success is not model
quality success. After this first measurement the public probe is exposed; further
runs are regression measurements, not a fresh holdout. The default model is unchanged.

## First measured results (2026-10-14)

Measured workflow [38629228493](https://github.com/deser1/NppAI/actions/runs/38629228493), evaluator commit `6ea1aa786867bb177518895fabde3a0c3f0f8faf`; Python3.12.15, torch2.14.1+cpu, two CPU threads. All13 tests passed in both jobs, including real full-weight initialization. Source verification and raw-output audits passed. Both quality flags were false; no checkpoint promotion.

Counts below are out of144 correlated views of36 task instances, not144 independent problems.

| Metric | Four-form continuation | Eight-form continuation |
| --- | ---: | ---: |
| Accepted headers | 82 | 115 |
| Exact reference headers | 4 | 6 |
| Staged syntax with a function | 40 | 67 |
| Staged requested name | 9 | 11 |
| Staged requested name and arguments | 2 | 3 |
| Staged full reference AST | 0 | 0 |
| Staged exact response | 0 | 0 |
| Free-body full reference AST / exact | 0 / 0 | 0 / 0 |

| Wording (36 views each) | Four: syntax / signature / AST / exact | Eight: syntax / signature / AST / exact |
| --- | --- | --- |
| known_format | 11 / 2 / 0 / 0 | 15 / 2 / 0 / 0 |
| request | 12 / 0 / 0 / 0 | 14 / 0 / 0 / 0 |
| contract | 12 / 0 / 0 / 0 | 22 / 0 / 0 / 0 |
| brief | 5 / 0 / 0 / 0 | 16 / 1 / 0 / 0 |

All12 family groups had zero AST and exact matches in both staged and free-body arms. All requested-name and signature matches occurred for nine-byte names; 12/15-byte names had none. This is an association in this small probe: name length also varies with argument aliases and instance, so it does not isolate a length cause. Paired wording outcomes and all family/name groups are retained in the artifacts.

Eight-form continuation produces more parsable functions here, but this is not evidence of correct new-task code. Both pairs fail full reference matching even in the known prompt format. Alternative correct implementations could differ from these references; generated code was not executed, so no functional-accuracy claim is made. The probe is now exposed; future reuse is regression evidence. The default model remains unchanged.

| Policy | Artifact | Archive SHA-256 | Report SHA-256 |
| --- | --- | --- | --- |
| four_forms | [11925551788](https://github.com/deser1/NppAI/actions/runs/38629228493/artifacts/11925551788) | `fe0ea3a828d06e2a7bdb72b96396e218496ea3aff55253646d77bb34abe5c78e` | `09926c61f18dc5f4ae6f97e86d9416ac422bed44a022c50a9ccb457eb5de0ce8` |
| eight_forms | [11925746666](https://github.com/deser1/NppAI/actions/runs/38629228493/artifacts/11925746666) | `912cd8e1ae85c464b1af234520fb6a47b4eed11a8ab2ea2f8388196577467bd3` | `cdefa9eeefd6209401a656ca45d2154b8427d47607376c2ce205abc209dc48e3` |
