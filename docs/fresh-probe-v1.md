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
