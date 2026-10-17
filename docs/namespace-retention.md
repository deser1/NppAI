# Namespace and original-task retention pilot

Status: predeclared 2×2 experiment measured; copying and retention gates fail. No promotion.

The prior copy curriculum had 5/132 exposed development exact hits, 0/6
functional task passes and original-training regression 4/16→1/16. This smaller
pilot separates namespace diversity from preserving the initial model's outputs.
Its new 512-update control is not a reproduction of the previous 1024-update
recipe. The frozen specification/schedule hashes are in
`evaluation/namespace-retention-v1.json`; no settings are selected by evaluation.

| Arm | Renamed training prefixes | Frozen-teacher KL |
|---|---|---|
| fixed | `nc_` | Off |
| diverse | `cp_`, `token_`, `word_`, `id_` | Off |
| retained | `nc_` | 0.5 |
| diverse_retained | `cp_`, `token_`, `word_`, `id_` | 0.5 |

All arms start from the exact portable both64 checkpoint and identical seed-4242
gated readout, use the same original/renamed source selection and 512 batches of
four examples, and freeze the base for the first 128 updates. Original training
examples retain **full-code targets even during warmup** in all arms; renamed
examples use `def name(` plus EOS during warmup and full-code targets afterward.
Original response supervision therefore remains active while the head learns.
All arms keep the prior source-position NLL coefficient 1 and gate BCE 0.1.
AdamW and decoding match the prior specification. Inputs/tokens and additional
teacher compute vary with the factors; this is not a matched-token/compute study.

The frozen teacher is a separate copy of the common initial gated model, in eval
mode with gradients disabled. Retention arms add 0.5 times token-mean
KL(teacher || student) on **original training response positions only**, excluding
prompts, padding and renamed rows. The teacher is never updated or selected using
development scores. Check teacher parameter identity and absent gradients after
training. This intervention can preserve existing errors; it is not a correctness
oracle. There is one coefficient and seed, no search or early stopping.

Sixteen fixed training-family views are renamed to new suffixes under `nc_`,
`id_`, `q_`, and `rename_`: 64 correlated diagnostic replies. None of those exact
aliases appears in optimization; `q_` and `rename_` prefixes are absent from all
training arms. These are train-family probes, not independent task generalization.
Measure the exact `def requested_name(` prefix separately from parseable complete
functions and exact full replies; all decoding is autonomous and unassisted.

Retain common initial and each arm's final replies on those probes plus the
existing 132 exposed development and 32 training regression views. Also run the
six previously exposed Docker functional regression probes. They are correlated,
not fresh holdout accuracy. Do not open `fresh-probe-v1` during training. Require
two byte-identical trace/bundle repetitions per arm, all finite updates, unchanged
teacher/evaluation parameters and safe Python-only export/load. CPU arithmetic
is native FP32 in the portable environment; same-host repeats do not certify
cross-vendor training. No default, native compatibility or weight promotion.

## Measured result

[Run 39019625587](https://github.com/deser1/NppAI/actions/runs/39019625587)
completed on AMD EPYC 9V74 under pinned Python 3.12.15, Torch 2.14.1+cpu and
NumPy 2.5.2. All eight 512-update trainings completed. Both repetitions of each
arm have identical complete trace and experimental bundle bytes; all four arms
have the same initial student/teacher parameter hashes. Teacher parameters and
evaluation parameters remained unchanged where required.

| Autonomous readout | fixed | diverse | retained | diverse_retained |
|---|---:|---:|---:|---:|
| New train-family alias name prefix | 0/64 | 0/64 | 0/64 | 0/64 |
| New train-family alias exact full reply | 0/64 | 0/64 | 0/64 | 0/64 |
| Parseable single-function train-family replies | 45/64 | 56/64 | 49/64 | 49/64 |
| Original training exact full reply | 1/16 | 2/16 | 6/16 | 1/16 |
| Initial original-training exact hits retained | 0/4 | 0/4 | 0/4 | 0/4 |
| New original-training exact hits gained | 1 | 2 | 6 | 1 |
| Exposed development exact full reply | 3/132 | 1/132 | 3/132 | 1/132 |
| Exposed Docker functional passes | 0/6 | 0/6 | 0/6 | 0/6 |

The common initial model has four exact original-training replies. **All four
are lost in every arm**. The retained arm's total rises to six entirely new hits;
that higher total is not successful retention. Original requested signatures are
1/16, 3/16, 6/16 and 1/16 respectively; only diverse and retained preserve one of
five initial signature hits, while the other two preserve none. Copying new
suffixes fails even with the familiar `nc_` and training-prefix `id_` views, not
only with the two prefixes absent from training. The 64 probe replies represent
16 correlated training-family views across four prefix variants, not 64
independent tasks.

Functional breakdown is fixed: four callable failures, one syntax failure and
one empty reply; diverse: three callable failures, two syntax failures and one
empty reply; both teacher arms: five callable failures and one empty reply.
The worker stops at the first failure. Configured authored cases are not all
executed, and valid syntax does not establish correct function bodies.

Within this one-seed fixed pilot, teacher KL increases total training exact hits
with the fixed prefix but does not preserve initial exact hits. Adding namespace
diversity does not resolve unseen-suffix copying, and the combined arm does not
show additive gains. These observations do not establish universal factor effects
or justify selecting a production recipe. They are not directly comparable to
the prior 1024-update experiment's counts because budget and warmup targets differ.

Original artifact ZIP SHA256/CRC, all repeated trace/bundle bytes, every minibatch
hash and loss composition, raw reply metrics and lost/retained/gained counts were
independently audited. `docs/evidence/namespace-retention/manifest.json` binds
the source artifact and reconstructs the exact report plus all four full traces.
Regression tests reject false retention/prefix/pass claims and altered exposures.
Experimental weights remain in the 14-day Actions artifact; retained hashes alone
cannot reconstruct them.

The next step should diagnose training-family **sequential source-position
transitions and name termination**, plus drift on the four original exact hits,
before another training expansion. A prompt-only monotonic pointer/cursor
mechanism or a protected base readout would require its own fixed protocol,
causal tests, complete-code/retention measurements and eventual fresh functional
evaluation. This report does not promote or implement either proposed mechanism.
