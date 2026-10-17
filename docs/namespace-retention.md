# Namespace and original-task retention pilot

Status: predeclared 2×2 experiment; measurement pending. No promotion.

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
