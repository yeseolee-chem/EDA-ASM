# eda-asm-prediction

EDA (Energy Decomposition Analysis) + ASM (Activation Strain Model)
proxies for activation-energy prediction. Stage 5–6 of the explainable
E_a pipeline; complements Stage 2 (TS structure generation) in
`ts-structure-prediction`.

## Layout

```
labels/         789-reaction ADF + ORCA EDA-ASM labels (parquet)
V1/             Claisen ASR-EDA v1 (spec, scripts, ADF/ORCA runs, viz)
m1/             Δ-learner, geom6 baseline (5 folds × 5 members)
m2/             Δ-learner, xtb_geom6 baseline (5 folds × 5 members)
m3/             Δ-learner, xtb_geom6_plus_v2 baseline (5 folds × member 0)
comparison/     m1 vs. m2 vs. m3 evaluation (figures + report)
CLAUDE.md       upstream task spec (older 400-trajectory phase)
```

Each of `labels/`, `V1/`, `m1/`, `m2/`, `m3/`, `comparison/` has its
own README with details and reproduction commands.

## Datasets — `labels/`

- `labels/adf/adf_labels_v6_multifamily.parquet` — 789 rows, canonical
  ADF NOCV-EDA labels across dipolar / rgd1 / qmrxn20_e2 / qmrxn20_sn2.
- `labels/orca/orca_eda_labels.parquet` — matching ORCA EDA recompute.
- `labels/orca/orca_strain_labels.parquet` — fragment-relaxed strain
  energies from ORCA.
- `labels/adf_vs_orca_{,full_}comparison.parquet` — side-by-side.

## Models — `m1/`, `m2/`, `m3/`

All three are Track B `m1_delta` (cross-attention + Δ-learning) with the
low-LR budget (LR 1e-5, ≤100k epochs, patience 10k) and share
`trackB_no_ood` splits. They differ only in the physics baseline:

| model | baseline                | features                                   | cells trained |
|-------|-------------------------|--------------------------------------------|---------------|
| m1    | `geom6`                 | d1–d6 (geometry only, no xTB)              | 25 (5×5)      |
| m2    | `xtb_geom6`             | d1–d6 + xTB descriptors                    | 25 (5×5)      |
| m3    | `xtb_geom6_plus_v2`     | d1–d6 + xTB + v2 xTB extras (rebuilt cache) | 5  (5×1)      |

Per-cell test outputs (`fold*/member*.json`) live under each folder's
`results/`; training code + SLURM submitters live under `code/`.

## Comparison — `comparison/`

Cross-model report at `comparison/report/REPORT.md`. Headline: xTB
descriptors (m2) give a large lift over the geometry-only m1 baseline;
the v2 xTB extras (m3) yield small gains on Pauli/oi and are within
noise on strain/disp.

## Provenance

The larger `runs/`, `outputs/`, and `analysis/` trees from the earlier
build have been retired — the frozen JSON outputs under each `m*/results/`
and the 789-row parquets under `labels/` are the archived deliverables.
