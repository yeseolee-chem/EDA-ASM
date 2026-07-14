# m3 (v9, 783-rxn cohort)

Delta-learner over MACE-OFF23_medium features with a 24-d physics baseline
(d1..d21 xTB/geom + d22 = mu^2 / 2 eta, d23 = sum q^2, d24 = sum |WBO_AB|).

## Bundle + splits

- Bundle: `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt`
- Splits: `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9/fold{0..4}/`
- Labels: `outputs/v8_review/labels/labels_v9_5channel.LOCKED_783.parquet`
- Cells:  `m3/code/trackB_lowlr_v9_xtb_geom6_plus_v2/m1_delta/fold{0..4}/member{0..4}.json`
- Fold x member = 5 x 5 = 25 cells (25/25 available).

## Aggregate metrics (mean +/- std across cells)

| channel  | MAE (kcal/mol) | RMSE (kcal/mol) | NMAE | R^2 |
|----------|----------------|-----------------|------|-----|
| strain   | 10.72 +/- 0.70 | 15.47 +/- 1.23 | 0.464 +/- 0.034 | 0.697 +/- 0.054 |
| Pauli    | 28.59 +/- 2.21 | 45.49 +/- 6.05 | 0.351 +/- 0.022 | 0.837 +/- 0.034 |
| Velst    | 14.52 +/- 1.13 | 21.87 +/- 3.12 | 0.396 +/- 0.024 | 0.808 +/- 0.042 |
| oi       | 16.80 +/- 1.26 | 25.30 +/- 2.21 | 0.305 +/- 0.016 | 0.882 +/- 0.015 |
| disp     | 1.20 +/- 0.14 | 1.79 +/- 0.21 | 0.206 +/- 0.026 | 0.941 +/- 0.012 |
| barrier  | 10.29 +/- 0.71 | 14.08 +/- 1.17 | 0.373 +/- 0.030 | 0.835 +/- 0.026 |

## Figures

- `figures/nmae_bar.png`  — per-channel NMAE +/- std
- `figures/rmse_bar.png`  — per-channel RMSE +/- std (kcal/mol)
- `figures/mae_bar.png`   — per-channel MAE +/- std (kcal/mol)
- `figures/parity_grid.png` — parity (pooled member-0 predictions across 5 folds)

## Regen script

`scripts/v9_ml/regen_m3_figures_v9.py` (submitted via `regen_m3_figures_v9.sh`).
Idempotent: overwrites existing outputs.
