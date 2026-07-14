# m1 / m2 / m3 consolidated (v9, 783-rxn cohort)

3-way comparison of delta-learners over MACE-OFF23_medium features.

| model | baseline | dim | cells |
|-------|----------|-----|-------|
| m1    | geom6              |  6  | 25/25 |
| m2    | xtb_geom6          | 21  | 25/25 |
| m3    | xtb_geom6_plus_v2  | 24  | 25/25 |

## Aggregate NMAE (mean +/- std across 25 cells)

| channel | m1 | m2 | m3 |
|---------|----|----|----|
| strain | 0.526 +/- 0.052 | 0.494 +/- 0.038 | 0.464 +/- 0.034 |
| Pauli | 0.592 +/- 0.062 | 0.485 +/- 0.056 | 0.351 +/- 0.022 |
| Velst | 0.620 +/- 0.050 | 0.498 +/- 0.042 | 0.396 +/- 0.024 |
| oi | 0.590 +/- 0.053 | 0.433 +/- 0.046 | 0.305 +/- 0.016 |
| disp | 0.237 +/- 0.027 | 0.199 +/- 0.024 | 0.206 +/- 0.026 |
| barrier | 0.566 +/- 0.057 | 0.405 +/- 0.035 | 0.373 +/- 0.030 |

## Aggregate RMSE (kcal/mol, mean +/- std)

| channel | m1 | m2 | m3 |
|---------|----|----|----|
| strain | 18.05 +/- 2.14 | 16.19 +/- 1.28 | 15.47 +/- 1.23 |
| Pauli | 77.86 +/- 6.10 | 61.19 +/- 8.46 | 45.49 +/- 6.05 |
| Velst | 35.08 +/- 3.37 | 27.19 +/- 3.77 | 21.87 +/- 3.12 |
| oi | 50.56 +/- 3.22 | 35.34 +/- 4.15 | 25.30 +/- 2.21 |
| disp | 1.96 +/- 0.17 | 1.74 +/- 0.25 | 1.79 +/- 0.21 |
| barrier | 20.97 +/- 2.22 | 15.28 +/- 1.54 | 14.08 +/- 1.17 |

## Figures

- `figures/nmae_bar.png` - per-channel NMAE +/- std, m1/m2/m3 side by side
- `figures/rmse_bar.png` - per-channel RMSE +/- std (kcal/mol)
- `figures/parity_grid.png` - 3 rows (m1/m2/m3) x 6 cols (channels), pooled member-0 across folds

## Regen

`scripts/v9_ml/aggregate_m123_v9.py` (via `aggregate_m123_v9.sh`, idempotent).
