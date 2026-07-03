# comparison/ — m1 vs. m2 vs. m3 (5 folds × member 0, no-OOD, outliers removed)

Cross-model evaluation that consumes the per-cell JSONs produced by
`m1/`, `m2/`, and `m3/`. Uses only member 0 from each model so the
three variants are compared on identical HP/seed/split.

## Key numbers (from `report/REPORT.md`)

| channel | m1 R²  | m2 R²  | m3 R²  |
|---|---|---|---|
| strain | 0.327 ± 0.10 | 0.507 ± 0.08 | 0.497 ± 0.08 |
| Pauli  | 0.404 ± 0.08 | 0.546 ± 0.10 | 0.555 ± 0.11 |
| Velst  | 0.376 ± 0.05 | 0.562 ± 0.09 | 0.568 ± 0.11 |
| oi     | 0.409 ± 0.06 | 0.537 ± 0.10 | 0.549 ± 0.14 |
| disp   | 0.908 ± 0.01 | 0.937 ± 0.01 | 0.936 ± 0.01 |

Adding xTB descriptors (m2) gives the big lift over the geometry-only
m1 baseline. The v2 xTB extras (m3) barely move the needle vs. m2 on
most channels; the wins are within noise for strain/disp but slightly
positive on Pauli/oi.

## Layout

- `code/finalize_compare_m1_m2_m3_member0_noOutliers_v2.py` — the
  evaluation script (pooled residual modified-Z outlier removal at
  threshold 5.0, then per-channel MAE/RMSE/NMAE/R²/tail-ratio).
- `code/src/` — shared helpers used by the finalize script
  (baselines, xTB features, inventory).
- `code/submit_finalize_v2.sh` — SLURM submitter.
- `figures/` — per-channel bar charts and parity plots.
- `report/REPORT.md` — full text report with per-channel tables and
  excluded reaction list.
- `report/{metrics,barrier_metrics}_compare.csv` — machine-readable
  numbers.
- `report/mX/*.csv` — per-model breakdown tables.
- `report/excluded_rids.json` — 39 reactions dropped as parity
  outliers.
