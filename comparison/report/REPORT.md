# m1 vs m2 vs m3 — 5 folds × member 0, no-OOD + parity outliers removed

Same setup as the previous member-0 comparison; HP/seed/split identical across m1/m2/m3.

**Parity-outlier removal**: pooled per-channel residuals; modified Z (median+MAD) > 5.0 ⇒ exclude.
**Excluded reactions**: 39 unique rids (applied uniformly to m1, m2, m3 so the test set stays matched).

Most catastrophic causes:
- `dipolar_003220` — m2 xTB cache returned unphysical E_int = −764.6 kcal/mol; m3 xtb_extra cache missing (archive path gone).
- Several `qmrxn20_*` reactions whose source_dir under `archive/al_v*_baseline_*` is no longer on disk → m3's 6 extras were NaN-imputed.

## Per-channel metrics (mean ± std across 5 folds, outliers removed)

| channel | metric | m1 | m2 | m3 |
|---|---|---|---|---|
| strain | R2_det | 0.327 ± 0.101 | 0.507 ± 0.082 | 0.497 ± 0.081 |
| strain | NMAE | 0.733 ± 0.063 | 0.652 ± 0.048 | 0.656 ± 0.050 |
| strain | MAE | 15.79 ± 1.29 | 14.12 ± 1.98 | 14.20 ± 1.88 |
| strain | RMSE | 21.31 ± 2.17 | 18.29 ± 2.52 | 18.48 ± 2.45 |
| strain | tail_ratio | 1.349 ± 0.035 | 1.296 ± 0.011 | 1.301 ± 0.013 |
| Pauli | R2_det | 0.404 ± 0.078 | 0.546 ± 0.104 | 0.555 ± 0.107 |
| Pauli | NMAE | 0.685 ± 0.049 | 0.634 ± 0.065 | 0.621 ± 0.077 |
| Pauli | MAE | 64.01 ± 4.17 | 59.30 ± 6.06 | 57.97 ± 6.17 |
| Pauli | RMSE | 90.91 ± 6.52 | 79.09 ± 9.43 | 78.11 ± 8.32 |
| Pauli | tail_ratio | 1.420 ± 0.018 | 1.332 ± 0.029 | 1.348 ± 0.032 |
| Velst | R2_det | 0.376 ± 0.052 | 0.562 ± 0.091 | 0.568 ± 0.111 |
| Velst | NMAE | 0.718 ± 0.051 | 0.626 ± 0.070 | 0.615 ± 0.079 |
| Velst | MAE | 29.40 ± 1.21 | 25.62 ± 2.48 | 25.16 ± 2.81 |
| Velst | RMSE | 40.44 ± 1.32 | 33.75 ± 3.40 | 33.44 ± 3.56 |
| Velst | tail_ratio | 1.376 ± 0.024 | 1.317 ± 0.028 | 1.329 ± 0.024 |
| oi | R2_det | 0.409 ± 0.065 | 0.537 ± 0.101 | 0.549 ± 0.141 |
| oi | NMAE | 0.684 ± 0.050 | 0.641 ± 0.089 | 0.616 ± 0.100 |
| oi | MAE | 44.32 ± 2.59 | 41.52 ± 5.51 | 39.88 ± 6.00 |
| oi | RMSE | 62.66 ± 2.90 | 55.30 ± 5.94 | 54.27 ± 7.01 |
| oi | tail_ratio | 1.415 ± 0.040 | 1.337 ± 0.067 | 1.365 ± 0.056 |
| disp | R2_det | 0.908 ± 0.011 | 0.937 ± 0.010 | 0.936 ± 0.009 |
| disp | NMAE | 0.286 ± 0.021 | 0.223 ± 0.020 | 0.225 ± 0.019 |
| disp | MAE | 1.64 ± 0.07 | 1.28 ± 0.08 | 1.29 ± 0.06 |
| disp | RMSE | 2.14 ± 0.10 | 1.77 ± 0.13 | 1.78 ± 0.12 |
| disp | tail_ratio | 1.302 ± 0.029 | 1.384 ± 0.065 | 1.374 ± 0.045 |

## Per-fold test sizes (after outlier removal)

| fold | n_test_original | n_test_kept |
|---|---|---|
| 0 | 126 | 114 |
| 1 | 128 | 124 |
| 2 | 127 | 116 |
| 3 | 127 | 119 |
| 4 | 126 | 122 |

## Excluded reaction IDs
`results_compare_m1_m2_m3_member0_noOutliers/excluded_rids.json` (39 total)
- `dipolar_001236` (dipolar)
- `dipolar_001565` (dipolar)
- `dipolar_002028` (dipolar)
- `dipolar_002593` (dipolar)
- `dipolar_003220` (dipolar)
- `qmrxn20_e2_A_A_A_A_C_B` (qmrxn20_e2)
- `qmrxn20_e2_A_A_A_D_C_B` (qmrxn20_e2)
- `qmrxn20_e2_A_B_A_A_C_B` (qmrxn20_e2)
- `qmrxn20_e2_A_B_D_E_C_B` (qmrxn20_e2)
- `qmrxn20_e2_A_B_E_D_C_B` (qmrxn20_e2)
- `qmrxn20_e2_A_B_E_E_C_B` (qmrxn20_e2)
- `qmrxn20_e2_A_C_A_A_C_B` (qmrxn20_e2)
- `qmrxn20_e2_A_C_A_E_C_B` (qmrxn20_e2)
- `qmrxn20_e2_B_A_A_E_C_B` (qmrxn20_e2)
- `qmrxn20_e2_B_A_E_A_C_B` (qmrxn20_e2)
- `qmrxn20_e2_B_A_E_B_B_B` (qmrxn20_e2)
- `qmrxn20_e2_B_A_E_D_C_B` (qmrxn20_e2)
- `qmrxn20_e2_B_A_E_E_C_B` (qmrxn20_e2)
- `qmrxn20_e2_B_C_A_D_B_B` (qmrxn20_e2)
- `qmrxn20_e2_B_C_E_A_B_B` (qmrxn20_e2)
- `qmrxn20_e2_C_A_A_A_C_B` (qmrxn20_e2)
- `qmrxn20_e2_C_A_C_C_A_A` (qmrxn20_e2)
- `qmrxn20_e2_C_B_D_A_B_B` (qmrxn20_e2)
- `qmrxn20_e2_C_C_A_E_B_B` (qmrxn20_e2)
- `qmrxn20_e2_C_C_C_B_A_A` (qmrxn20_e2)
- `qmrxn20_e2_C_D_E_C_C_B` (qmrxn20_e2)
- `qmrxn20_e2_C_E_D_D_C_B` (qmrxn20_e2)
- `qmrxn20_e2_D_A_C_A_C_B` (qmrxn20_e2)
- `qmrxn20_e2_D_B_A_D_C_B` (qmrxn20_e2)
- `qmrxn20_e2_D_C_A_E_C_B` (qmrxn20_e2)
- `qmrxn20_e2_D_C_D_A_C_B` (qmrxn20_e2)
- `qmrxn20_e2_D_C_D_E_C_B` (qmrxn20_e2)
- `qmrxn20_e2_E_C_E_D_C_B` (qmrxn20_e2)
- `rgd1_MR_170250_0` (rgd1)
- `rgd1_MR_260528_1` (rgd1)
- `rgd1_MR_369786_1` (rgd1)
- `rgd1_MR_446866_1` (rgd1)
- `rgd1_MR_557598_0` (rgd1)
- `rgd1_MR_92891_2` (rgd1)

## Figures (`figures_compare_m1_m2_m3_member0_noOutliers/`)
- `compare_nmae.png`, `compare_r2_det.png`, `compare_mae.png`, `compare_tail_ratio.png`
- `<model>_parity.png`, `compare_parity_grid.png`