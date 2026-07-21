# SPEC_11 - 2-arm comparison: xgb_33d (base) vs xgb33 + delta

- Cohort: 783 rxns (v9 in-distribution m3, family-stratified 5-fold, identical split for both arms)
- Bootstrap: B=1000, seed=42, reaction-level resampling
- Descriptor set: 33-d = m3 (d1..d24) + d25 + d26 + d27 + d28 + d29 + d30 + d31 + d32 + d33
- xgb33+delta: MACE-OFF23 medium + 4-block CA + AttnPool + MLP

## Pooled OOF NMAE (95% CI)

| channel | xgb_33d (base) | xgb33 + delta | delta NMAE (arm2 - arm1) |
|---|---|---|---|
| strain | 0.219 [0.200, 0.238] | 0.219 [0.200, 0.238] | +0.000 [-0.000, +0.000]   |
| Pauli | 0.197 [0.177, 0.219] | 0.197 [0.177, 0.218] | -0.000 [-0.000, +0.000]   |
| elst | 0.251 [0.229, 0.276] | 0.251 [0.229, 0.276] | -0.000 [-0.000, +0.000]   |
| oi | 0.141 [0.126, 0.158] | 0.141 [0.126, 0.158] | -0.000 [-0.000, +0.000]   |
| disp | 0.130 [0.117, 0.145] | 0.130 [0.117, 0.145] | +0.000 [-0.000, +0.000]   |
| barrier | 0.291 [0.271, 0.312] | 0.291 [0.271, 0.312] | +0.000 [-0.000, +0.000]   |

## Pooled OOF RMSE (kcal/mol, 95% CI)

| channel | xgb_33d (base) | xgb33 + delta |
|---|---|---|
| strain | 8.015 [7.117, 9.075] | 8.013 [7.118, 9.069] |
| Pauli | 28.758 [25.542, 32.122] | 28.760 [25.537, 32.121] |
| elst | 15.292 [13.694, 16.969] | 15.292 [13.689, 16.978] |
| oi | 14.479 [12.796, 16.223] | 14.480 [12.795, 16.222] |
| disp | 1.389 [1.181, 1.636] | 1.388 [1.179, 1.635] |
| barrier | 11.655 [10.810, 12.495] | 11.661 [10.825, 12.502] |

## Files
- pooled_oof.parquet (xgb33+delta), xgb_33d_oof.parquet (base)
- metrics.csv, head_to_head.csv, leaderboard.csv
- figures/nmae_bars.png, figures/rmse_bars.png, figures/parity_grid.png