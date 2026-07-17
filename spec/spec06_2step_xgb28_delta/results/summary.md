# SPEC_06 — 2-arm comparison: xgb_28d (base) vs xgb28 + δ

- Cohort: 783 rxns (v9 in-distribution m3, family-stratified 5-fold, identical split for both arms)
- Bootstrap: B=1000, seed=42, reaction-level resampling
- Descriptor set: 28-d = m3 (d1..d24) ⊕ d25 ⊕ d26 ⊕ d27 ⊕ d28 (spec05 no_sum_28d)
- xgb28_delta: 5 members averaged per (fold, rxn); ModelM1Delta (MACE-OFF23 medium + 4-block CA + AttnPool + MLP)

## Pooled OOF NMAE (95% CI)

| channel | xgb_28d (base) | xgb28 + δ | Δ NMAE (δ − base) |
|---|---|---|---|
| strain | 0.227 [0.208, 0.247] | 0.227 [0.208, 0.247] | +0.000 [-0.000, +0.000]   |
| Pauli | 0.205 [0.184, 0.230] | 0.205 [0.184, 0.230] | -0.000 [-0.001, +0.000]   |
| elst | 0.265 [0.242, 0.291] | 0.264 [0.241, 0.290] | -0.002 [-0.003, -0.001] ✓ |
| oi | 0.149 [0.134, 0.166] | 0.149 [0.134, 0.165] | -0.000 [-0.001, +0.000]   |
| disp | 0.150 [0.136, 0.165] | 0.141 [0.128, 0.156] | -0.009 [-0.013, -0.004] ✓ |
| barrier | 0.299 [0.278, 0.321] | 0.299 [0.278, 0.321] | -0.000 [-0.001, +0.001]   |

## Pooled OOF RMSE (kcal/mol, 95% CI)

| channel | xgb_28d (base) | xgb28 + δ |
|---|---|---|
| strain | 8.398 [7.477, 9.451] | 8.396 [7.481, 9.447] |
| Pauli | 30.192 [26.817, 33.790] | 30.156 [26.797, 33.740] |
| elst | 16.270 [14.579, 17.933] | 16.219 [14.518, 17.899] |
| oi | 15.167 [13.342, 16.952] | 15.149 [13.337, 16.948] |
| disp | 1.559 [1.363, 1.799] | 1.462 [1.273, 1.697] |
| barrier | 12.076 [11.198, 13.033] | 12.074 [11.205, 13.035] |

## Files
- pooled_oof.parquet (xgb28_delta), xgb_28d_oof.parquet (base)
- metrics.csv, head_to_head.csv, leaderboard.csv
- figures/nmae_bars.png, figures/rmse_bars.png, figures/parity_grid.png