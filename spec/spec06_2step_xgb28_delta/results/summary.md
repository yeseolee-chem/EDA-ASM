# SPEC_06 — 2-step xgb28 + δ — summary

- Cohort: 783 rxns (v9 in-distribution m3, family-stratified 5-fold)
- Bootstrap: B=1000, seed=42, reaction-level resampling
- Descriptor set: 28-d = m3 (d1..d24) ⊕ d25 ⊕ d26 ⊕ d27 ⊕ d28 (spec05 no_sum_28d)
- Delta model: ModelM1Delta (MACE-OFF23 medium + 4-block CA + AttnPool + MLP)
- Fixed hp: LR=1e-05, WD=0.001, EPOCHS_MAX=100000, PATIENCE=10000, batch=16

## Pooled OOF NMAE (95% CI)

| channel | xgb28_delta | ridge_delta | xgb_delta | xgb_24d |
|---|---|---|---|---|
| strain | 0.227 [0.208, 0.247] | 0.476 [0.443, 0.509] | 0.391 [0.363, 0.420] | 0.393 [0.365, 0.423] |
| Pauli | 0.205 [0.184, 0.230] | 0.381 [0.352, 0.415] | 0.269 [0.246, 0.295] | 0.272 [0.248, 0.298] |
| elst | 0.265 [0.242, 0.291] | 0.431 [0.401, 0.466] | 0.305 [0.278, 0.333] | 0.307 [0.282, 0.336] |
| oi | 0.149 [0.134, 0.166] | 0.329 [0.306, 0.356] | 0.214 [0.195, 0.235] | 0.215 [0.196, 0.236] |
| disp | 0.142 [0.129, 0.157] | 0.219 [0.204, 0.236] | 0.155 [0.141, 0.172] | 0.156 [0.142, 0.173] |
| barrier | 0.299 [0.278, 0.322] | 0.371 [0.346, 0.397] | 0.393 [0.367, 0.422] | 0.395 [0.370, 0.423] |

## Head-to-head vs other arms (NMAE delta, 95% CI)

### xgb28_delta − ridge_delta

| channel | Δ NMAE | 95% CI |
|---|---|---|
| strain | -0.249 | [-0.278, -0.224] ✓ (better) |
| Pauli | -0.176 | [-0.203, -0.151] ✓ (better) |
| elst | -0.166 | [-0.195, -0.138] ✓ (better) |
| oi | -0.180 | [-0.202, -0.158] ✓ (better) |
| disp | -0.077 | [-0.092, -0.063] ✓ (better) |
| barrier | -0.071 | [-0.100, -0.045] ✓ (better) |

### xgb28_delta − xgb_delta

| channel | Δ NMAE | 95% CI |
|---|---|---|
| strain | -0.164 | [-0.186, -0.141] ✓ (better) |
| Pauli | -0.064 | [-0.083, -0.046] ✓ (better) |
| elst | -0.040 | [-0.057, -0.021] ✓ (better) |
| oi | -0.065 | [-0.080, -0.049] ✓ (better) |
| disp | -0.013 | [-0.020, -0.007] ✓ (better) |
| barrier | -0.094 | [-0.120, -0.068] ✓ (better) |

### xgb28_delta − xgb_24d

| channel | Δ NMAE | 95% CI |
|---|---|---|
| strain | -0.166 | [-0.189, -0.144] ✓ (better) |
| Pauli | -0.067 | [-0.086, -0.048] ✓ (better) |
| elst | -0.042 | [-0.060, -0.023] ✓ (better) |
| oi | -0.066 | [-0.081, -0.051] ✓ (better) |
| disp | -0.014 | [-0.022, -0.007] ✓ (better) |
| barrier | -0.096 | [-0.122, -0.069] ✓ (better) |

## Reference: spec05 xgb_28d base-only (fold0)

| channel | NMAE (base fold0) |
|---|---|
| strain | 0.232 |
| Pauli | 0.242 |
| elst | 0.309 |
| oi | 0.160 |
| disp | 0.138 |
| barrier | 0.288 |

## Files
- pooled_oof.parquet, metrics.csv, head_to_head.csv, leaderboard.csv
- figures/nmae_bars.png, figures/rmse_bars.png, figures/parity_grid.png