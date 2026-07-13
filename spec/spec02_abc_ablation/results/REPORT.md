# SPEC_02 A/B/C ablation - REPORT

- Cohort: 783 rxns (v9 in-distribution m3, 24-d, 5-fold family-stratified)
- Bootstrap: B=1000, seed=42, reaction-level resampling

## Ensemble NMAE (pooled OOF, 95% CI)

| channel | xgb_direct | ridge_delta | xgb_delta |
|---|---|---|---|
| strain | 0.393 [0.365, 0.421] | 0.476 [0.444, 0.507] | 0.391 [0.362, 0.418] |
| Pauli | 0.272 [0.249, 0.298] | 0.381 [0.350, 0.410] | 0.269 [0.247, 0.296] |
| elst | 0.307 [0.282, 0.336] | 0.431 [0.402, 0.462] | 0.305 [0.279, 0.334] |
| oi | 0.215 [0.194, 0.237] | 0.329 [0.305, 0.353] | 0.214 [0.194, 0.236] |
| disp | 0.156 [0.142, 0.172] | 0.219 [0.203, 0.235] | 0.155 [0.140, 0.170] |
| barrier | 0.395 [0.368, 0.423] | 0.371 [0.347, 0.396] | 0.393 [0.367, 0.422] |

## Pairwise NMAE delta (95% CI over reactions)
See abc_deltas.csv.

## Barrier verdict (B - C)
- delta NMAE = -0.022, 95% CI = [-0.047, +0.002]
- verdict: **indistinguishable (CI crosses 0) - keep ridge for simplicity**