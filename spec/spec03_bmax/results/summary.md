# SPEC_03 summary

## Best per channel (classical baselines)

| channel | best method | NMAE | RMSE |
|---|---|---|---|
| strain | xgb | 0.604 | 19.78 |
| Pauli | xgb | 0.428 | 65.31 |
| elst | xgb | 0.420 | 30.55 |
| oi | xgb | 0.405 | 43.44 |
| disp | xgb | 0.186 | 2.11 |
| barrier_sum | ridge | 0.503 | 22.58 |

## Barrier via sum-of-channels vs direct
See barrier_routes.csv.

## Head-to-head vs neural (v7 m3)
See best_vs_neural.csv.