# SPEC_03 summary

## Best per channel (classical baselines)

| channel | best method | NMAE | RMSE |
|---|---|---|---|
| strain | xgb | 0.404 | 13.44 |
| Pauli | xgb | 0.264 | 36.51 |
| elst | xgb | 0.319 | 19.66 |
| oi | xgb | 0.220 | 20.27 |
| disp | xgb | 0.164 | 1.63 |
| barrier_sum | ridge | 0.366 | 13.92 |

## Barrier via sum-of-channels vs direct
See barrier_routes.csv.

## Head-to-head vs neural (v9 m3)
See best_vs_neural.csv.