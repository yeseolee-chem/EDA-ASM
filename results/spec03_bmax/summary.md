# SPEC_03 — classical b-only maximization (m3, 787-rxn cohort)

4 methods on 24-d descriptors: **ridge, lasso, enet, xgb**.
All tuned by 5-fold CV on each fold's train split.

| channel | best method | best NMAE | M_bδ NMAE | Δ (pp) |
|---|---|---|---|---|
| strain | xgb | 0.759 | 0.798 | -3.9 |
| Pauli | xgb | 0.654 | 0.755 | -10.0 |
| V_elst | xgb | 0.686 | 0.765 | -7.9 |
| oi | xgb | 0.676 | 0.797 | -12.1 |
| disp | xgb | 0.249 | 0.286 | -3.7 |
| barrier | ridge | 0.538 | 0.567 | -2.9 |