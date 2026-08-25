# Phase 6 vs Phase 5 XGB

Setup: Phase 5 features (78, oracle-clean, no filter) + paper 80/10/10 split (5 seeds: 22/23/14/1/2).
Phase 6 adds an attention-pooled MACE-OFF23 residual head on top of the XGB base.

## Per-channel MAE (seed-averaged, kcal/mol)
```
canonical   mae_p5  mae_p6_xgb_only  mae_p6_full  delta_mae  pct_change  mace_contrib
       d1 1.813609         1.813609     1.813517  -0.000092   -0.005061      0.000092
       d2 1.439145         1.439145     1.439073  -0.000072   -0.005035      0.000072
    pauli 5.565936         5.565936     5.566298   0.000363    0.006517     -0.000363
       oi 3.011671         3.011671     3.011545  -0.000126   -0.004169      0.000126
     elst 3.722486         3.722486     3.722576   0.000090    0.002429     -0.000090
     disp 1.010290         1.010290     1.010328   0.000038    0.003772     -0.000038
     cpcm 2.382415         2.382415     2.382321  -0.000094   -0.003962      0.000094
      cds 0.294263         0.294263     0.294274   0.000011    0.003795     -0.000011
```

- `mace_contrib` = XGB-only MAE − full (XGB+MACE) MAE; positive → residual helped.
- `pct_change` = Phase 6 vs Phase 5 XGB percentage change (positive → Phase 6 worse).
