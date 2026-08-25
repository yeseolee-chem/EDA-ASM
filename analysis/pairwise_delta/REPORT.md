# Pairwise δ direct learning — REPORT (spec12)

_Generated: 2026-08-25T05:23:45.001048+00:00, commit `afaed99a564a`_

## Gates
| Gate | Status |
|---|---|
| GATE-0 (data prep) | ✅ PASS n_pairs=1936 unique_rxns=882 n_swaptypes=30 kept_features=59 |
| GATE-1 (split integrity) | ✅ PASS split_A_sizes=[388, 387, 387, 387, 387] split_B_sizes=[387, 388, 386, 387, 388] |
| GATE-2 (Arm-A beats Arm-0 on ≥5/7) | ❌ FAIL Arm-A wins 0/7 (need 5+) — H1 rejected |
| GATE-4 (Arm-C verdict per channel) | ℹ️ C_symmetric/component: goal_hits(A)=0/7 improved_vs_arm0(B+)=0/7 |
| GATE-5 (swaptype generalization) | ✅ PASS all channels ratio<=2 (Arm-C) |

## Main — MAE_δ per (arm, split, channel)

### split = **component**
```
        arm    channel  mae_delta  arm0_mae_delta  improve_vs_arm0  sign_acc  spearman    slope verdict
     A_diff   cpcm_dft   2.546324        2.414926        -0.054411  0.821591  0.848528 0.756310 C_WORSE
     A_diff d1_own_dft   2.089661        1.718079        -0.216278  0.900723  0.944352 0.865581 C_WORSE
     A_diff d2_own_dft   1.930670        1.799376        -0.072967  0.879236  0.933193 0.879271 C_WORSE
     A_diff   disp_dft   1.219414        1.092914        -0.115746  0.858574  0.870949 0.679232 C_WORSE
     A_diff   elst_dft   4.671032        4.201664        -0.111710  0.751653  0.667489 0.424498 C_WORSE
     A_diff     oi_dft   4.845537        3.830580        -0.264962  0.782645  0.720868 0.559212 C_WORSE
     A_diff  pauli_dft   8.481923        6.578410        -0.289358  0.774897  0.709739 0.441108 C_WORSE
   B_concat   cpcm_dft   2.858356        2.414926        -0.183620  0.783264  0.789427 0.680410 C_WORSE
   B_concat d1_own_dft   2.594067        1.718079        -0.509865  0.860744  0.909314 0.752496 C_WORSE
   B_concat d2_own_dft   2.535656        1.799376        -0.409186  0.848140  0.901726 0.792421 C_WORSE
   B_concat   disp_dft   1.520720        1.092914        -0.391437  0.800620  0.787875 0.538452 C_WORSE
   B_concat   elst_dft   5.119574        4.201664        -0.218464  0.735640  0.600548 0.333114 C_WORSE
   B_concat     oi_dft   5.411689        3.830580        -0.412760  0.727893  0.645521 0.471252 C_WORSE
   B_concat  pauli_dft   9.363147        6.578410        -0.423315  0.741529  0.644741 0.394911 C_WORSE
C_symmetric   cpcm_dft   2.582074        2.414926        -0.069215  0.814463  0.844495 0.761912 C_WORSE
C_symmetric d1_own_dft   1.949047        1.718079        -0.134434  0.903409  0.945911 0.880530 C_WORSE
C_symmetric d2_own_dft   1.832741        1.799376        -0.018543  0.889876  0.939334 0.895739 C_WORSE
C_symmetric   disp_dft   1.124927        1.092914        -0.029292  0.871488  0.889920 0.708801 C_WORSE
C_symmetric   elst_dft   4.505095        4.201664        -0.072217  0.760434  0.720038 0.457183 C_WORSE
C_symmetric     oi_dft   4.642777        3.830580        -0.212030  0.777376  0.753186 0.594762 C_WORSE
C_symmetric  pauli_dft   7.649881        6.578410        -0.162877  0.811777  0.797788 0.514711 C_WORSE
```

### split = **swaptype_holdout**
```
        arm    channel  mae_delta  arm0_mae_delta  improve_vs_arm0  sign_acc  spearman    slope    verdict
     A_diff   cpcm_dft   2.507060        2.414926        -0.038152  0.820558  0.847188 0.765052    C_WORSE
     A_diff d1_own_dft   1.873326        1.718079        -0.090361  0.902996  0.945107 0.885272    C_WORSE
     A_diff d2_own_dft   1.815446        1.799376        -0.008931  0.879959  0.931957 0.911772    C_WORSE
     A_diff   disp_dft   1.196534        1.092914        -0.094811  0.864463  0.874940 0.695981    C_WORSE
     A_diff   elst_dft   4.362909        4.201664        -0.038377  0.763017  0.704660 0.483673    C_WORSE
     A_diff     oi_dft   4.409307        3.830580        -0.151081  0.800620  0.748297 0.620921    C_WORSE
     A_diff  pauli_dft   7.645984        6.578410        -0.162285  0.793698  0.748099 0.518445    C_WORSE
   B_concat   cpcm_dft   2.476246        2.414926        -0.025392  0.817872  0.846952 0.730908    C_WORSE
   B_concat d1_own_dft   1.990292        1.718079        -0.158440  0.904959  0.933298 0.812729    C_WORSE
   B_concat d2_own_dft   2.162283        1.799376        -0.201685  0.856508  0.907795 0.848975    C_WORSE
   B_concat   disp_dft   1.335852        1.092914        -0.222285  0.820558  0.834539 0.624063    C_WORSE
   B_concat   elst_dft   4.317790        4.201664        -0.027638  0.768079  0.706528 0.482678    C_WORSE
   B_concat     oi_dft   4.564915        3.830580        -0.191703  0.783471  0.726689 0.594837    C_WORSE
   B_concat  pauli_dft   7.507013        6.578410        -0.141159  0.790186  0.756253 0.540442    C_WORSE
C_symmetric   cpcm_dft   2.073799        2.414926         0.141258  0.855682  0.893276 0.824449 B_IMPROVED
C_symmetric d1_own_dft   1.586730        1.718079         0.076451  0.926653  0.953637 0.897606 B_IMPROVED
C_symmetric d2_own_dft   1.540715        1.799376         0.143750  0.901550  0.949082 0.935826 B_IMPROVED
C_symmetric   disp_dft   0.938094        1.092914         0.141658  0.892562  0.923153 0.775446     A_GOAL
C_symmetric   elst_dft   3.607129        4.201664         0.141500  0.811880  0.816562 0.595910 B_IMPROVED
C_symmetric     oi_dft   3.513249        3.830580         0.082841  0.844421  0.848483 0.710007 B_IMPROVED
C_symmetric  pauli_dft   5.837612        6.578410         0.112611  0.847831  0.866556 0.659232 B_IMPROVED
```

## H5 — swap-type generalization (swap_holdout / component)

Ratio > 2.0 means the model 'memorized swap types'; unseen swaps double the error.
```
        arm    channel  component  swaptype_holdout  ratio_swap_over_comp
     A_diff   cpcm_dft   2.546324          2.507060              0.984580
     A_diff d1_own_dft   2.089661          1.873326              0.896474
     A_diff d2_own_dft   1.930670          1.815446              0.940319
     A_diff   disp_dft   1.219414          1.196534              0.981236
     A_diff   elst_dft   4.671032          4.362909              0.934035
     A_diff     oi_dft   4.845537          4.409307              0.909973
     A_diff  pauli_dft   8.481923          7.645984              0.901445
   B_concat   cpcm_dft   2.858356          2.476246              0.866318
   B_concat d1_own_dft   2.594067          1.990292              0.767248
   B_concat d2_own_dft   2.535656          2.162283              0.852751
   B_concat   disp_dft   1.520720          1.335852              0.878434
   B_concat   elst_dft   5.119574          4.317790              0.843388
   B_concat     oi_dft   5.411689          4.564915              0.843529
   B_concat  pauli_dft   9.363147          7.507013              0.801762
C_symmetric   cpcm_dft   2.582074          2.073799              0.803152
C_symmetric d1_own_dft   1.949047          1.586730              0.814106
C_symmetric d2_own_dft   1.832741          1.540715              0.840661
C_symmetric   disp_dft   1.124927          0.938094              0.833915
C_symmetric   elst_dft   4.505095          3.607129              0.800678
C_symmetric     oi_dft   4.642777          3.513249              0.756713
C_symmetric  pauli_dft   7.649881          5.837612              0.763098
```

## Resolution — sign accuracy binned by |δ_true|

### arm = **A_diff** (split=component)
```
bin         [0,1)  [1,2)  [10,inf)  [2,3)  [3,5)  [5,10)
channel                                                 
cpcm_dft    0.558  0.676     1.000  0.742  0.866   0.963
d1_own_dft  0.670  0.856     1.000  0.918  0.940   0.980
d2_own_dft  0.663  0.773     1.000  0.807  0.957   0.974
disp_dft    0.660  0.868     1.000  0.934  0.976   0.997
elst_dft    0.509  0.594     0.905  0.665  0.757   0.827
oi_dft      0.549  0.655     0.905  0.683  0.764   0.859
pauli_dft   0.491  0.590     0.879  0.629  0.655   0.805
```

### arm = **B_concat** (split=component)
```
bin         [0,1)  [1,2)  [10,inf)  [2,3)  [3,5)  [5,10)
channel                                                 
cpcm_dft    0.573  0.628     1.000  0.696  0.790   0.912
d1_own_dft  0.639  0.744     0.996  0.830  0.882   0.975
d2_own_dft  0.603  0.692     0.999  0.807  0.919   0.955
disp_dft    0.599  0.749     1.000  0.895  0.945   0.993
elst_dft    0.542  0.594     0.877  0.662  0.707   0.808
oi_dft      0.496  0.587     0.892  0.645  0.707   0.764
pauli_dft   0.536  0.559     0.851  0.579  0.616   0.741
```

### arm = **C_symmetric** (split=component)
```
bin         [0,1)  [1,2)  [10,inf)  [2,3)  [3,5)  [5,10)
channel                                                 
cpcm_dft    0.564  0.689     0.996  0.738  0.828   0.956
d1_own_dft  0.686  0.856     1.000  0.927  0.934   0.978
d2_own_dft  0.690  0.794     1.000  0.836  0.957   0.975
disp_dft    0.680  0.879     1.000  0.947  0.986   0.999
elst_dft    0.509  0.570     0.926  0.646  0.783   0.843
oi_dft      0.499  0.612     0.932  0.737  0.775   0.826
pauli_dft   0.507  0.619     0.922  0.647  0.693   0.844
```

## Invariant checks (seed=42, fold=0)
- self_check: predicted δ(A,A) should be ~0
- antisym: |δ(A,B) + δ(B,A)| should be ~0
```
        arm    channel  self_mean_abs  self_median_abs  self_p95_abs  asym_mean  asym_median  asym_p95
     A_diff d1_own_dft       0.203265         0.203265      0.203265   1.823063     1.207197  5.798492
     A_diff d2_own_dft       0.007613         0.007613      0.007613   2.075652     1.455796  5.920786
     A_diff   elst_dft       1.162224         1.162224      1.162224   4.519629     3.337282 12.612556
     A_diff  pauli_dft       1.206340         1.206340      1.206340   7.426825     4.835583 23.369379
     A_diff     oi_dft       0.549976         0.549976      0.549976   4.622206     3.269927 14.778053
     A_diff   disp_dft       0.125671         0.125671      0.125671   1.004739     0.753492  2.779476
     A_diff   cpcm_dft       0.455038         0.455038      0.455038   2.571075     2.057817  6.622783
   B_concat d1_own_dft       1.260312         0.846707      3.863322   2.184890     1.406956  6.666464
   B_concat d2_own_dft       0.716273         0.490275      2.123320   1.952315     1.321699  5.899984
   B_concat   elst_dft       2.513543         1.881713      6.624023   4.546794     3.379677 12.692541
   B_concat  pauli_dft       3.980142         2.833710     11.881712   7.405544     5.460434 20.685838
   B_concat     oi_dft       2.605643         1.968473      7.022168   4.819021     3.530385 14.064540
   B_concat   disp_dft       0.716335         0.557528      1.994030   1.433191     1.244916  3.493169
   B_concat   cpcm_dft       1.315137         1.018636      3.519797   2.580522     2.052346  6.520974
C_symmetric d1_own_dft       0.309177         0.228520      0.926659   0.160163     0.071321  0.661937
C_symmetric d2_own_dft       0.152104         0.116564      0.419837   0.167104     0.060576  0.822137
C_symmetric   elst_dft       0.529841         0.393826      1.495495   0.348197     0.153062  1.480880
C_symmetric  pauli_dft       0.794840         0.524770      2.738558   0.570039     0.231700  2.456606
C_symmetric     oi_dft       0.469938         0.360558      1.327127   0.330737     0.135132  1.505615
C_symmetric   disp_dft       0.129996         0.102458      0.319376   0.108284     0.042715  0.490776
C_symmetric   cpcm_dft       0.298852         0.249467      0.775503   0.198263     0.095660  0.799487
```

## ρ_equiv — 'ρ boost' that subtract-baseline would need to match this arm
```
        arm            split    channel  abs_mae_arm0  mae_delta  arm0_mae_delta  rho_equiv
     A_diff        component   cpcm_dft      2.323243   2.546324        2.414926   0.399368
     A_diff        component d1_own_dft      1.825004   2.089661        1.718079   0.344468
     A_diff        component d2_own_dft      1.565030   1.930670        1.799376   0.239077
     A_diff        component   disp_dft      1.023051   1.219414        1.092914   0.289641
     A_diff        component   elst_dft      3.819571   4.671032        4.201664   0.252233
     A_diff        component     oi_dft      3.079482   4.845537        3.830580  -0.237937
     A_diff        component  pauli_dft      5.664576   8.481923        6.578410  -0.121047
     A_diff swaptype_holdout   cpcm_dft      2.323243   2.507060        2.414926   0.417749
     A_diff swaptype_holdout d1_own_dft      1.825004   1.873326        1.718079   0.473172
     A_diff swaptype_holdout d2_own_dft      1.565030   1.815446        1.799376   0.327191
     A_diff swaptype_holdout   disp_dft      1.023051   1.196534        1.092914   0.316049
     A_diff swaptype_holdout   elst_dft      3.819571   4.362909        4.201664   0.347631
     A_diff swaptype_holdout     oi_dft      3.079482   4.409307        3.830580  -0.025074
     A_diff swaptype_holdout  pauli_dft      5.664576   7.645984        6.578410   0.089034
   B_concat        component   cpcm_dft      2.323243   2.858356        2.414926   0.243143
   B_concat        component d1_own_dft      1.825004   2.594067        1.718079  -0.010193
   B_concat        component d2_own_dft      1.565030   2.535656        1.799376  -0.312518
   B_concat        component   disp_dft      1.023051   1.520720        1.092914  -0.104775
   B_concat        component   elst_dft      3.819571   5.119574        4.201664   0.101727
   B_concat        component     oi_dft      3.079482   5.411689        3.830580  -0.544118
   B_concat        component  pauli_dft      5.664576   9.363147        6.578410  -0.366089
   B_concat swaptype_holdout   cpcm_dft      2.323243   2.476246        2.414926   0.431974
   B_concat swaptype_holdout d1_own_dft      1.825004   1.990292        1.718079   0.405330
   B_concat swaptype_holdout d2_own_dft      1.565030   2.162283        1.799376   0.045558
   B_concat swaptype_holdout   disp_dft      1.023051   1.335852        1.092914   0.147505
   B_concat swaptype_holdout   elst_dft      3.819571   4.317790        4.201664   0.361055
   B_concat swaptype_holdout     oi_dft      3.079482   4.564915        3.830580  -0.098702
   B_concat swaptype_holdout  pauli_dft      5.664576   7.507013        6.578410   0.121848
C_symmetric        component   cpcm_dft      2.323243   2.582074        2.414926   0.382384
C_symmetric        component d1_own_dft      1.825004   1.949047        1.718079   0.429722
C_symmetric        component d2_own_dft      1.565030   1.832741        1.799376   0.314311
C_symmetric        component   disp_dft      1.023051   1.124927        1.092914   0.395462
C_symmetric        component   elst_dft      3.819571   4.505095        4.201664   0.304417
C_symmetric        component     oi_dft      3.079482   4.642777        3.830580  -0.136503
C_symmetric        component  pauli_dft      5.664576   7.649881        6.578410   0.088106
C_symmetric swaptype_holdout   cpcm_dft      2.323243   2.073799        2.414926   0.601605
C_symmetric swaptype_holdout d1_own_dft      1.825004   1.586730        1.718079   0.622038
C_symmetric swaptype_holdout d2_own_dft      1.565030   1.540715        1.799376   0.515416
C_symmetric swaptype_holdout   disp_dft      1.023051   0.938094        1.092914   0.579595
C_symmetric swaptype_holdout   elst_dft      3.819571   3.607129        4.201664   0.554073
C_symmetric swaptype_holdout     oi_dft      3.079482   3.513249        3.830580   0.349222
C_symmetric swaptype_holdout  pauli_dft      5.664576   5.837612        6.578410   0.468986
```

## Files
```
artifacts/GATE0_STATUS.txt
artifacts/GATE1_STATUS.txt
artifacts/GATE2_STATUS.txt
artifacts/GATE4_STATUS.txt
artifacts/GATE5_STATUS.txt
artifacts/delta_mae_by_arm.csv
artifacts/dropped_zerovar_columns.txt
artifacts/fit_log.csv
artifacts/invariant_check.csv
artifacts/pair_dataset.pkl
artifacts/per_seed_metrics.csv
artifacts/predictions_all.pkl
artifacts/resolution_table.csv
artifacts/rho_equiv.csv
artifacts/run.896065.log
artifacts/run.896125.log
artifacts/splits_summary.txt
artifacts/swap_type_performance.csv
artifacts/swaptype_generalization.csv
```