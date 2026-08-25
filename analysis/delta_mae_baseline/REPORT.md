# δ-MAE Baseline — REPORT (spec11)

_Generated: 2026-08-25T02:31:58.636996+00:00, commit `86383aa2068e`_

## Gates
| Gate | Status |
|---|---|
| GATE-0 (data integrity) | ✅ PASS phase5=3504 pairs_dedup=1936 unique_rxns=882 |
| GATE-1 (absolute MAE within ±10%) | ✅ PASS kfold_random within ±10% of phase5 reference |
| GATE-2 (mae_δ < √2·abs_mae) | ✅ PASS mae_delta < sqrt(2)*abs_mae for all (scheme,channel) |
| GATE-3 (ρ_pair vs required — informational) | ℹ️ groupkfold_component: |

## Q — δ-MAE per channel (main deliverable)

### scheme = **groupkfold_component**
```
            channel  abs_mae  baseline_mae_delta  mae_delta  improvement_over_zero  sign_accuracy  spearman_delta   verdict  excluded_from_learning
            cds_dft 0.306710            0.379497   0.339683               0.104913       0.703926        0.533120    A_GOAL                    True
           disp_dft 1.023051            2.498613   1.092914               0.562592       0.867872        0.893882    B_NEAR                   False
         d1_own_dft 1.825004            6.614572   1.718079               0.740259       0.900620        0.949883    B_NEAR                   False
         d2_own_dft 1.565030            6.783577   1.799376               0.734745       0.875310        0.933399    B_NEAR                   False
interaction_own_dft 2.008266            4.674186   2.296249               0.508738       0.819938        0.851526 C_LIMITED                    True
           cpcm_dft 2.323243            5.161780   2.414926               0.532152       0.835950        0.858563 C_LIMITED                   False
             oi_dft 3.079482            7.644814   3.830580               0.498931       0.823864        0.831565 C_LIMITED                   False
           elst_dft 3.819571            6.618379   4.201664               0.365152       0.777169        0.754452 C_LIMITED                   False
          pauli_dft 5.664576           12.601847   6.578410               0.477981       0.832025        0.836678 C_LIMITED                   False
```

### scheme = **kfold_random**
```
            channel  abs_mae  baseline_mae_delta  mae_delta  improvement_over_zero  sign_accuracy  spearman_delta   verdict  excluded_from_learning
            cds_dft 0.297686            0.379497   0.307824               0.188862       0.718595        0.560241    A_GOAL                    True
           disp_dft 0.998293            2.498613   1.061513               0.575159       0.866426        0.898821    B_NEAR                   False
         d2_own_dft 1.497082            6.783577   1.623227               0.760712       0.879029        0.939328    B_NEAR                   False
         d1_own_dft 1.818215            6.614572   1.680200               0.745985       0.899897        0.949149    B_NEAR                   False
interaction_own_dft 1.981535            4.674186   2.203006               0.528687       0.825826        0.856640 C_LIMITED                    True
           cpcm_dft 2.277248            5.161780   2.347552               0.545205       0.833161        0.864518 C_LIMITED                   False
             oi_dft 2.971422            7.644814   3.606504               0.528242       0.832335        0.847156 C_LIMITED                   False
           elst_dft 3.719860            6.618379   4.025763               0.391730       0.786054        0.770794 C_LIMITED                   False
          pauli_dft 5.486885           12.601847   6.058963               0.519200       0.841219        0.856999 C_LIMITED                   False
```

## ρ_pair — pairwise error correlation (strategy decision)

### scheme = **groupkfold_component**
```
            channel  abs_mae_pair  rho_pair_measured  predicted_mae_delta  required_rho_for_1kcal  gap_actual_minus_required strategy_tier
            cds_dft      0.271557           0.231049             0.336752               -5.785401                   6.016450  A_SUFFICIENT
           cpcm_dft      2.105890           0.322099             2.451238                0.887036                  -0.564937        C_HARD
         d1_own_dft      1.285913           0.135305             1.690683                0.697388                  -0.562083     D_NO_HOPE
         d2_own_dft      1.462552           0.288620             1.744009                0.766035                  -0.477414     D_NO_HOPE
           disp_dft      0.891564           0.286420             1.064553                0.370060                  -0.083640     D_NO_HOPE
           elst_dft      3.628974           0.319677             4.231840                0.962005                  -0.642328        C_HARD
interaction_own_dft      1.805221           0.189563             2.297974                0.846412                  -0.656850     D_NO_HOPE
             oi_dft      3.176448           0.320787             3.701576                0.950395                  -0.629608        C_HARD
          pauli_dft      5.865742           0.363504             6.615418                0.985462                  -0.621958        C_HARD
```

### scheme = **kfold_random**
```
            channel  abs_mae_pair  rho_pair_measured  predicted_mae_delta  required_rho_for_1kcal  gap_actual_minus_required strategy_tier
            cds_dft      0.230646           0.165241             0.298010               -8.407595                   8.572836  A_SUFFICIENT
           cpcm_dft      1.858282           0.207968             2.338814                0.855177                  -0.647209     D_NO_HOPE
         d1_own_dft      1.198873           0.076592             1.629153                0.651280                  -0.574688     D_NO_HOPE
         d2_own_dft      1.171385           0.110057             1.562857                0.635046                  -0.524989     D_NO_HOPE
           disp_dft      0.785996           0.130953             1.036235                0.190287                  -0.059335     D_NO_HOPE
           elst_dft      3.117334           0.205529             3.930181                0.948458                  -0.742929     D_NO_HOPE
interaction_own_dft      1.612576           0.090600             2.174695                0.807685                  -0.717085     D_NO_HOPE
             oi_dft      2.721806           0.191521             3.460809                0.932437                  -0.740916     D_NO_HOPE
          pauli_dft      4.820861           0.248441             5.911015                0.978471                  -0.730031     D_NO_HOPE
```

## Split-scheme comparison (leakage size)
```
scheme               groupkfold_component  kfold_random  leakage_ratio  abs_ratio_ref
channel                                                                              
cds_dft                          0.339683      0.307824       0.906212            1.0
cpcm_dft                         2.414926      2.347552       0.972101            1.0
d1_own_dft                       1.718079      1.680200       0.977953            1.0
d2_own_dft                       1.799376      1.623227       0.902106            1.0
disp_dft                         1.092914      1.061513       0.971268            1.0
elst_dft                         4.201664      4.025763       0.958136            1.0
interaction_own_dft              2.296249      2.203006       0.959393            1.0
oi_dft                           3.830580      3.606504       0.941504            1.0
pauli_dft                        6.578410      6.058963       0.921038            1.0
```

A ratio < 1.0 in `leakage_ratio` means random KFold is optimistic 
(same connected component leaked between train/test).

## Cross-channel δ-ERROR correlation

> ⚠️ This is **model prediction error** correlation — different from 
> `mmp_gate_a` §5.4 which was correlation of **actual δ values** (physics).
> - Actual δ correlation → physics; channels co-move
> - **Prediction-error correlation → model limitation**; what one channel
>   gets wrong, another gets wrong the same way
> Do not conflate.

### scheme = **kfold_random**
```
            d1_own_dft  d2_own_dft  elst_dft  pauli_dft  oi_dft  disp_dft  cpcm_dft  cds_dft
d1_own_dft       1.000      -0.239     0.068     -0.062   0.005    -0.096    -0.028   -0.041
d2_own_dft      -0.239       1.000     0.010     -0.000  -0.111    -0.080     0.009   -0.061
elst_dft         0.068       0.010     1.000     -0.720   0.325     0.082    -0.216   -0.029
pauli_dft       -0.062      -0.000    -0.720      1.000  -0.492    -0.124     0.039   -0.062
oi_dft           0.005      -0.111     0.325     -0.492   1.000     0.036    -0.245    0.080
disp_dft        -0.096      -0.080     0.082     -0.124   0.036     1.000     0.084    0.285
cpcm_dft        -0.028       0.009    -0.216      0.039  -0.245     0.084     1.000    0.093
cds_dft         -0.041      -0.061    -0.029     -0.062   0.080     0.285     0.093    1.000
```
cancel_ratio=0.2276 for scheme=kfold_random

### scheme = **groupkfold_component**
```
            d1_own_dft  d2_own_dft  elst_dft  pauli_dft  oi_dft  disp_dft  cpcm_dft  cds_dft
d1_own_dft       1.000      -0.273     0.066     -0.065  -0.010    -0.051    -0.012   -0.022
d2_own_dft      -0.273       1.000     0.046     -0.053  -0.070    -0.091     0.036   -0.004
elst_dft         0.066       0.046     1.000     -0.749   0.380     0.080    -0.186   -0.018
pauli_dft       -0.065      -0.053    -0.749      1.000  -0.556    -0.121     0.018   -0.041
oi_dft          -0.010      -0.070     0.380     -0.556   1.000     0.059    -0.194    0.024
disp_dft        -0.051      -0.091     0.080     -0.121   0.059     1.000     0.118    0.212
cpcm_dft        -0.012       0.036    -0.186      0.018  -0.194     0.118     1.000    0.080
cds_dft         -0.022      -0.004    -0.018     -0.041   0.024     0.212     0.080    1.000
```
cancel_ratio=0.2173 for scheme=groupkfold_component

## Barrier δ — direct vs channel-sum
```
              scheme  direct_mae_delta  direct_baseline  sum_mae_delta  sum_baseline  n_wins_direct
groupkfold_component          2.296249         4.674186       4.457305      4.912946              5
        kfold_random          2.203006         4.674186       4.360082      4.912946              5
```

## Files
```
artifacts/GATE0_STATUS.txt
artifacts/GATE1_STATUS.txt
artifacts/GATE2_STATUS.txt
artifacts/GATE3_STATUS.txt
artifacts/barrier_delta.csv
artifacts/barrier_delta_per_seed.csv
artifacts/cancellation_groupkfold_component.txt
artifacts/cancellation_kfold_random.txt
artifacts/components.pkl
artifacts/delta_mae_per_seed.csv
artifacts/delta_mae_table.csv
artifacts/error_corr_pearson_groupkfold_component.csv
artifacts/error_corr_pearson_kfold_random.csv
artifacts/error_corr_spearman_groupkfold_component.csv
artifacts/error_corr_spearman_kfold_random.csv
artifacts/gate1_absolute_mae_check.csv
artifacts/oof_predictions.pkl
artifacts/pairs_dedup.pkl
artifacts/rho_pair.csv
artifacts/rho_pair_per_seed.csv
artifacts/run.896016.log
artifacts/test_mae_per_seed.csv
artifacts/train_mae_per_seed.csv
```