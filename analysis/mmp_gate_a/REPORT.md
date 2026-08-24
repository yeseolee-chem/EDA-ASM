# MMP Gate A — Report

_Generated: 2026-08-24T23:45:40.146924+00:00, commit `1cba33d18445`_

## Gate summary
| Gate | Status |
|---|---|
| GATE-0 (cohort join, composition) | ✅ PASS (asserted in 00_load_join.py) |
| GATE-1 (known-library hit rate) | ❌ FAIL hit_rate=93.56% |
| GATE-2 (eda.inp fragment cross-check) | ✅ PASS unique_rate=100.00% ambig=0 mismatch=0 |
| GATE-4 (8-channel MMP pair count) | ✅ PASS n_all8=4230 |
| GATE-5 (channel δ distribution) | ⚠️ WARN |
| GATE-6 (outlier fraction) | ⚠️ REVIEW frac_outliers=0.085 |

## Q1 — 8-channel-complete MMP pair count
```
Q1: 8-channel-complete MMP pairs = 4230
Total pairs: 4230
  d1: 4230
  d2: 4230
  elst: 4230
  pauli: 4230
  oi: 4230
  disp: 4230
  cpcm: 4230
  cds: 4230
```

## Q2 — per-channel δ distribution
```
channel    n  mean_abs  median_abs  median_abs_ci_lo  median_abs_ci_hi  std_signed  frac_lt_1.0  baseline_MAE
     d1 4230  7.317563    4.941317          4.663517          5.176811   10.071487     0.154137      7.317563
     d2 4230  7.467203    4.697357          4.464000          4.940791   10.596512     0.160993      7.467203
   elst 4230  8.169478    6.129400          5.952540          6.353618   10.935431     0.099527      8.169478
  pauli 4230 16.009333   11.882115         11.474416         12.275830   21.603669     0.053901     16.009333
     oi 4230  9.159282    6.426329          6.211929          6.773388   12.457177     0.085579      9.159282
   disp 4230  2.878887    2.251032          2.170319          2.324001    3.754173     0.250355      2.878887
   cpcm 4230  5.038224    3.708221          3.592750          3.817101    6.901888     0.144917      5.038224
    cds 4230  0.512235    0.348795          0.338034          0.360139    0.691809     0.848463      0.512235
```

## Cross-channel δ Pearson correlation (all-8 subset)
```
             delta_d1  delta_d2  delta_elst  delta_pauli  delta_oi  delta_disp  delta_cpcm  delta_cds
delta_d1        1.000    -0.445      -0.168        0.182    -0.282      -0.107       0.033     -0.086
delta_d2       -0.445     1.000      -0.151        0.166    -0.259      -0.472       0.076     -0.036
delta_elst     -0.168    -0.151       1.000       -0.911     0.764       0.137      -0.252     -0.171
delta_pauli     0.182     0.166      -0.911        1.000    -0.814      -0.184       0.035      0.146
delta_oi       -0.282    -0.259       0.764       -0.814     1.000       0.275      -0.362      0.003
delta_disp     -0.107    -0.472       0.137       -0.184     0.275       1.000      -0.240      0.423
delta_cpcm      0.033     0.076      -0.252        0.035    -0.362      -0.240       1.000     -0.160
delta_cds      -0.086    -0.036      -0.171        0.146     0.003       0.423      -0.160      1.000
```

## Channel coverage (labels_v1 vs 3504 cohort)
```
channel    column  n_present  n_total  coverage
     d1    d1_own       3504     3504       1.0
     d2    d2_own       3504     3504       1.0
   elst  elst_dft       3504     3504       1.0
  pauli pauli_dft       3504     3504       1.0
     oi    oi_dft       3504     3504       1.0
   disp  disp_dft       3504     3504       1.0
   cpcm  cpcm_dft       3504     3504       1.0
    cds   cds_dft       3504     3504       1.0
```

## Substituent inventory (top 20)
```
    sub_canon     n known_name
         *[H] 71645          H
           *C 13418        CH3
          *OC  2927        OMe
         *C#N  2280         CN
     *C(=O)OC  2055      CO2Me
      *C(C)=O  2047    C(=O)Me
    *c1ccccc1  2008         Ph
          *NC  1968        NaN
     *C(=O)NC  1901  C(=O)NHMe
           *F  1371          F
      *C(=C)C   468        NaN
     *C=C(C)C   468        NaN
          *Cl   397         Cl
          *Br   382         Br
           *N   325        NaN
      *C(N)=O   325        NaN
    *C(F)(F)F   302        CF3
         *C=O   167        NaN
*[C-](C#N)C#N   113        NaN
    *[C-](C)C   103        NaN
```

## Outlier summary
- Outliers (|Δ G_act| > 15 kcal/mol): **360**
```
 rxn_id_A  rxn_id_B                                                                       sub_A                                                             sub_B  delta_G_act
     5723      5215 *[N+](=C(C(=O)N([H])C([H])([H])[H])C(=O)N([H])C([H])([H])[H])C([H])([H])[H]                             *[O+]=C(C#N)C(=O)N([H])C([H])([H])[H]   -37.020913
     4848      4722                                         *[C-](C([H])([H])[H])C([H])([H])[H]                                                     *[C-]([H])C#N   -36.398997
     5723      4991 *[N+](=C(C(=O)N([H])C([H])([H])[H])C(=O)N([H])C([H])([H])[H])C([H])([H])[H]                                   *[O+]=C(C#N)C(=O)C([H])([H])[H]   -34.402301
     5749      4565                              *[C-](C(=O)OC([H])([H])[H])C(=O)C([H])([H])[H]                                                          *[N-][H]   -34.033628
     4697      4613                                            *[N+]#CC(=O)N([H])C([H])([H])[H]             *[N+](=C(C([H])([H])[H])C([H])([H])[H])C([H])([H])[H]    33.866237
     4848      4724                                         *[C-](C([H])([H])[H])C([H])([H])[H]                                                     *[C-]([H])C#N   -31.416884
     3810      3782                                                              *N=[O+][N-][H]                                                  *[C-]([H])[N+]#N    30.528547
     5351      5959                                                       *[N+]#CC([H])([H])[H] *[N+]([H])=C(C(=O)OC([H])([H])[H])c1c([H])c([H])c([H])c([H])c1[H]    30.185888
     3810      3781                                                              *N=[O+][N-][H]                                                  *[C-]([H])[N+]#N    30.029722
     5260      4102                                   *[C-](C(=O)OC([H])([H])[H])C([H])([H])[H]                                                     *[C-]([H])[H]   -29.669094
```

## Files
```
artifacts/GATE1_STATUS.txt
artifacts/GATE2_STATUS.txt
artifacts/GATE4_STATUS.txt
artifacts/GATE5_STATUS.txt
artifacts/GATE6_STATUS.txt
artifacts/Q1_ANSWER.txt
artifacts/channel_coverage.csv
artifacts/cohort_join.pkl
artifacts/delta_channel_stats.csv
artifacts/delta_corr_pearson.csv
artifacts/delta_corr_spearman.csv
artifacts/frag_map.pkl
artifacts/fragments.pkl
artifacts/mmp_pairs.pkl
artifacts/mmp_pairs_labeled.pkl
artifacts/outliers.csv
artifacts/run.895949.log
artifacts/substituent_inventory.csv
artifacts/unknown_subs.csv
figures/delta_corr_heatmap.png
figures/delta_distributions.png
```