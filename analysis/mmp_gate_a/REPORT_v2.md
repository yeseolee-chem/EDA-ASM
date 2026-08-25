# MMP Gate A — REPORT v2 (spec10rev patch)

_Generated: 2026-08-25T00:04:33.366544+00:00, commit `6d5bc0ccb312`_

This v2 supersedes REPORT.md. v1 artifacts are preserved for comparison.

## Patch gate summary (spec10rev)
| Gate | Status |
|---|---|
| GATE-0 (cohort join, composition) | ✅ PASS (from 00_load_join.py) |
| GATE-1a (core detection, 5-ring) | ✅ PASS 3504/3504 5-membered cores |
| GATE-1 (library hit — report-only) | ℹ️ REPORT total_hit_rate=93.56% |
| GATE-1b (core-adjacent hit rate ≥95%) | ✅ PASS core_adjacent_hit_rate=96.71% n=104161 |
| GATE-2 (eda.inp fragment cross-check) | ✅ PASS unique_rate=100.00% ambig=0 mismatch=0 |
| GATE-3-rev (pair count 2207±5%) | ✅ PASS n_pairs=2207 expected=2207±5% [2096,2317] |
| GATE-4-rev (8-channel MMP pair count) | ✅ PASS n_all8=2207 |
| GATE-5-rev (learned-channel δ) | ⚠️ CONDITIONAL learned_channels=['d1', 'd2', 'elst', 'pauli', 'oi', 'disp', 'cpcm'] excluded=['cds'] |
| GATE-6-rev (outlier + filter integrity) | ✅ PASS frac_outliers=0.025 |

## Q1 — 8-channel-complete MMP pair count (v2)
```
Q1 (v2): 8-channel-complete MMP pairs = 2207
Total pairs: 2207
  d1: 2207
  d2: 2207
  elst: 2207
  pauli: 2207
  oi: 2207
  disp: 2207
  cpcm: 2207
  cds: 2207
```

## Q2 — per-channel δ distribution (v2)
`cds` is **excluded from learning** — see §CDS-exclusion below.
```
channel  excluded_from_learning    n  mean_abs  median_abs  median_abs_ci_lo  median_abs_ci_hi  std_signed  frac_lt_1.0  baseline_MAE
     d1                   False 2207  6.620920    3.847634          3.547554          4.224875    9.248701     0.184866      6.620920
     d2                   False 2207  6.801107    3.988328          3.776707          4.209141    9.915602     0.191663      6.801107
   elst                   False 2207  6.510081    4.947560          4.604748          5.144193    8.808500     0.117807      6.510081
  pauli                   False 2207 12.298634    8.829905          8.367970          9.424283   17.072698     0.072950     12.298634
     oi                   False 2207  7.557888    5.258461          4.973466          5.490655   10.700981     0.113276      7.557888
   disp                   False 2207  2.463134    1.889333          1.835024          1.991570    3.210924     0.291346      2.463134
   cpcm                   False 2207  5.135783    3.771596          3.627773          3.979530    6.950193     0.138650      5.135783
    cds                    True 2207  0.375526    0.262341          0.248526          0.279858    0.530139     0.932488      0.375526
  G_act                   False 2207  4.507160    3.339401          3.176682          3.472334    6.045580     0.160399      4.507160
```

## Cross-channel δ Pearson correlation (v2, all-8 subset)

> ⚠️ **H5 caution.** This matrix is over **actual δ values**, NOT prediction errors.
> - Actual δ correlation → physics; channels move together; the 'channel-independent
>   manipulation' claim is weakened.
> - Prediction-error correlation → model limitation.
> Do not conflate the two.

```
          d1     d2   elst  pauli     oi   disp   cpcm    cds
d1     1.000 -0.637 -0.042  0.057 -0.181  0.013  0.062 -0.176
d2    -0.637  1.000 -0.169  0.185 -0.275 -0.515  0.096 -0.098
elst  -0.042 -0.169  1.000 -0.865  0.672  0.274 -0.367  0.008
pauli  0.057  0.185 -0.865  1.000 -0.748 -0.292  0.093  0.049
oi    -0.181 -0.275  0.672 -0.748  1.000  0.435 -0.472  0.230
disp   0.013 -0.515  0.274 -0.292  0.435  1.000 -0.323  0.379
cpcm   0.062  0.096 -0.367  0.093 -0.472 -0.323  1.000 -0.309
cds   -0.176 -0.098  0.008  0.049  0.230  0.379 -0.309  1.000
```

## v1 → v2 shape shift (spec10rev §5.4)
| pair | v1 (all-side, no core filter) | v2 (core-preserved) |
|---|---|---|
| pair count | 4230 | see Q1 above |
| elst↔pauli | −0.911 | see matrix |
| elst↔oi    | +0.764 | see matrix |
| d1↔d2      | −0.445 | see matrix |

Direction of change is the diagnostic: attenuation → v1 had confounded pairs; 
intensification → filter isolated a real physical coupling.

## CDS-exclusion rationale (spec10rev §5.2)
- cds δ is <1 kcal/mol on **>93%** of pairs (see stats table)
- A trivial `δ=0` predictor already achieves MAE ≈ 0.4 kcal/mol
- No learnable signal → **removed from learning targets**
- Analogous to treating dispersion via D4 analytical; disp itself is NOT excluded
  (disp median|δ| ≈ 2.25 has real signal)
- LEARNED_CHANNELS = [d1, d2, elst, pauli, oi, disp, cpcm]
- EXCLUDED_CHANNELS = [cds]

## Substituent inventory (v2 top 20 with core-adjacent counts)
```
    sub_canon     n known_name  n_core_adjacent
         *[H] 71645          H          71645.0
           *C 13418        CH3          13418.0
          *OC  2927        OMe           2927.0
         *C#N  2280         CN           2280.0
     *C(=O)OC  2055      CO2Me           2055.0
      *C(C)=O  2047    C(=O)Me           2047.0
    *c1ccccc1  2008         Ph           2008.0
          *NC  1968        NaN           1968.0
     *C(=O)NC  1901  C(=O)NHMe           1901.0
           *F  1371          F           1371.0
      *C(=C)C   468        NaN            349.0
     *C=C(C)C   468        NaN            294.0
          *Cl   397         Cl            397.0
          *Br   382         Br            382.0
           *N   325        NaN            325.0
      *C(N)=O   325        NaN            325.0
    *C(F)(F)F   302        CF3            302.0
         *C=O   167        NaN            167.0
*[C-](C#N)C#N   113        NaN              0.0
    *[C-](C)C   103        NaN              0.0
```

## Outlier summary (v2 with filter-integrity flags)
- Outliers (|Δ G_act| > 15 kcal/mol): **55**
- Reason (a) core mismatch: **0** (must be 0 if filter works)
- Reason (b) sub_heavy>8: **0** (must be 0 if filter works)
```
 rxn_id_A  rxn_id_B                                       sub_A                            sub_B  delta_G_act  reason_a_core_mismatch  reason_b_sub_too_big
     4912      4684                             *C([H])([H])[H]                             *[H]   -26.660195                   False                 False
     5251      4352                       *C(=O)OC([H])([H])[H]                             *[H]   -24.498437                   False                 False
     5495      5628                             *C([H])([H])[H]                 *OC([H])([H])[H]    24.061914                   False                 False
     5495      5628                        *C(=O)C([H])([H])[H]            *C(=O)OC([H])([H])[H]    24.061914                   False                 False
     5554      5042 */C([H])=C(\[H])C(=C([H])[H])C([H])([H])[H]                        *C([H])=O    23.899831                   False                 False
     4912      4683                             *C([H])([H])[H]                             *[H]   -23.116619                   False                 False
     4995      4331 */C([H])=C(\[H])C(=C([H])[H])C([H])([H])[H]                        *C([H])=O   -23.038198                   False                 False
     5251      5465                       *C(=O)OC([H])([H])[H] *c1c([H])c([H])c([H])c([H])c1[H]   -22.468298                   False                 False
     5555      5042 */C([H])=C(\[H])C(=C([H])[H])C([H])([H])[H]                        *C([H])=O    22.095718                   False                 False
     5252      4352                       *C(=O)OC([H])([H])[H]                             *[H]   -21.708733                   False                 False
```

## Deferred (spec10rev §12)
- Partial correlation ctrl. on TS distance — needed to separate
  physical channel coupling from geometric confounder. TS xyz
  already sits in `cohort_v1/reactions/rxn_XXXX/eda.inp`; cost = 0.
- Deferred to SPEC11 or v3 patch.

## Files (v2)
```
artifacts/GATE1_STATUS_v2.txt
artifacts/GATE1a_STATUS.txt
artifacts/GATE1b_STATUS.txt
artifacts/GATE3rev_STATUS.txt
artifacts/GATE4_STATUS_v2.txt
artifacts/GATE5_STATUS_v2.txt
artifacts/GATE6_STATUS_v2.txt
artifacts/Q1_ANSWER_v2.txt
artifacts/core_v2.pkl
artifacts/delta_channel_stats_v2.csv
artifacts/delta_corr_pearson_v2.csv
artifacts/delta_corr_spearman_v2.csv
artifacts/fragments_v2.pkl
artifacts/mmp_pairs_labeled_v2.pkl
artifacts/mmp_pairs_v2.pkl
artifacts/outliers_v2.csv
artifacts/run_v2.895950.log
artifacts/substituent_inventory_v2.csv
artifacts/unknown_subs_v2.csv
figures/delta_corr_heatmap_v2.png
figures/delta_distributions_v2.png
```