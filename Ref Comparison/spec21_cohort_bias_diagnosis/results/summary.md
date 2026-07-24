# spec21_cohort_bias_diagnosis — summary

Env: python 3.10.14, pandas 2.3.3.

**Diagnostic scope:** no compute, no re-labelling. Positions the dipolar-400 within Stuyver's 5269, classifies scaffolds by topology (RDKit), and compares our TS geometries to Stuyver's originals.

## D1 — Reactivity position (ΔG‡, ΔG_r)

Stuyver's Gibbs energies used only as coordinates to locate our reactions. No comparison to our own electronic labels intended.

### G_act

| group | n | mean | sd | min | q05 | q25 | med | q75 | q95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_5269 | 5269 | 21.040 | 9.763 | 0.508 | 7.515 | 13.960 | 19.807 | 26.842 | 38.276 | 75.798 |
| ours_400 | 400 | 19.812 | 9.250 | 1.021 | 7.616 | 12.905 | 17.573 | 25.664 | 36.933 | 48.094 |
| locked_192 | 192 | 19.887 | 9.492 | 1.021 | 7.317 | 12.879 | 17.407 | 25.664 | 37.904 | 48.094 |
| spec16_208 | 208 | 19.743 | 9.044 | 2.539 | 7.867 | 12.905 | 17.847 | 25.596 | 36.498 | 45.341 |

### G_r

| group | n | mean | sd | min | q05 | q25 | med | q75 | q95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_5269 | 5269 | -26.829 | 21.654 | -98.368 | -63.331 | -41.295 | -27.334 | -11.201 | 6.524 | 67.533 |
| ours_400 | 400 | -28.390 | 21.203 | -82.710 | -62.670 | -43.599 | -29.415 | -13.354 | 6.027 | 32.408 |
| locked_192 | 192 | -29.055 | 22.823 | -80.323 | -65.693 | -44.836 | -30.303 | -11.734 | 5.683 | 32.408 |
| spec16_208 | 208 | -27.777 | 19.624 | -82.710 | -58.812 | -41.300 | -28.977 | -14.748 | 5.626 | 23.976 |

### KS two-sample

| target | a | b | n_a | n_b | KS stat | p |
|---|---|---|---:|---:|---:|---:|
| G_act | ours_400 | full_5269 | 400 | 5269 | 0.095 | 2.211e-03 |
| G_act | locked_192 | spec16_208 | 192 | 208 | 0.036 | 9.987e-01 |
| G_r | ours_400 | full_5269 | 400 | 5269 | 0.054 | 2.190e-01 |
| G_r | locked_192 | spec16_208 | 192 | 208 | 0.090 | 3.700e-01 |

Density overlay: `figures/D1_reactivity_position.png`.

## D2 — Scaffold composition (RDKit topology)

Dipolarophile classification by reacting-bond topology.

| class | full_5269 | ours_400 | locked_192 | spec16_208 |
|---|---:|---:|---:|---:|
| alkyne_in_ring | 0.159 [0.150,0.170] | 0.165 [0.132,0.205] | 0.151 [0.107,0.209] | 0.178 [0.132,0.236] |
| bridged_alkene | 0.274 [0.262,0.286] | 0.253 [0.212,0.297] | 0.250 [0.194,0.316] | 0.255 [0.200,0.318] |
| other_cyclic_alkene | 0.146 [0.137,0.156] | 0.107 [0.081,0.142] | 0.115 [0.077,0.167] | 0.101 [0.067,0.149] |
| acyclic_alkene | 0.358 [0.345,0.371] | 0.405 [0.358,0.454] | 0.391 [0.324,0.461] | 0.418 [0.353,0.486] |
| acyclic_alkyne | 0.062 [0.056,0.069] | 0.070 [0.049,0.099] | 0.094 [0.060,0.143] | 0.048 [0.026,0.086] |
| unresolved | 0.000 [0.000,0.001] | 0.000 [0.000,0.010] | 0.000 [0.000,0.020] | 0.000 [0.000,0.018] |

Bar chart: `figures/D2_scaffold_composition.png`.

**G21-C classifier validation:** `results/D2_spotcheck.csv` written (20 rows, 10 per half, seed 42). **Unreviewed — D2 fractions above are PROVISIONAL until the spotcheck is signed off.**

## D3 — TS geometry provenance vs. Stuyver (G21-B)

n_compared = 400/400, n_missing = 0, n_atom_mismatch = 0.

### Verdict distribution

| sub_source   |   identical_lineage |   same_structure_precision_diff |
|:-------------|--------------------:|--------------------------------:|
| locked_778   |                  85 |                             107 |
| spec16       |                 208 |                               0 |

### RMSD distribution (heavy-atom Kabsch, Å)

| half | n | median | mean | q95 | max |
|---|---:|---:|---:|---:|---:|
| locked_778 | 192 | 0.0103 | 0.0103 | 0.0127 | 0.0137 |
| spec16 | 208 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Histogram: `figures/D3_rmsd_by_half.png`.

**G21-B PASS** — halves in the same RMSD regime.

## Files

```
Ref Comparison/spec21_cohort_bias_diagnosis/
  code/{join_cohort.py, d1_reactivity_position.py, d2_scaffold_composition.py,
        d3_geometry_provenance.py, aggregate.py, submit_s21.sh}
  logs/{gates.log, G21_B_HALT.flag OR G21_B_PASS.flag}
  results/{cohort_joined.parquet, stuyver_full.parquet,
           D1_reactivity_stats.csv, D1_ks_tests.csv,
           D2_scaffold_fractions.csv, D2_per_reaction.csv, D2_spotcheck.csv,
           D3_geometry_provenance.csv,
           spec22_cohort_recommendation.md, summary.md}
  figures/{D1_reactivity_position.png, D2_scaffold_composition.png,
           D3_rmsd_by_half.png}
```

