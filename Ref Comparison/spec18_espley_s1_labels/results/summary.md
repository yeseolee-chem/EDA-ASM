# spec18_espley_s1_labels — Stage 1 summary (DIPOLAR-400)

Recasts the 400-reaction dipolar [3+2]-cycloaddition set into Espley et al.'s 2-channel DIAS schema (Digital Discovery 2024, DOI 10.1039/d4dd00224e). Because both this cohort and Espley's ds3 are [3+2] cycloadditions, the ds3 distribution anchors below are for once directly comparable modulo the reference DFT level (Deviation #4).

## Output of record

`results/labels_2ch_400dipolar.parquet` — 400 rows, columns:

- `reaction_number` (int32, contiguous 0..399)
- `sum_distortion_energies_dft` (float64, kcal/mol, positive)
- `interaction_energies_dft` (float64, kcal/mol, positive — sign flip vs. our E_int)
- `e_barrier_dft` (float64, kcal/mol; = sum_distortion − interaction)
- `distortion_contributions_dft` (object, dict `{rxn_1: strain_A, rxn_2: strain_B}`)
- `family`, `reaction_id`, `act_kcal_source`, `sub_source` — provenance columns

## Cohort composition

400 dipolar [3+2] cycloadditions:

| sub_source | n |
|---|---:|
| locked_778 | 192 |
| spec16 | 208 |
| **total** | **400** |

Source parquet: `outputs/spec16_orca/labels/dipolar_400_merged.parquet`.
Provenance in `data/cohort_notes.json`.

## Target statistics (kcal/mol)

### e_barrier_dft

| sub_source | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| locked_778 | 192 | 3.377 | 11.540 | -51.889 | 61.227 |
| spec16 | 208 | 4.331 | 12.563 | -39.476 | 68.397 |
| ALL | 400 | 3.873 | 12.077 | -51.889 | 68.397 |

### sum_distortion_energies_dft

| sub_source | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| locked_778 | 192 | 39.503 | 32.654 | 0.623 | 123.416 |
| spec16 | 208 | 49.270 | 36.371 | 3.248 | 150.587 |
| ALL | 400 | 44.582 | 34.937 | 0.623 | 150.587 |

### interaction_energies_dft

| sub_source | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| locked_778 | 192 | 36.127 | 32.146 | -0.990 | 114.900 |
| spec16 | 208 | 44.939 | 35.876 | -3.940 | 123.260 |
| ALL | 400 | 40.709 | 34.378 | -3.940 | 123.260 |

## Distribution comparison vs. Espley ds3 (n=3510)

Their ds3 raw pickle is not available on this HPC; the ds3 columns below are the literal statistics recorded in the spec (§6 item 3). Both cohorts are dipolar [3+2] cycloadditions; the remaining offset is the reference DFT level (Deviation #4).

| target | stat | ours (n=400) | ds3 ref (n=3510) |
|---|---|---:|---:|
| e_barrier_dft | n | 400.0000 | 3510.0000 |
| e_barrier_dft | mean | 3.8730 | 5.9200 |
| e_barrier_dft | std | 12.0772 | 8.4900 |
| e_barrier_dft | min | -51.8887 | -14.8100 |
| e_barrier_dft | max | 68.3970 | 44.6500 |
| sum_distortion_energies_dft | n | 400.0000 | 3510.0000 |
| sum_distortion_energies_dft | mean | 44.5820 | 27.1300 |
| sum_distortion_energies_dft | std | 34.9365 |  |
| sum_distortion_energies_dft | min | 0.6234 | 2.4700 |
| sum_distortion_energies_dft | max | 150.5870 | 79.1500 |
| interaction_energies_dft | n | 400.0000 | 3510.0000 |
| interaction_energies_dft | mean | 40.7091 | 21.2100 |
| interaction_energies_dft | std | 34.3780 |  |
| interaction_energies_dft | min | -3.9400 | 5.8100 |
| interaction_energies_dft | max | 123.2600 | 49.3600 |

## Gates (verification)

All six correctness gates pass; see `logs/gates.log`.

- Gate #1: cohort n = 400
- Gate #2a: interaction > 0 in ≥ 95% of rows
- Gate #2b: sum_distortion > 0 in all rows
- Gate #2c: `sign(interaction_dft) == -sign(source int_eda)` for every row
- Gate #3: `e_barrier_dft == sum_distortion − interaction` (max abs diff < 1e-6)
- Gate #4: `|e_barrier_dft − act_kcal_source| < 0.1 kcal/mol` for all rows
- Gate #5: target column names include the substring `dft` (required by `f_select.py::Manual._manual_runner`)
- Gate #6: dtypes reaction_number=int32, energies=float64, dict=object

## Downstream contracts

- **Do not rename `_dft` → `_wb97x3c`.** `f_select.py` line ~226 keeps features by literal substring `dft`; a rename silently drops every target column at feature selection.
- **`q_barrier_dft` is intentionally absent** (Deviation #3). Stage 4 will emit 44 features vs. their 45.
- **Fragment ordering.** Key `<rxn>_1` = fragment A in the source parquet (`strain_A_kcal`); key `_2` = fragment B. Chemical role (dipole / dipolarophile) is *not* asserted here — it will be resolved at Stage 4 when common-atom masks are defined.

## Files

```
Ref Comparison/spec18_espley_s1_labels/
  code/{build_2ch_labels.py, compare_to_ds3.py, aggregate.py, submit_s1.sh}
  data/cohort_notes.json
  logs/{build.log, gates.log}
  results/{labels_2ch_400dipolar.parquet, sub_source_stats.csv,
           ds3_distribution_comparison.csv, DEVIATIONS.md, summary.md}
  figures/{target_hist_3panel.png, sub_source_box.png}
```

