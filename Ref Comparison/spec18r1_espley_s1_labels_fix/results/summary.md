# spec18r1_espley_s1_labels_fix — Stage 1 summary (DIPOLAR-400)

Revision 1 of `spec18_espley_s1_labels`. Fixes Rev 0's parquet serialisation bug (F1), replaces three tautological gates with real ones (F3–F5), and re-arms the ASM identity tripwire on the four-channel sum (F5).

## Environment

- python: 3.10.14
- pandas: 2.3.3
- numpy:  1.26.4

Espley's own pickles were written with `pandas==2.1.1`; if the downstream env pins that version, this artifact will need to be re-pickled from a matching env (see `logs/build.log` for the version stamp at build time).

## Artifact of record

`results/labels_2ch_400dipolar.pkl` (pickle; size 41509 bytes)

Inspection parquet with the dict column dropped:
`results/labels_2ch_400dipolar.INSPECTION_ONLY.parquet` (size 23494 bytes) — **NOT the artifact of record.** Do not pass to their scripts.

## Cohort composition

| sub_source | n |
|---|---:|
| locked_778 | 192 |
| spec16 | 208 |
| **total** | **400** |

Source parquet: `outputs/spec16_orca/labels/dipolar_400_merged.parquet`.

## Gates (all run on the reloaded pickle)

- **0 FAIL**, **1 WARN** — see `logs/gates.log`.

- G-A round-trip fidelity — all 400 rows carry a 2-key dict, no NaN, keys match `^\d+_[12]$`, per-row prefix equals the row's `reaction_number`, `sum(dict) − sum_distortion` within 5e-3 kcal/mol.
- G-B file sanity — artifact suffix `.pkl`; inspection parquet has no `distortion_contributions_dft` column.
- G-C independent interaction — `|interaction_dft + resum(channels)|` from the four **source** channels; independent from the identity used to build the column.
- G-D ASM identity residual — `|strain + resum − act_kcal|`. Rev 0 substituted `int_eda_kcal` for `resum`, which reduced this to 0 by construction. Rev 1 re-derives from the four channels.
- G-E end-to-end contract — imports Espley's own `General._clean_dist_contr` and `Manual._manual_runner` and runs them on the reloaded artifact.
- G-F reaction_number stability — sorting by `reaction_id` reproduces `reaction_number == 0..399`; two rebuilds hash-compared.
- G-G anomaly triage — count-and-cross-tab flagged rows against `sub_source` and G-D residual. **No rows excluded.**
- Regression guards (labelled) — the Rev 0 sign / algebraic-identity checks are kept as future-edit guards. They are not evidence.

### Gate digest
```
    [G-A PASS] n=400, dict-shape errors=0, None values=0, key-format errors=0, prefix mismatches=0, max|sum(dict) − sum_distortion|=1.000e-03 (tol 5e-03)
    [G-A-dtypes PASS] dtypes={'reaction_number': 'int32', 'e_barrier_dft': 'float64', 'sum_distortion_energies_dft': 'float64', 'interaction_energies_dft': 'float64', 'distortion_contributions_dft': 'object'}
    [G-B INFO] artifact of record: labels_2ch_400dipolar.pkl size=41509 bytes
    [G-B-artifact PASS] suffix '.pkl' and file exists
    [G-B-inspection PASS] inspection parquet dict-column-free, size=23494 bytes
    [G-C PASS] |interaction_dft + resum| max=2.000e-02 mean=5.225e-03 p99=2.000e-02
    [G-D-distribution INFO] n=400.0000, min=0.0000, median=0.0000, mean=0.0052, p95=0.0100, p99=0.0200, max=0.0200
    [G-D-locked_778 INFO] n=192.0000, median=0.0100, mean=0.0057, p95=0.0100, max=0.0200
    [G-D-spec16 INFO] n=208.0000, median=0.0000, mean=0.0048, p95=0.0100, max=0.0200
    [G-D-outliers INFO] n_outliers_above_0.1kcal=0 written to asm_residual_outliers.csv
    [G-D PASS] max residual 0.0200 < 0.05 — source rounding floor confirmed
    [G-E-clean-rxnum PASS] reaction_number == 0..399 on expanded frame
    [G-E-clean-sum PASS] (distortion_energy_1_dft+distortion_energy_2_dft) matches sum_distortion within 5e-03 (max diff 1.000e-03)
    [G-E-runner PASS] all dft targets survive _manual_runner (4 cols kept)
    [G-E-deviation6 INFO] _gfn2 columns dropped as predicted (['e_barrier_gfn2', 'interaction_energies_gfn2']) — Deviation #6 (f_select.py line 226 patch) confirmed necessary at Stage 3
    [G-F-sort PASS] reaction_number == 0..399 under sort by reaction_id
    [G-F-rebuild PASS] pickle sha256 stable across rebuilds: 30b61b47c8092d7b…
    [G-G INFO] flagged rows total: 174  (anomaly_triage.csv)
    [G-G-interaction_lt_0 INFO] n=8 by sub_source={'locked_778': 5, 'spec16': 3}  (with G-D residual>0.05: 0)
    [G-G-e_barrier_lt_0 INFO] n=138 by sub_source={'locked_778': 64, 'spec16': 74}  (with G-D residual>0.05: 0)
    [G-G-e_barrier_lt_neg20 INFO] n=7 by sub_source={'locked_778': 3, 'spec16': 4}  (with G-D residual>0.05: 0)
    [G-G-sum_distortion_gt_100 INFO] n=29 by sub_source={'locked_778': 7, 'spec16': 22}  (with G-D residual>0.05: 0)
    [REG-sd-positive PASS] sum_distortion > 0 in all rows
    [REG-identity-tautology INFO] max|e_barrier − (sum_dist − int)|=0.000e+00  (guard, not evidence — algebraic identity by construction)
    [REG-interaction-positive-fraction INFO] interaction > 0 in 392/400 rows (0.9800)  (physical, not tautological)
    === SUMMARY: 0 FAIL, 0 WARN ===
```

## G-G anomaly triage

- 174 rows carry at least one anomaly flag (see `results/anomaly_triage.csv`).
- 0 rows exceed the G-D 0.1 kcal/mol residual (see `results/asm_residual_outliers.csv`).
- **No exclusions applied.** Any decision on removing rows belongs to the user.

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

## Distribution vs. Espley ds3 (n=3510)

Both cohorts are dipolar [3+2] cycloadditions. Systematic ~1.6–1.9× scale in `sum_distortion` and `interaction` while `e_barrier` is similar in magnitude — the two large terms cancel. Candidate explanations (not yet distinguished): reference DFT level (Deviation #4), broader coverage of distorted TSs, or pathological rows. **G-D separates 'pathological rows' from the other two.** Do not conclude — surface.

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

## Downstream contracts

- **Do not rename `_dft` → `_wb97x3c`.** `f_select.py` line ~226 selects by the literal substring `dft`.
- **`q_barrier_dft` is intentionally absent** (Deviation #3). Stage 4 will emit 44 features vs their 45.
- **`_gfn2` will be dropped** by the unpatched `_manual_runner` — confirmed in Rev 1 G-E. Either patch the line-226 substring to `am1|gfn2`, or rename Stage-3 SQM columns to `_am1` at write.
- **Stage 4 hazard: `_clean_dist_contr` overwrite.** The function loops every `contributions` column and emits `'1'` / `'2'` (renamed to `distortion_energy_{1,2}_{method}`). When both `distortion_contributions_am1` and `distortion_contributions_dft` are present at Stage 4, the second overwrites the first on columns named identically. Prefix or method-suffix must be preserved by construction.

## Files

```
Ref Comparison/spec18r1_espley_s1_labels_fix/
  code/{build_2ch_labels.py, verify_artifact.py, aggregate.py, submit_s1.sh}
  data/cohort_notes.json
  logs/{build.log, gates.log}
  results/{labels_2ch_400dipolar.pkl,
           labels_2ch_400dipolar.INSPECTION_ONLY.parquet,
           anomaly_triage.csv, asm_residual_outliers.csv,
           sub_source_stats.csv, ds3_distribution_comparison.csv,
           DEVIATIONS.md, summary.md}
  figures/{target_hist_3panel.png, sub_source_box.png,
           asm_residual_hist.png}
```

