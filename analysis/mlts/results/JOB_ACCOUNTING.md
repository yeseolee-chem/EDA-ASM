# Job accounting — yeseo1ee, 2026-11-04T00:00 → 2026-11-05 21:25

sacct -X --array (allocations + array elements; running jobs counted up to now; pending jobs count 0 h). Rows: 388. core-h = ΣCPUTimeRAW/3600; GPU-h = Σ ElapsedRaw × gres/gpu / 3600.

Classification is by job name (patterns in `job_accounting.py`); `mltp_<stage>` jobs other than `mltp_launch` count as 본 계산.

## By class

| class | jobs | core-h | GPU-h | states |
|---|---:|---:|---:|---|
| 본 계산 | 135 | 9,187.3 | 76.0 | COMPLETED 108, CANCELLED 20, FAILED 7 |
| 검사 · 시험 · dry-run · 시뮬레이터 | 65 | 394.8 | 10.7 | COMPLETED 53, FAILED 6, CANCELLED 4, RUNNING 2 |
| launcher | 117 | 0.1 | 0.0 | COMPLETED 117 |
| unclassified | 71 | 30.9 | 4.2 | COMPLETED 59, CANCELLED 6, FAILED 3, RUNNING 3 |
| **total** | **388** | **9,613.1** | **90.9** | |

## 본 계산 — top 15 name prefixes (by core-h + GPU-h)

| prefix | jobs | core-h | GPU-h |
|---|---:|---:|---:|
| `g_run2` | 3 | 1,344.3 | 0.0 |
| `g_run_d0g2` | 3 | 1,286.6 | 0.0 |
| `g_run1` | 3 | 1,277.0 | 0.0 |
| `g_run_d0g2rev` | 6 | 922.5 | 0.0 |
| `g_run_d0g1` | 2 | 711.2 | 0.0 |
| `g_run3` | 1 | 387.4 | 0.0 |
| `g_run1rev` | 1 | 381.8 | 0.0 |
| `g_run_d0g1rev` | 1 | 328.8 | 0.0 |
| `g_run_d0g1shuf2` | 1 | 321.2 | 0.0 |
| `g_run_d0g1shuf1` | 1 | 321.1 | 0.0 |
| `g_run_d0g1shuf3` | 1 | 320.1 | 0.0 |
| `g_run_d0g3` | 1 | 237.3 | 0.0 |
| `ft_spec_AIMNet2` | 8 | 199.4 | 24.9 |
| `g_run_d0g3rev` | 1 | 205.6 | 0.0 |
| `g_run_d0g3shuf1` | 1 | 197.6 | 0.0 |
| (other 37 prefixes) | 101 | 745.4 | 51.0 |

## 검사 · 시험 · dry-run · 시뮬레이터 — top 15 name prefixes (by core-h + GPU-h)

| prefix | jobs | core-h | GPU-h |
|---|---:|---:|---:|
| `mlt_gsandbox` | 1 | 167.0 | 10.4 |
| `dep_sandbox` | 1 | 151.3 | 0.0 |
| `mlt_sandbox` | 1 | 39.4 | 0.0 |
| `mlt_audit_orca` | 1 | 13.4 | 0.0 |
| `mlt_test_h1` | 1 | 12.6 | 0.0 |
| `mlt_test_c6` | 1 | 2.9 | 0.0 |
| `mlt_test_c1` | 1 | 2.1 | 0.3 |
| `mlt_audit_leak2` | 2 | 1.2 | 0.0 |
| `mlt_test_ft` | 2 | 1.2 | 0.0 |
| `mlt_test_integ` | 1 | 1.1 | 0.0 |
| `mc_smoke_h` | 1 | 0.8 | 0.0 |
| `mlt_test_ezleak` | 1 | 0.6 | 0.0 |
| `mlt_quota` | 1 | 0.3 | 0.0 |
| `mlt_audit_leak` | 1 | 0.3 | 0.0 |
| `mlt_audit_gpu` | 1 | 0.2 | 0.0 |
| (other 16 prefixes) | 48 | 0.7 | 0.0 |

## launcher — top 15 name prefixes (by core-h + GPU-h)

| prefix | jobs | core-h | GPU-h |
|---|---:|---:|---:|
| `mltp_launch` | 117 | 0.1 | 0.0 |

## unclassified — top 15 name prefixes (by core-h + GPU-h)

| prefix | jobs | core-h | GPU-h |
|---|---:|---:|---:|
| `dw_K` | 5 | 10.5 | 2.6 |
| `mc_h1` | 1 | 6.5 | 0.0 |
| `dep_null` | 10 | 3.3 | 0.8 |
| `mlt_env` | 3 | 3.4 | 0.4 |
| `i3_g2` | 1 | 3.4 | 0.0 |
| `mc_c6_xgb` | 1 | 1.2 | 0.0 |
| `dep_w` | 1 | 0.5 | 0.1 |
| `dw_I2` | 1 | 0.3 | 0.1 |
| `mlt_du` | 1 | 0.3 | 0.0 |
| `dw_I2b` | 1 | 0.3 | 0.1 |
| `mc_c6_atoms` | 1 | 0.3 | 0.0 |
| `dw_P` | 15 | 0.2 | 0.0 |
| `mc_peek` | 3 | 0.2 | 0.0 |
| `d0p_plan` | 1 | 0.1 | 0.0 |
| `mlt_ezall` | 1 | 0.1 | 0.0 |
| (other 22 prefixes) | 25 | 0.4 | 0.0 |

## Unclassified job names (for the lead to reclassify)

`d0p_plan` ×1, `dep_i12` ×1, `dep_null` ×10, `dep_nullgc` ×1, `dep_w` ×1, `dw_D` ×1, `dw_E` ×1, `dw_I2` ×1, `dw_I2b` ×1, `dw_K` ×5, `dw_Kc` ×1, `dw_P` ×15, `dw_W2` ×1, `dw_apply` ×1, `dw_i3view` ×1, `dw_patch` ×2, `dw_prereg` ×1, `i3_ft` ×1, `i3_g2` ×1, `i3_lm` ×1, `i3_pred` ×1, `i3_prep` ×1, `i3_rep` ×1, `mc_c6` ×1, `mc_c6_atoms` ×1, `mc_c6_xgb` ×1, `mc_cmds` ×2, `mc_gaps` ×2, `mc_h1` ×1, `mc_peek` ×3, `mlt_delete_app` ×1, `mlt_du` ×1, `mlt_env` ×3, `mlt_ezall` ×1, `mlt_mkdir` ×1, `refchkA` ×1, `refchkB` ×1
