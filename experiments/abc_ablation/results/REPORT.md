# A/B/C Baseline Ablation — REPORT

**Data.** in-distribution 787 reactions (dipolar / qmrxn20_e2 / qmrxn20_sn2 / rgd1). Descriptor set = m3 (24-d).
**Split.** 5-fold reaction-level, family-stratified, seed=42, materialised in `splits/outer_folds.json`.
**Anti-leakage.** δ trained on residuals from an *inner K′=5 cross-fit* OOF baseline; the outer held-out uses `b_full` (re-fit on the whole outer-train).

## Pooled OOF metrics (95% bootstrap CI, B_boot = 1000)

### NMAE

| channel | A · xgb_direct | B · ridge+δ | C · xgb+δ |
|---|---|---|---|
| strain | 0.780 [0.720, 0.848] | 0.764 [0.705, 0.829] | 0.781 [0.721, 0.848] |
| Pauli | 0.638 [0.600, 0.681] | 0.741 [0.690, 0.794] | 0.599 [0.555, 0.641] |
| V_elst | 0.668 [0.631, 0.710] | 0.750 [0.701, 0.797] | 0.641 [0.600, 0.680] |
| oi | 0.681 [0.632, 0.734] | 0.766 [0.711, 0.828] | 0.681 [0.630, 0.737] |
| disp | 0.243 [0.226, 0.262] | 0.287 [0.270, 0.304] | 0.243 [0.226, 0.262] |
| barrier | 0.644 [0.605, 0.683] | 0.571 [0.534, 0.607] | 0.677 [0.638, 0.718] |

### RMSE

| channel | A · xgb_direct | B · ridge+δ | C · xgb+δ |
|---|---|---|---|
| strain | 26.139 [21.248, 32.079] | 24.960 [21.150, 29.806] | 26.550 [21.302, 32.872] |
| Pauli | 89.124 [83.303, 95.542] | 105.503 [98.455, 112.943] | 87.703 [81.109, 94.691] |
| V_elst | 36.547 [34.443, 38.808] | 43.099 [40.257, 45.627] | 36.340 [33.975, 38.544] |
| oi | 77.513 [70.917, 84.185] | 85.039 [78.315, 92.162] | 78.590 [71.926, 85.803] |
| disp | 1.912 [1.709, 2.112] | 1.998 [1.851, 2.152] | 1.920 [1.717, 2.118] |
| barrier | 28.442 [26.508, 30.328] | 24.998 [23.386, 26.658] | 29.209 [27.432, 31.077] |

### R2

| channel | A · xgb_direct | B · ridge+δ | C · xgb+δ |
|---|---|---|---|
| strain | 0.228 [0.129, 0.350] | 0.296 [0.226, 0.374] | 0.204 [0.094, 0.347] |
| Pauli | 0.500 [0.420, 0.559] | 0.299 [0.223, 0.365] | 0.516 [0.430, 0.581] |
| V_elst | 0.509 [0.445, 0.565] | 0.317 [0.234, 0.391] | 0.515 [0.442, 0.585] |
| oi | 0.385 [0.304, 0.451] | 0.260 [0.195, 0.322] | 0.368 [0.280, 0.441] |
| disp | 0.921 [0.904, 0.936] | 0.913 [0.898, 0.926] | 0.920 [0.903, 0.935] |
| barrier | 0.557 [0.493, 0.612] | 0.658 [0.617, 0.693] | 0.532 [0.471, 0.589] |

### slope

| channel | A · xgb_direct | B · ridge+δ | C · xgb+δ |
|---|---|---|---|
| strain | 0.315 [0.216, 0.440] | 0.372 [0.294, 0.464] | 0.399 [0.275, 0.555] |
| Pauli | 0.513 [0.461, 0.554] | 0.402 [0.358, 0.452] | 0.635 [0.581, 0.687] |
| V_elst | 0.509 [0.464, 0.546] | 0.475 [0.426, 0.530] | 0.700 [0.651, 0.746] |
| oi | 0.425 [0.372, 0.476] | 0.346 [0.302, 0.394] | 0.515 [0.454, 0.575] |
| disp | 0.916 [0.888, 0.945] | 0.917 [0.893, 0.940] | 0.920 [0.891, 0.950] |
| barrier | 0.754 [0.704, 0.807] | 0.686 [0.651, 0.722] | 0.757 [0.705, 0.812] |

## ΔNMAE(B − C) — paired bootstrap CI + Wilcoxon

| channel | ΔNMAE(B−C) | 95% CI | Wilcoxon p | verdict |
|---|---|---|---|---|
| strain | -0.017 | [-0.058, +0.023] | 0.44 | indistinguishable |
| Pauli | +0.142 | [+0.091, +0.189] | 5.2e-08 | B > C (C better) |
| V_elst | +0.109 | [+0.059, +0.164] | 0.00095 | B > C (C better) |
| oi | +0.085 | [+0.039, +0.133] | 0.00039 | B > C (C better) |
| disp | +0.044 | [+0.028, +0.058] | 1.4e-12 | B > C (C better) |
| barrier | -0.106 | [-0.146, -0.067] | 1.7e-06 | B < C (B better) |

## Sanity gates (SPEC §8)

**#1 same fold index across arms** — ✅  
outer_folds.json = 5 folds, coverage=787 unique test rids

**#2 no reaction-level leakage** — ✅  
no overlap

**#3 δ target ≠ 0 (OOF baseline)** — ✅

```
  B/fold0.json: median|r_train|/median|y_train| = 0.392
  B/fold1.json: median|r_train|/median|y_train| = 0.426
  B/fold2.json: median|r_train|/median|y_train| = 0.422
  B/fold3.json: median|r_train|/median|y_train| = 0.428
  B/fold4.json: median|r_train|/median|y_train| = 0.439
  C/fold0.json: median|r_train|/median|y_train| = 0.278
  C/fold1.json: median|r_train|/median|y_train| = 0.321
  C/fold2.json: median|r_train|/median|y_train| = 0.299
  C/fold3.json: median|r_train|/median|y_train| = 0.303
  C/fold4.json: median|r_train|/median|y_train| = 0.299
```

**#4 smoke test (B / A vs SPEC targets, tol ±0.05)** — 6/11 outside tol

| arm | channel | NMAE | expected | Δ | in-tol? |
|---|---|---|---|---|---|
| B | strain | 0.764 | 0.66 | +0.104 | ⚠️ |
| B | Pauli | 0.741 | 0.62 | +0.121 | ⚠️ |
| B | V_elst | 0.750 | 0.62 | +0.130 | ⚠️ |
| B | oi | 0.766 | 0.61 | +0.156 | ⚠️ |
| B | disp | 0.287 | 0.22 | +0.067 | ⚠️ |
| B | barrier | 0.571 | 0.43 | +0.141 | ⚠️ |
| A | strain | 0.780 | 0.76 | +0.020 | ✅ |
| A | Pauli | 0.638 | 0.65 | -0.012 | ✅ |
| A | V_elst | 0.668 | 0.68 | -0.012 | ✅ |
| A | oi | 0.681 | 0.67 | +0.011 | ✅ |
| A | disp | 0.243 | 0.25 | -0.007 | ✅ |

**#5 ΔNMAE(B−C) verdict (barrier)** — **B < C (B better)**

| channel | ΔNMAE(B−C) | 95% CI | Wilcoxon p | verdict |
|---|---|---|---|---|
| strain | -0.017 | [-0.058, +0.023] | 0.44 | indistinguishable |
| Pauli | +0.142 | [+0.091, +0.189] | 5.2e-08 | B > C (C better) |
| V_elst | +0.109 | [+0.059, +0.164] | 0.00095 | B > C (C better) |
| oi | +0.085 | [+0.039, +0.133] | 0.00039 | B > C (C better) |
| disp | +0.044 | [+0.028, +0.058] | 1.4e-12 | B > C (C better) |
| barrier | -0.106 | [-0.146, -0.067] | 1.7e-06 | B < C (B better) |

## Verdict

**B (ridge+δ) beats C (xgb+δ)** on the barrier at the 95% CI. Keep `ridge` as the default baseline.
