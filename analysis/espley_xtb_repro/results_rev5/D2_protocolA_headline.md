# rev 5 D-2 Protocol A — g1 · KRR_rbf · arms ESPLEY46, ESPLEY73, EXT_SEL, EXT_ALL

Protocol A test folds: 80/10/10 over seeds 22/23/14/1/2, nested per-seed GridSearchCV 5-fold, StandardScaler on X, y standardised. n = 4838 rxns (`rows_rev5.csv`, sha256 `a89a3d923d6f`). Arm column counts in parentheses.

## test MAE ± sd over seeds (kcal/mol)

| Target | ESPLEY46 (46) | ESPLEY73 (73) | EXT_SEL † (133) | EXT_ALL (182) | EXT_SEL − ESPLEY73 | EXT_ALL − ESPLEY73 |
|---|---:|---:|---:|---:|---:|---:|
| ΔE‡ (barrier) | 2.10 ± 0.03 | 1.83 ± 0.06 | 1.71 ± 0.06 | 1.73 ± 0.08 | -0.12 | -0.10 |
| d1 (dipole strain) | 1.84 ± 0.10 | 1.78 ± 0.08 | 1.70 ± 0.09 | 1.70 ± 0.10 | -0.08 | -0.08 |
| d2 (dipolarophile strain) | 1.51 ± 0.08 | 1.36 ± 0.04 | 1.29 ± 0.05 | 1.28 ± 0.06 | -0.07 | -0.08 |
| elst | 5.15 ± 0.26 | 3.95 ± 0.16 | 3.62 ± 0.15 | 3.59 ± 0.14 | -0.33 | -0.36 |
| Pauli | 8.02 ± 0.39 | 7.19 ± 0.22 | 6.51 ± 0.26 | 6.45 ± 0.25 | -0.69 | -0.74 |
| OI | 4.92 ± 0.19 | 4.41 ± 0.08 | 4.02 ± 0.15 | 3.96 ± 0.10 | -0.38 | -0.45 |
| disp | 1.36 ± 0.05 | 0.47 ± 0.02 | 0.46 ± 0.02 | 0.47 ± 0.02 | -0.01 | -0.01 |
| CPCM | 3.01 ± 0.06 | 1.60 ± 0.07 | 1.55 ± 0.08 | 1.51 ± 0.09 | -0.04 | -0.09 |
| CDS | 0.34 ± 0.01 | 0.26 ± 0.00 | 0.22 ± 0.00 | 0.22 ± 0.01 | -0.04 | -0.04 |

## NMAE (MAE / mean absolute deviation of the test targets)

| Target | ESPLEY46 (46) | ESPLEY73 (73) | EXT_SEL † (133) | EXT_ALL (182) |
|---|---:|---:|---:|---:|
| ΔE‡ (barrier) | 0.300 | 0.262 | 0.244 | 0.247 |
| d1 (dipole strain) | 0.310 | 0.299 | 0.286 | 0.286 |
| d2 (dipolarophile strain) | 0.322 | 0.290 | 0.275 | 0.272 |
| elst | 0.377 | 0.289 | 0.265 | 0.263 |
| Pauli | 0.311 | 0.279 | 0.252 | 0.250 |
| OI | 0.317 | 0.284 | 0.259 | 0.255 |
| disp | 0.365 | 0.127 | 0.124 | 0.126 |
| CPCM | 0.757 | 0.402 | 0.391 | 0.380 |
| CDS | 0.495 | 0.383 | 0.319 | 0.322 |

## r²

| Target | ESPLEY46 (46) | ESPLEY73 (73) | EXT_SEL † (133) | EXT_ALL (182) |
|---|---:|---:|---:|---:|
| ΔE‡ (barrier) | 0.900 | 0.923 | 0.931 | 0.929 |
| d1 (dipole strain) | 0.883 | 0.892 | 0.894 | 0.896 |
| d2 (dipolarophile strain) | 0.859 | 0.888 | 0.893 | 0.897 |
| elst | 0.811 | 0.877 | 0.890 | 0.893 |
| Pauli | 0.863 | 0.883 | 0.895 | 0.899 |
| OI | 0.853 | 0.876 | 0.883 | 0.889 |
| disp | 0.851 | 0.978 | 0.980 | 0.980 |
| CPCM | 0.461 | 0.841 | 0.845 | 0.844 |
| CDS | 0.750 | 0.846 | 0.892 | 0.889 |

† EXT_SEL: its blocks were selected in Phase C by dev-CV on the dev rows, which overlap these Protocol A test rows, so these numbers carry selection bias; the unbiased estimate is D-1 (lockbox).

EXT_SEL = ESPLEY73 + blocks ['B1', 'B4'] (`prereg_rev5b.json`, sha256 `2a77e0034e0c`).

Every model and arm: `D2_protocolA_table.csv`; inputs: `D2_protocolA_meta.json`.
