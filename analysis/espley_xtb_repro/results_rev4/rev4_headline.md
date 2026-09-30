# rev 4 headline — G1 · ESPLEY73 · KRR_rbf (pre-registered)

Protocol A test folds, mean over seeds 22/23/14/1/2 (80/10/10, nested per-seed GridSearchCV 5-fold): MAE ± sd over seeds (kcal/mol), NMAE = MAE / mean absolute deviation of the test targets, r². n = 4839 rxns (`rows_rev4.csv`, sha256 `5c72dc7d686e`).

- **G1 = GFN2-xTB/ALPB geometry**: TS and references re-optimised with GFN2-xTB/ALPB(water) from the DFT structures (conformer and stereo choice retained); the Espley-comparable arm, not a deployment result.

| Target | MAE | NMAE | r² |
|---|---:|---:|---:|
| ΔE‡ (barrier) | 1.92 ± 0.05 | 0.272 | 0.916 |
| d1 (dipole strain) | 1.77 ± 0.10 | 0.291 | 0.896 |
| d2 (dipolarophile strain) | 1.42 ± 0.07 | 0.304 | 0.875 |
| elst | 3.91 ± 0.14 | 0.291 | 0.897 |
| Pauli | 7.15 ± 0.25 | 0.281 | 0.901 |
| OI | 4.32 ± 0.12 | 0.278 | 0.902 |
| disp | 0.45 ± 0.01 | 0.122 | 0.980 |
| CPCM | 1.56 ± 0.08 | 0.399 | 0.847 |
| CDS | 0.26 ± 0.01 | 0.374 | 0.854 |

Appendix: `rev4_appendix_by_arm.csv` (KRR_rbf × ESPLEY46 / ESPLEY54 / ESPLEY73), `rev4_appendix_by_model.csv` (ESPLEY73 × Ridge / KRR_rbf / SVR_rbf / XGB), `rev4_appendix_full.csv` (all cells), pre-ML baselines `rev4_pre_ml.csv`.
