# rev 4 headline — G1 · ESPLEY73 · KRR_rbf (pre-registered)

Protocol A test folds, mean over seeds 22/23/14/1/2 (80/10/10, nested per-seed GridSearchCV 5-fold): MAE ± sd over seeds (kcal/mol), NMAE = MAE / mean absolute deviation of the test targets, r². n = 4839 rxns (`rows_rev4.csv`, sha256 `5c72dc7d686e`), identical rows and test splits on both geometries.

- **G0 = DFT oracle geometry (upper bound)**: features on the DFT TS and references the labels were computed on; not deployable, never the headline.
- **G1 = GFN2-xTB/ALPB geometry**: TS and references re-optimised with GFN2-xTB/ALPB(water) from the DFT structures (conformer and stereo choice retained); the Espley-comparable arm, not a deployment result.

| Target | G0 upper bound MAE | G0 NMAE | G0 r² | G1 MAE | G1 NMAE | G1 r² | G1 − G0 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ΔE‡ (barrier) | 1.47 ± 0.05 | 0.207 | 0.952 | 1.92 ± 0.05 | 0.272 | 0.916 | +0.46 |
| d1 (dipole strain) | 1.12 ± 0.05 | 0.184 | 0.959 | 1.77 ± 0.10 | 0.291 | 0.896 | +0.65 |
| d2 (dipolarophile strain) | 0.84 ± 0.04 | 0.181 | 0.962 | 1.42 ± 0.07 | 0.304 | 0.875 | +0.57 |
| elst | 1.53 ± 0.06 | 0.114 | 0.984 | 3.91 ± 0.14 | 0.291 | 0.897 | +2.38 |
| Pauli | 2.06 ± 0.06 | 0.081 | 0.991 | 7.15 ± 0.25 | 0.281 | 0.901 | +5.09 |
| OI | 1.17 ± 0.03 | 0.075 | 0.992 | 4.32 ± 0.12 | 0.278 | 0.902 | +3.15 |
| disp | n/a † | n/a † | n/a † | 0.45 ± 0.01 | 0.122 | 0.980 | — |
| CPCM | 0.96 ± 0.05 | 0.246 | 0.949 | 1.56 ± 0.08 | 0.399 | 0.847 | +0.60 |
| CDS | 0.23 ± 0.01 | 0.342 | 0.881 | 0.26 ± 0.01 | 0.374 | 0.854 | +0.02 |

† G0 disp: analytic identity (b_disp) — not an ML result. On the DFT geometry the b_disp feature equals the disp label; the G0 numbers are blank in `rev4_headline.csv` and kept, flagged in `g0_note`, in the appendix CSVs for traceability only. G1 disp, and G0 disp in ESPLEY46 (no b_disp), are ordinary predictions.

Appendix: `rev4_appendix_by_arm.csv` (KRR_rbf × ESPLEY46 / ESPLEY54 / ESPLEY73), `rev4_appendix_by_model.csv` (ESPLEY73 × Ridge / KRR_rbf / SVR_rbf / XGB), `rev4_appendix_full.csv` (all cells), pre-ML baselines `rev4_pre_ml.csv`.
