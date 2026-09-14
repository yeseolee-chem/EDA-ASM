# espley_xtb_repro — Espley 2024 [3+2] barrier-prediction protocol, reproduced with xTB features on the Coley 5269 labels

**Task**: reproduce Espley et al. 2024 (Digital Discovery, DOI:10.1039/D4DD00224E)
distortion-interaction ML pipeline, but with two changes:

1. Replace **AM1** with **GFN2-xTB** (semi-empirical DFTB) for the 5 SPE
   species per reaction (gs1, gs2, dist1, dist2, ts).
2. Replace the Bath 1480 tt subset (3,504 rxns) with **Coley 5269 dipolar
   cycloaddition** (5,260 status=ok records in `labels_all.json`).

No pre-filtering: all 5,260 records are attempted. Reactions whose xTB SPE
fails are recorded with NaN xTB fields (not dropped) so the failure mode is
visible in the final table.

## Files

- `xtb_slice.py` — process a slice of rxns; compute 5 SPEs + 5 Espley
  derived features per rxn; write partial parquet.
- `aggregate.py` — merge partial parquets into `xtb_features.parquet`
  (5,260 rows, all Espley 5-feature energies + DFT target columns from labels).
- `train_ml.py` — train 4 sklearn regressors on xTB features → DFT
  quantities; 5-fold CV; write `ml_report.json` and `predictions.parquet`.
- `s01_xtb_array.sh`, `s02_aggregate.sh`, `s03_ml.sh` — sbatch chain.

## Method choices (documented in code + reports)

- **Geometries**: Coley DFT-relaxed (r*.xyz, r*_alt.xyz per labels_all's
  `rel1_file` / `rel2_file`, TS from `ts_file`) used AS-IS. xTB is not
  re-optimising. This is "xTB energy at DFT geometry" — a surrogate for
  the DFT interaction/strain surface, not a full xTB PES.
- **Fragment split**: graph rule (`fragmenter.partition`) re-derives
  which TS atoms belong to fragment 1 (dipole) vs fragment 2
  (dipolarophile). Deterministic and consistent with the labeling.
- **Charges**: `charge1`, `charge2` from labels_all; total `q_tot = q1+q2`.
- **Solvation**: ALPB water (Grimme). If tblite refuses, gas phase (logged).
- **Features** (5, no q_barrier): `xtb_e_barrier`, `xtb_dist_dipole`,
  `xtb_dist_dipolarophile`, `xtb_sum_distortion`, `xtb_interaction`.
- **Targets** (from labels_all, DFT B3LYP-D3(BJ)/def2-TZVP CPCM(SMD water)):
  `dft_barrier` (= (e_ab − e_frag1_rel − e_frag2_rel) × 627.5), `d1_kcal`
  (dipole strain), `d2_kcal` (dipolarophile strain), `e_bond_kcal` (EDA
  interaction), plus 6 individual EDA channels (elst/pauli/oi/disp/cpcm/cds).
- **ML models**: LinearRegression (baseline), Ridge, RandomForest,
  GradientBoosting. 5-fold CV, seed 42.
- **ML metrics**: MAE, RMSE, r² (per fold + aggregated).

## Deliverables

- `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features.parquet`
- `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/ml_report.json`
- `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/predictions.parquet`

## Compute cost

xTB SPE-only on Coley DFT geoms is fast (~0.1–0.5 s per SPE on 1 core).
5 SPE × 5,260 rxns = 26,300 xTB calls. Array of 20 elements ≈ 30 min wall.
ML sklearn on 5-feature × 5,260 → ~10 min. End-to-end ~1 h.
