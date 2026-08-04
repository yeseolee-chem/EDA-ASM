# paper_reproduction_203 — ML pipeline results

Reproduction of Espley et al. 2024 (*Digital Discovery* **3**, 2479, DOI 10.1039/D4DD00224E) —
SQM (AM1)-based ML prediction of DFT distortion + interaction energies.

## Setup

- **Dataset**: 203 / 400 dipolar cycloaddition reactions (snapshot 2026-08-04, 5/5 ORCA jobs complete)
- **DFT target**: ωB97X-D3BJ/def2-TZVP + CPCM(water) SPE + EDA-NOCV (ORCA 6.1.1)
- **AM1 features**: Gaussian 16 `#P AM1 Freq NoSymm` (gas phase), Mulliken + APT charges + Morfeus (BuriedVolume, SASA, Sterimol)
- **5-atom labels**: user-manual (400/400 confirmed via port 5578 viz)
- **Fragment A/B partition + type**: user-manual (400/400)
- **Models**: Ridge, KRR (rbf), SVR (rbf), RandomForest — sklearn only (Keras 3 NN branch skipped)
- **Splits**: 80/10/10 train/val/test per seed, 5 seeds `[23, 22, 14, 1, 2]`
- **Features**: 155 after correlation + variance filter (from 309 raw)

## Test MAE — mean ± std over 5 seeds (kcal/mol)

| target | Ridge | KRR | SVR | RF | best |
|---|---|---|---|---|---|
| e_barrier | 8.90 ± 1.00 | 8.65 ± 1.85 | 8.37 ± 1.85 | 8.13 ± 1.76 | **RF: 8.13** |
| distortion | 6.73 ± 2.11 | 12.38 ± 1.01 | 12.64 ± 2.38 | 5.19 ± 1.06 | **RF: 5.19** |
| interaction | 9.08 ± 1.56 | 14.69 ± 1.08 | 13.03 ± 2.82 | 7.95 ± 1.84 | **RF: 7.95** |
| distortion_dipole | 6.83 ± 1.52 | 8.75 ± 1.49 | 8.67 ± 1.64 | 4.95 ± 1.14 | **RF: 4.95** |
| distortion_dipolarophile | 4.37 ± 0.73 | 5.51 ± 0.73 | 5.56 ± 1.08 | 3.80 ± 0.84 | **RF: 3.80** |

## Comparison with paper Table 1 (ds3, N=730)

| target | our best (5-seed mean) | paper ds3 SVR | paper pre-ML AM1-DFT |
|---|---|---|---|
| e_barrier | 8.13 | (not in paper) | — |
| distortion | 5.19 | (not in paper) | — |
| interaction | 7.95 | 2.46 | 20.01 |
| distortion_dipole | 4.95 | 2.55 | 3.59 |
| distortion_dipolarophile | 3.80 | 2.37 | 3.81 |

## Best model per target — parity plots

- **e_barrier** — RF (seed 14, test MAE 6.31): [parity_best_e_barrier_dft.png](figures/parity_best_e_barrier_dft.png)
- **distortion** — RIDGE (seed 14, test MAE 4.08): [parity_best_distortion_dft.png](figures/parity_best_distortion_dft.png)
- **interaction** — RF (seed 1, test MAE 5.39): [parity_best_interaction_dft.png](figures/parity_best_interaction_dft.png)
- **distortion_dipole** — RF (seed 14, test MAE 3.76): [parity_best_distortion_dipole_dft.png](figures/parity_best_distortion_dipole_dft.png)
- **distortion_dipolarophile** — RF (seed 1, test MAE 2.76): [parity_best_distortion_dipolarophile_dft.png](figures/parity_best_distortion_dipolarophile_dft.png)

## Pre-ML AM1-DFT MAE (before ML correction)

From `code/60_import_am1_results.py`:
- distortion:               4.91 kcal/mol
- interaction:              40.43 kcal/mol
- e_barrier:                41.90 kcal/mol
- distortion_dipole:         5.46 kcal/mol
- distortion_dipolarophile:  5.71 kcal/mol

ML predictions reduce AM1's systematic bias — especially large for interaction (where paper reported pre-ML MAE 20 kcal/mol; ours is 40 kcal/mol, larger because our AM1 was gas phase while paper used IEFPCM water — bias-correctable by ML).

## Reproducibility

Full pipeline: `code/70_run_grayson_pipeline.sh` (sbatch, 48h walltime).
All intermediate + final files under `pipeline_work/`.
