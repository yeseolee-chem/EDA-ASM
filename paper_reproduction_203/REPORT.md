# paper_reproduction_203 — ML pipeline results

Reproduction of Espley et al. 2024 (*Digital Discovery* **3**, 2479, DOI 10.1039/D4DD00224E) —
SQM (AM1)-based ML prediction of DFT distortion + interaction energies.

## Setup

- **Dataset (gated cohort)**: 121 / 203 rxns passing structure gate (d_forming ∈ [1.8, 3.2] Å) + barrier > 0
- **Structure gate excluded**: 82 rxns (49 bonded d < 1.8 Å, 33 nonphysical barrier). See results/excluded_rxns.csv
- **DFT target formula**: paper DIAS (Espley 2024) — barrier = E_TS − E_R₁ − E_R₂; interaction = barrier − distortion. All 5 SPEs at ωB97X-D3BJ/def2-TZVP + CPCM(water) via ORCA 6.1.1. **NOT** using EDA-NOCV Bond Energy (gave non-physical barriers).
- **AM1 features**: Gaussian 16 `#P AM1 Freq NoSymm` (gas phase), Mulliken + APT charges + Morfeus (BuriedVolume, SASA, Sterimol)
- **5-atom labels**: user-manual (400/400 confirmed via port 5578 viz)
- **Fragment A/B partition + type**: user-manual (400/400)
- **Models**: Ridge, KRR (rbf), SVR (rbf), RandomForest — sklearn only (Keras 3 NN branch skipped)
- **CV**: 80/10/10 Monte Carlo × 25 seeds `{1..25} \ {23}` + paper seeds `{23, 22, 14, 1, 2}`
- **Feature count**: filtered from 307 to 154 via correlation (|r| > 0.99) + variance (< 0.05) thresholds

## Test MAE — mean ± std over 25 seeds (kcal/mol)

| target | Ridge | KRR | SVR | RF | best |
|---|---|---|---|---|---|
| e_barrier | 5.05 ± 0.51 | 5.72 ± 0.91 | 5.24 ± 0.97 | 4.98 ± 0.41 | **RF: 4.98** |
| distortion | 3.98 ± 0.47 | 6.65 ± 2.81 | 6.97 ± 2.14 | 4.88 ± 1.40 | **RIDGE: 3.98** |
| interaction | 3.41 ± 0.69 | 5.05 ± 1.64 | 6.04 ± 1.23 | 3.75 ± 0.83 | **RIDGE: 3.41** |
| distortion_dipole | 4.28 ± 0.69 | 4.93 ± 1.26 | 5.06 ± 1.19 | 4.35 ± 1.09 | **RIDGE: 4.28** |
| distortion_dipolarophile | 2.79 ± 0.38 | 2.99 ± 1.87 | 3.42 ± 1.19 | 2.33 ± 0.80 | **RF: 2.33** |

## Comparison with paper Table 1 (ds3, N=730)

| target | our best (25-seed mean) | paper ds3 SVR | paper pre-ML AM1-DFT |
|---|---|---|---|
| e_barrier | 4.98 | (not in paper) | — |
| distortion | 3.98 | (not in paper) | — |
| interaction | 3.41 | 2.46 | 20.01 |
| distortion_dipole | 4.28 | 2.55 | 3.59 |
| distortion_dipolarophile | 2.33 | 2.37 | 3.81 |

## Best model per target — parity plots

- **e_barrier** — SVR (seed 22, test MAE 4.06): [parity_best_e_barrier_dft.png](figures/parity_best_e_barrier_dft.png)
- **distortion** — RIDGE (seed 2, test MAE 3.35): [parity_best_distortion_dft.png](figures/parity_best_distortion_dft.png)
- **interaction** — RIDGE (seed 1, test MAE 2.55): [parity_best_interaction_dft.png](figures/parity_best_interaction_dft.png)
- **distortion_dipole** — RF (seed 23, test MAE 2.78): [parity_best_distortion_dipole_dft.png](figures/parity_best_distortion_dipole_dft.png)
- **distortion_dipolarophile** — KRR (seed 14, test MAE 1.13): [parity_best_distortion_dipolarophile_dft.png](figures/parity_best_distortion_dipolarophile_dft.png)

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
