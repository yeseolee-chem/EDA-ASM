# espley_xtb_repro (v7, results rev 4) — GFN2-xTB / ALPB(water) features for the Espley 2024 protocol

This reproduces the Espley 2024 protocol (DOI: 10.1039/D4DD00224E) on the Coley 5,269 dipolar cycloaddition dataset, with **AM1 replaced by GFN2-xTB** and **ALPB(water) solvation**. The targets are the DFT labels on the DFT geometries: the SMD(water) relabel assembled by method ① (repo-root `labels_all.json`, see `label_true/SMD_RELABEL.md`).

**rev 4 (2026-09-30)** computes the features on **xTB-level geometries (G1)**. It does not use the DFT geometries the labels were computed on. The task is therefore "cheap geometry → expensive DFT label", the same task as Espley (AM1 geometry → DFT targets).

The rev 3 features were computed on the label DFT geometry. They are kept only as **G0 = DFT oracle geometry (upper bound, not deployable)**, re-run on the same rows.

- Spec: [`REV4_XTB_GEOMETRY.md`](REV4_XTB_GEOMETRY.md)
- Pre-registration: [`results_rev4/PREREG_REV4.md`](results_rev4/PREREG_REV4.md)
- Results: [`results_rev4/SUMMARY.md`](results_rev4/SUMMARY.md)
- The rev 1–3 result files were deleted; they remain in git history (≤ 1ada37aa).

## Geometries (`xtb_slice.py --geom`)

| mode | TS | reactant references | DFT information left | use |
|---|---|---|---|---|
| `dft` = **G0** | Coley DFT TS (label `ts_file`) | Coley DFT references (`rel1_file`, `rel2_file`, incl. `_alt`) | all | upper bound only |
| `g1` = **G1** | ORCA 6.1.1 `! XTB2 ALPB(water) OptTS Freq`, started from the DFT TS | xtb 6.7.1 `--opt tight --alpb water`, started from the DFT references | conformer and stereo choice | **headline** |
| `g2` | autodE from SMILES at the xTB level (Phase 5, not run) | autodE xTB conformers | none | deployment (pending user decision) |

**G1 generation** is done by `g1_geom.py` (`r4_g1_smoke.sh`, `r4_g1_array.sh`, `r4_g1_summary.sh`).
- The ORCA input is `%geom Calc_Hess true Recalc_Hess 5 MaxIter 200 end`, `%pal nprocs 2`.
- ORCA calls `otool_xtb`, which is the same xtb 6.7.1 build (edcfbbe) as the feature engine. The xtb 6.7.1 gradient at ORCA's TS is < 5e-5 Eh/bohr.
- The atom order is kept, so the label's `formed_pairs_ts`, roles and `A_idx` apply unchanged.
- **Gates.** These are the D1 pipeline functions, imported from a read-only snapshot of `d1-build@d573111e` (`$D1_SNAPSHOT`). D1 itself is a separate project and is not touched.
  - First imaginary frequency ≤ −40 cm⁻¹, and every other one > −50 cm⁻¹.
  - Imaginary-mode share on the forming bonds ≥ 0.5.
  - Longer forming bond ≥ 1.6 Å, and shorter forming bond < 3.3 Å.
  - No foreign inter-fragment bond.
  - G1 partition = label `A_idx`.
  - The optimised references are graph-isomorphic to the DFT references.
- A reaction that fails a gate gets `xtb_status = g1_fail:<reason>`.

## Features (identical code for every geometry)

- **Engine.** Every semi-empirical quantity comes from **xtb 6.7.1** (GFN2-xTB, ALPB water). Each structure gets one single-point calculation, which yields the term-wise energies, Mulliken charges, per-atom Wiberg valences, the HOMO-LUMO gap and the dipole moment. **tblite is not used.** The exceptions are `b_disp` (simple-dftd3 D3(BJ)/B3LYP) and `dsasa_*` (morfeus SASA).
- **Five single points per reaction:** two references, two TS-geometry fragments, and the TS.

B_CH8 terms, one calculated value per EDA channel. Δ = TS − frag1 − frag2 at the TS geometry unless stated; everything is computed with ALPB(water).
- `b_strain_1/2` — gas-part strain of each fragment (Δ(E_total − G_solv) vs its relaxed reference).
- `b_elst` — frozen-fragment monopole Coulomb with charges from the isolated TS-geometry fragments.
- `b_pauli` — Δ(repulsion energy).
- `b_oi` — Δ(EHT band-structure energy).
- `b_cpcm` — Δ(G_elec).
- `b_cds` — Δ(G_sasa + G_hb + G_shift).
- `b_disp` — inter-fragment D3(BJ)/B3LYP dispersion.
  - **On G0 this is analytically the DFT disp label** (MAE 2.4e-6 kcal/mol): the label itself is a feature, so the G0 disp channel is not an ML result and is blanked in every rev 4 table.
  - On G1 it is a genuine feature (MAE vs the label 0.94 kcal/mol).

| arm | composition | count |
|---|---|---:|
| `ESPLEY46` | 11 distances + 15 Mulliken + 15 Wiberg valences + 5 xTB energies | 46 |
| `ESPLEY54` | 46 + B_CH8 | 54 |
| **`ESPLEY73`** | 54 + AUX19 (b_elst_scc, b_disp_d4, b_axc, b_ct, gaps, dipoles, dmu_complexation, is_charged, dsasa_{H,C,N,O,F,Cl,Br}) | **73** |

## Models, rows, targets (pre-registered, `results_rev4/PREREG_REV4.md`)

- **Protocol A only** (as rev 3):
  - 80/10/10 splits over seeds 22/23/14/1/2 (`split_80_10_10`).
  - Nested per-seed GridSearchCV (5-fold).
  - StandardScaler on X, y standardised (`TransformedTargetRegressor`).
  - Models: Ridge, KRR(RBF), SVR(RBF), XGB.
- **Rows:** `results_rev4/rows_rev4.csv`, **n = 4,839** = G1 ok (4,860) ∩ G0 ok ∩ hygiene (d1 ≥ 0, d2 ≥ 0, d2 ≤ 50). G0 and G1 use exactly these rows, so their splits are identical.
- **Targets (9):** barrier, d1, d2, elst, pauli, oi, disp, cpcm, cds.
- **Headline: G1 · ESPLEY73 · KRR(RBF)**, fixed a priori. G0 is reported only as the upper bound.

## Pipeline (submit from this directory of the checkout; every script sources `r4_env.sh`)

```
r4_g1_smoke.sh        Phase 1-1  G1 engine smoke test (5 rxns)                       -> $G1_ROOT/_smoke
r4_g1_array.sh        Phase 1-2  G1 geometries, 18 slices as 9 x 2      (g1_geom.py) -> $G1_ROOT/<rid>/
r4_g1_summary.sh      Phase 1-2  success rate, failure types, RMSD                   -> results_rev4/g1_geometry_summary.*
r4_feat_all.sh        Phase 2    GEOM=dft|g1: 18 slices + aggregate (+ dft: 1e-8 check vs a reference parquet)
                                 (or r4_feat_array.sh / r4_feat_aggregate.sh / r4_feat_verify.sh)
r4_phase2_report.sh   Phase 2    G0 vs G1 feature shift, MAE(b_disp - disp)          -> results_rev4/phase2_*
r4_make_rows.sh       Phase 3-1  pre-registered rows                                 -> results_rev4/rows_rev4.*
r4_ml_smoketest.sh    Phase 3    compile / import / rows gate (no training)
r4_ml_array.sh        Phase 3-2  element i: geom g0 (i<9) / g1, target i%9; refuses unless PREREG is committed
r4_ml_aggregate.sh    Phase 3-4  ml_report / predictions / figures per geometry + rev4_tables.py -> results_rev4/rev4_*
r4_downstream.sh      Phase 3-3  GEOM=g0|g1: analyze_extra.py + evaluate_pairs.py     -> results_rev4/downstream_<geom>/
r4_compare_prep.sh    Phase 4    compare_espley.py prep (both geometries)
r4_compare_train.sh   Phase 4    0-13 ours (G0/G1 x 7 targets), 14-15 Espley role d1/d2 (their protocol / our pipeline)
r4_compare_plot.sh    Phase 4    intersection scoring, tables, figures                -> results_rev4/espley_compare_*
r4_ablation.sh        Phase 4-3  geometry-information ablation (ABL_ROWS=common|own) (espley_fairness_ablation.py)
r4_ablation_aggregate.sh                                                             -> results_rev4/espley_geometry_ablation*
r4_bath_am1.sh        Phase 4-4  start structure of Espley's AM1 TS optimisations (HTTP range reads of BATH-01480)
r4_pack.sh            run several elements of an r4 array script in one allocation (one queue slot)
```

Inputs outside the repo:
- Coley profiles `$ESPLEY_PROF`.
- The Coley CSV (mapped SMILES).
- `label_true/work/input_meta.csv` (label `A_idx`, `$ESPLEY_META`).
- The authors' repo files under `$ESPLEY_REPO_DATA` (default `/gpfs/tmp_cpu2/yeseo1ee/espley_compare`):
  - `feature_selection/_f_selection/tt/manual_tt_solvent.pkl`
  - `machine_learning/tt/solvent/ml_results.pkl`
  - `hyperparameter_tuning/tt/solvent/hps.pkl`
  - `hyperparameter_tuning/hyp_tuning.py` (fetched)

Scratch outputs go under `/gpfs/tmp_cpu2/yeseo1ee/{espley_xtb_g1, espley_xtb, espley_rev4}`.

## Compute

- **G1 geometries.** 216 core-h for 5,260 reactions.
  - ORCA OptTS+Freq takes 0.2–3 min per reaction with 2 cores (analytic xTB Hessian).
  - The references take about 1 s.
  - Concurrent ORCA runs on one node need `OMPI_MCA_rmaps_base_oversubscribe=1` and no core binding (`r4_env.sh`). 16 runs hit an MPI finalize bus error on shared nodes and were rerun unchanged.
- **Features.** About 0.25 s per reaction per core. One geometry takes about 15 min with 4–8 cores.
- **ML.** One (geometry, target) element takes 3 arms × 4 models × 5 seeds on 8 cores. Pin BLAS to one thread (`OMP_NUM_THREADS=1`), otherwise every GridSearchCV worker spawns one thread per allocated core.
