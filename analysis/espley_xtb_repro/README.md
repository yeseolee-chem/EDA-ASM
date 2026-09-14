# espley_xtb_repro (v5) — GFN2-xTB / ALPB(water) features for Espley 2024 protocol

Espley 2024 protocol (DOI: 10.1039/D4DD00224E) reproduced on the Coley 5,269 dipolar
cycloaddition dataset with **AM1 replaced by GFN2-xTB** and **ALPB(water) solvation**.

## Method

All semi-empirical quantities were obtained with a single program, **xtb 6.7.1**
(GFN2-xTB with the ALPB water model), from one single-point calculation per structure:
the term-wise energy decomposition, Mulliken partial charges, per-atom Wiberg
valences, the HOMO-LUMO gap and the dipole moment. **tblite is NOT used.**

Features: **54 = 41 structural + 5 energies + 8 per-channel calculated values (B_CH8)**;
`AUX18` gives an extended 72-feature arm; `ESPLEY46` (= 41 + 5) is an ablation of the
channel block. No variance filter. **5,260 reactions** (`labels_all.json` accepted set
minus 5 excluded: 3090/3766/4252 foreign-bond, 3400/5783 oi>0).

## B_CH8 — one real calculated value per EDA channel

All Δ = TS − dist1 − dist2 unless stated; all with ALPB(water).

- `b_strain_1/2` — gas-part strain of each fragment = Δ(E_total − G_solv) between the
  TS-geometry fragment and its relaxed reference. Distinct from `xtb_dist_*` (full
  ALPB strain); the difference is the solvation contribution to strain.
- `b_elst` — frozen-fragment monopole Coulomb, Σ_{i∈A,j∈B} q_i q_j / r_ij × 332.0637,
  with q from SEPARATE GFN2/ALPB single points on the isolated fragments at the TS
  geometry (densities are frozen, not relaxed in the complex).
- `b_pauli` — Δ(repulsion energy), GFN2's classical repulsion term.
- `b_oi` — Δ(EHT band-structure energy) = Δ(SCC − IES − AES − AXC − dispersion − G_solv),
  the one-electron/orbital part.
- `b_disp` — inter-fragment D3(BJ)/B3LYP dispersion at the TS geometry. This is the
  SAME quantity as the DFT disp channel; b_disp is not an ML prediction but is
  analytically determined by the TS geometry. Gate §4 verifies MAE(b_disp − dft_disp_dft) < 0.01 kcal/mol.
- `b_cpcm` — Δ(G_elec), the ALPB Born/dielectric term (analogue of Delta CPCM Dielectric).
- `b_cds` — Δ(G_sasa + G_hb + G_shift), ALPB non-polar/SASA term (analogue of
  Delta SMD CDS correction). Weakest of the eight; per-element `dsasa_*` in AUX
  carries the Σ σ_k A_k functional form a single scalar cannot.

## Feature sets

| arm | composition | count | 용도 |
|---|---|---:|---|
| `ESPLEY46` | 41 structural + 5 energies | 46 | ablation of channel block |
| **`ESPLEY54`** | 41 + 5 + B_CH8 | **54** | main (paper's 55 minus q_barrier) |
| `ESPLEY72` | 54 + AUX18 | 72 | extended |

- `D_STRUCT41`: 11 distances + 15 Mulliken + 15 Wiberg-valence (Table S2 analogue)
- `AUX18`: b_elst_scc, b_disp_d4, b_axc, b_ct, gap_ts/dip/dph, mu_ts/dip/dph,
  dmu_complexation, dsasa_{H,C,N,O,F,Cl,Br}

Pairwise Wiberg bond orders (`wbo_inter`, `wbo_form_*`) are deliberately NOT in the
feature set: xtb prints them only above ~0.1 threshold; per-atom valences carry the
same information exactly.

## Models · Protocol

- Ridge, KRR(RBF), SVR(RBF), XGB — StandardScaler in front.
- Protocol A: 80/10/10 × seeds 22/23/14/1/2, GridSearchCV(5-fold) on the seed-23
  training split, best params reused for all 5 seeds.
- Protocol B: 5-fold OOF (Linear/Ridge/RF/GBR/XGB).
- **Metrics**: MAE, NMAE (= MAE / MAD_target), RMSE, r², Pearson r.
  For b^ch channels the raw pre-ML MAE is misleading (different absolute scales);
  NMAE + pre_ml_r are the informative comparisons.
- No variance filter.

## Targets (11)

`dft_barrier_kcal, dft_d1_kcal, dft_d2_kcal, dft_eint_spe_kcal, dft_e_bond_kcal,`
`dft_{elst,pauli,oi,disp,cpcm,cds}_dft`

## Pipeline

```
1. atomic coords -> xtb (5 SPEs per rxn, ALPB water) -> 54/46/72 features
2. features + DFT targets -> parquet
3. ML: 11-target array x 3 feature sets x 4 models (Protocol A + B)
4. plots: scatter per model, MAE bar per channel
```

## Compute

- xTB single point: ~0.25 s/rxn, single core.
- 5,260 rxns × 5 SPEs = 26,300 xtb calls per full pipeline (~2 h single-thread,
  ~15–20 min with 10 concurrent slice tasks).
- ML: 11-target array, wall ~10 min per array element (%10 concurrent).

## Files (`analysis/espley_xtb_repro/`)

- `xtb_slice.py` — one slice of the xtb feature computation
- `aggregate.py` — merge slice parquets → `xtb_features.parquet`
- `train_ml_single.py` — one target for the parallel ML array
- `aggregate_ml.py` — merge per-target ML JSONs → report + CSV + preds
- `plot_results.py` — scatter (per model, 8 channels) + MAE bar (grouped by model)
- `fragmenter.py` — graph-rule partition (local copy)
- `s01_xtb_array.sh`, `s02_aggregate.sh`, `s03_ml_array.sh`, `s04_aggregate_ml.sh`, `s05_plot.sh`
