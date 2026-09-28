# espley_xtb_repro (v6, results rev 3) — GFN2-xTB / ALPB(water) features for the Espley 2024 protocol

This reproduces the Espley 2024 protocol (DOI: 10.1039/D4DD00224E) on the Coley 5,269 dipolar cycloaddition dataset, with **AM1 replaced by GFN2-xTB** and **ALPB(water) solvation**. The DFT labels are the SMD(water) relabel assembled by method ① (repo-root `labels_all.json`, see `label_true/SMD_RELABEL.md`). Current results and their history are in [`results/SUMMARY.md`](results/SUMMARY.md).

## Method

- **Engine.** Every semi-empirical quantity comes from one program, **xtb 6.7.1** (GFN2-xTB with the ALPB water model). Each structure gets one single-point calculation, which yields:
  - the term-wise energy decomposition
  - Mulliken partial charges
  - per-atom Wiberg valences
  - the HOMO-LUMO gap and the dipole moment

  **tblite is NOT used.** The exceptions: `b_disp` uses simple-dftd3 D3(BJ)/B3LYP, and `dsasa_*` uses morfeus SASA.
- **Headline arm.** **`ESPLEY73` = 41 structural + 5 energies + 8 per-channel calculated values (B_CH8) + 19 auxiliary (AUX19)**.
- **Reactions.** 5,260 reactions: the `labels_all.json` accepted set minus 5 excluded.
  - 3090, 3766 and 4252: foreign bond.
  - 3400 and 5783: `no_forming_bond_ts`. Both forming bonds are >= 3.3 Å at the TS (3.63 / 3.65 and 3.34 / 3.56 Å), so these are not bond-forming TSs; every accepted reaction has its shorter forming bond <= 3.18 Å.

  The hygiene filter (d1 < 0, d2 < 0, d2 > 50) removes 26 more, which leaves **n = 5,234** for ML.

## B_CH8 — one real calculated value per EDA channel

All Δ = TS − dist1 − dist2 unless stated; all with ALPB(water).

- `b_strain_1/2`
  - Gas-part strain of each fragment = Δ(E_total − G_solv) between the TS-geometry fragment and its relaxed reference.
  - Distinct from `xtb_dist_*`, which is the full ALPB strain; the difference is the solvation contribution to strain.
- `b_elst`
  - Frozen-fragment monopole Coulomb: Σ_{i∈A,j∈B} q_i q_j / r_ij × 332.0637.
  - The charges q come from SEPARATE GFN2/ALPB single points on the isolated fragments at the TS geometry.
- `b_pauli` — Δ(repulsion energy), GFN2's classical repulsion term.
- `b_oi` — Δ(EHT band-structure energy) = Δ(SCC − IES − AES − AXC − dispersion − G_solv).
- `b_disp`
  - Inter-fragment D3(BJ)/B3LYP dispersion at the TS geometry.
  - This is the SAME quantity as the DFT disp channel (analytic identity, MAE(b_disp − dft_disp_dft) ≈ 2e-6 kcal/mol).
- `b_cpcm` — Δ(G_elec), the ALPB Born/dielectric term.
- `b_cds`
  - Δ(G_sasa + G_hb + G_shift), the ALPB non-polar/SASA term.
  - The per-element `dsasa_*` in AUX19 carry the functional form Σ σ_k A_k.

## Feature sets

| arm | composition | count | use |
|---|---|---:|---|
| `ESPLEY46` | 41 structural + 5 energies | 46 | ablation of the channel block |
| `ESPLEY54` | 41 + 5 + B_CH8 | 54 | the paper's 55 minus q_barrier |
| **`ESPLEY73`** | 54 + AUX19 | **73** | **headline** |

- `D_STRUCT41`: 11 distances + 15 Mulliken + 15 Wiberg valences (analogue of Table S2).
- `AUX19`: b_elst_scc, b_disp_d4, b_axc, b_ct, gap_ts/dip/dph, mu_ts/dip/dph, dmu_complexation, is_charged, dsasa_{H,C,N,O,F,Cl,Br}.

## Models · Protocol

- **Models:** Ridge, KRR(RBF), SVR(RBF), XGB. StandardScaler on X; y is standardized with `TransformedTargetRegressor`.
- **Protocol A:** 80/10/10 splits over seeds 22/23/14/1/2. **Nested CV:** GridSearchCV(5-fold) runs on each seed's own training split. KRR(RBF) is fixed a priori as the headline model.
- **Protocol B:** 5-fold OOF (Linear/Ridge/RF/GBR/XGB).
- **Metrics:** MAE, NMAE (= MAE / MAD_target), RMSE, r², Pearson r.

## Targets

- **Reported (9):** `dft_barrier_kcal`, `dft_d1_kcal`, `dft_d2_kcal`, `dft_{elst,pauli,oi,disp,cpcm,cds}_dft`.
- **Also trained (12 array elements) but not reported:**
  - `dft_e_bond_kcal`: the sum of the 6 channels.
  - `dft_eint_spe_kcal`: derived from d1, d2 and barrier. It is used only as the baseline in the Espley comparison.
  - `dft_c_ghost_kcal`: = eint_spe − e_bond, a method artefact (BSSE + cavity). It is reported once in the SI.

## Pipeline

```
s01_xtb_array.sh    xtb features, 18 slices  (xtb_slice.py)
s02_aggregate.sh    -> xtb_features.parquet  (aggregate.py; derives dft_c_ghost_kcal, is_charged)
refresh_targets.py  (labels changed only) swap the dft_* targets in xtb_features.parquet, no xTB rerun
s03_smoketest.sh    schema / grid check
s03_ml_array.sh     12-target array x 3 arms x (Protocol A 4 + Protocol B 5)   (train_ml_single.py)
s04_aggregate_ml.sh -> ml_report.json, ml_table_espley.csv, predictions.parquet
s05_plot.sh         ESPLEY73 figures, 9 reported targets (plot_results.py)
s06/s07             charge / group-split / MMP analyses on results/
s08_publish_rev3.sh copy outputs into results/ and run s06/s07 (rev 3)
```

## Compute

- **xTB single point:** about 0.25 s per reaction on a single core. One full pipeline is 5,260 rxns × 5 SPEs = 26,300 xtb calls (about 15–20 min with 10 concurrent slice tasks).
- **ML:** a 12-element array; each element runs about 25–70 min on 8 CPUs, with 10 running at once.
