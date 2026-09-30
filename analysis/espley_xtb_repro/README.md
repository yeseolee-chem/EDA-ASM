# espley_xtb_repro (v8, results rev 4, rev 5 in progress) — GFN2-xTB / ALPB(water) features for the Espley 2024 protocol

> G0(DFT oracle geometry) 결과는 2026-09-30 사용자 결정으로 폐기. git ≤ `740ba89`에만 남음.

This reproduces the Espley 2024 protocol (DOI: 10.1039/D4DD00224E) on the Coley 5,269 dipolar cycloaddition dataset, with **AM1 replaced by GFN2-xTB** and **ALPB(water) solvation**. The targets are the DFT labels on the DFT geometries: the SMD(water) relabel assembled by method ① (repo-root `labels_all.json`, see `label_true/SMD_RELABEL.md`).

**rev 4 (2026-09-30)** computes the features on **xTB-level geometries (G1)**. It does not use the DFT geometries the labels were computed on. The task is therefore "cheap geometry → expensive DFT label", the same task as Espley (AM1 geometry → DFT targets).

- Specs: [`REV4_XTB_GEOMETRY.md`](../../docs/specs/REV4_XTB_GEOMETRY.md), [`REV5_FEATURES_FIGURES.md`](../../docs/specs/REV5_FEATURES_FIGURES.md)
- Pre-registration: [`results_rev4/PREREG_REV4.md`](results_rev4/PREREG_REV4.md) (historical record, not edited)
- Results: [`results_rev4/SUMMARY.md`](results_rev4/SUMMARY.md); rev 5 outputs go to `results_rev5/`
- The rev 1–3 result files were deleted; they remain in git history (≤ 1ada37aa).

## Geometries (`xtb_slice.py --geom`)

| mode | TS | reactant references | DFT information left | use |
|---|---|---|---|---|
| `g1` = **G1** (default) | ORCA 6.1.1 `! XTB2 ALPB(water) OptTS Freq`, started from the DFT TS | xtb 6.7.1 `--opt tight --alpb water`, started from the DFT references | conformer and stereo choice | **headline** |
| `g2` | autodE from SMILES at the xTB level (not run) | autodE xTB conformers | none | deployment (pending user decision) |

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

## Features (`xtb_slice.py`, the same code for g1 and g2)

- **Engine.** Every semi-empirical quantity comes from **xtb 6.7.1** (GFN2-xTB, ALPB water). Each structure gets one single-point calculation, which yields the term-wise energies, Mulliken charges, per-atom Wiberg valences, the HOMO-LUMO gap and the dipole moment. **tblite is not used** here (rev 5 uses tblite 0.7.0 only for the orbital-overlap block B1). The exceptions are `b_disp` (simple-dftd3 D3(BJ)/B3LYP) and `dsasa_*` (morfeus SASA).
- **Five single points per reaction:** two references, two TS-geometry fragments, and the TS.

B_CH8 terms, one calculated value per EDA channel. Δ = TS − frag1 − frag2 at the TS geometry unless stated; everything is computed with ALPB(water).
- `b_strain_1/2` — gas-part strain of each fragment (Δ(E_total − G_solv) vs its relaxed reference).
- `b_elst` — frozen-fragment monopole Coulomb with charges from the isolated TS-geometry fragments.
- `b_pauli` — Δ(repulsion energy).
- `b_oi` — Δ(EHT band-structure energy).
- `b_cpcm` — Δ(G_elec).
- `b_cds` — Δ(G_sasa + G_hb + G_shift).
- `b_disp` — inter-fragment D3(BJ)/B3LYP dispersion on the xTB TS: a genuine feature (MAE vs the DFT disp label 0.94 kcal/mol).

| arm | composition | count |
|---|---|---:|
| `ESPLEY46` | 11 distances + 15 Mulliken + 15 Wiberg valences + 5 xTB energies | 46 |
| `ESPLEY54` | 46 + B_CH8 | 54 |
| **`ESPLEY73`** | 54 + AUX19 (b_elst_scc, b_disp_d4, b_axc, b_ct, gaps, dipoles, dmu_complexation, is_charged, dsasa_{H,C,N,O,F,Cl,Br}) | **73** |

## rev 5 feature blocks B1..B6 (`ext_features.py`)

The 112 columns and every constant are in `rev5_common.py` (`BLOCKS`); the definitions are in the `ext_features.py` docstring. Everything is computed on the G1 structures only: the TS, rel1, rel2, the TS cut by the label `A_idx` into fA (dipole) and fB (dipolarophile), and, for B6, the xtb-optimised product. The geometry step is `xtb_slice.ts_fragments()`, the same code path as the rev 4 features. Each reaction is gated on its 11 distances matching the G1 parquet.

| block | engine | what |
|---|---|---|
| B1 (21) | tblite 0.7.0, ALPB water | MO overlap 𝒮 = C_Aᵀ S_AB C_B of the TS-geometry fragments: occupied–occupied Σ𝒮², Σ𝒮²/Δε (Δε floored at 1 eV), frontier terms; HOMO/LUMO of fA, fB, rel1, rel2 from xtb `--json` |
| B2 (23) | geometry | inter-fragment exp(−r/ρ) sums, Bondi overlap, closest non-forming contacts, heavy–heavy distance histogram |
| B3 (7) | xtb `--json` of fA, fB | frozen-fragment charge–dipole, dipole–dipole and charge–quadrupole energies; penetration proxies |
| B4 (39) | xtb `--vipea`, `--vfukui`, D4 block | μ, η, ω of fA, fB, rel1, rel2; ΔN; Fukui f⁺ f⁻ f⁰ and D4 α(0) of the 5 reacting atoms; molecular α(0); GEDT (= `b_ct`) |
| B5 (16) | xtb (`xtb_slice.xtb_sp`) | fragment B moved ±0.05 and ±0.10 Å along the forming-bond direction: slope, curvature and end values of E_int, Pauli, OI, elst |
| B6 (6) | ORCA `ts.hess`, xtb `--opt` | imaginary frequency, mode share and asynchronicity; xtb reaction energy; TS/product forming-bond ratios |

- **Quadrupole convention (B3).** xtb stores traceless atomic quadrupoles in the Buckingham form Θ = ½ Σ q (3rr − r²1), as in GFN2's anisotropic electrostatics, so the charge–quadrupole energy is q R·Θ·R / R⁵. The json component order (xx, xy, yy, xz, yz, zz) is verified per structure by the trace test (`qc_b3_quad_order`).
- **Product (B6).** The xtb optimisation starts from the Coley (DFT) product. That is the same information level as the G1 references. A G2 run would have to use the autodE product instead. The TS → product atom map is asserted against the Coley start product: the mapped forming pairs must be bonded there.
- **Per-reaction engine gates (B1).** Every reaction must pass B-0 gates 1–3 on fA, fB and the TS (tblite − xtb energy < 1e-6 Eh against the G1 parquet, overlap block, CᵀSC = I), or B1 fails for it.
- **Self-checks** are stored as `qc_*` columns. A non-finite value fails its block, so a row with `ext_status` ok has no NaN. The merge turns any such failure among the `rows_rev4` reactions into a STOP (spec B-7 NaN gate).
- **Cache.** `$R5_SCRATCH/ext/<rid>/{geom,B1..B6}.json`, keyed by the code version and a fingerprint of the G1 inputs. A rerun recomputes only what is missing.

## Models, rows, targets (pre-registered, `results_rev4/PREREG_REV4.md`)

- **Protocol A only** (as rev 3):
  - 80/10/10 splits over seeds 22/23/14/1/2 (`split_80_10_10`).
  - Nested per-seed GridSearchCV (5-fold).
  - StandardScaler on X, y standardised (`TransformedTargetRegressor`).
  - Models: Ridge, KRR(RBF), SVR(RBF), XGB.
- **Rows:** `results_rev4/rows_rev4.csv`, **n = 4,839** = G1 ok (4,860) ∩ no NaN (ESPLEY73 + 9 targets) ∩ hygiene (d1 ≥ 0, d2 ≥ 0, d2 ≤ 50). `make_rows_rev4.py --check-g1-only` rebuilds them from the G1 parquet alone and asserts equality (`results_rev5/A_rows_check.json`).
- **Targets (9):** barrier, d1, d2, elst, pauli, oi, disp, cpcm, cds.
- **Headline: G1 · ESPLEY73 · KRR(RBF)**, fixed a priori.

## Pipeline (submit from this directory of the checkout; every script sources `r4_env.sh`)

```
r4_g1_smoke.sh        Phase 1-1  G1 engine smoke test (5 rxns)                       -> $G1_ROOT/_smoke
r4_g1_array.sh        Phase 1-2  G1 geometries, 18 slices as 9 x 2      (g1_geom.py) -> $G1_ROOT/<rid>/
r4_g1_summary.sh      Phase 1-2  success rate, failure types, RMSD                   -> results_rev4/g1_geometry_summary.*
r4_feat_all.sh        Phase 2    GEOM=g1|g2: 18 slices + aggregate (or r4_feat_array.sh / r4_feat_aggregate.sh)
r4_make_rows.sh       Phase 3-1  pre-registered rows (MODE=check: G1-only row check)  -> results_rev4/rows_rev4.*
r4_ml_smoketest.sh    Phase 3    compile / import / rows gate (no training)
r4_ml_array.sh        Phase 3-2  element i = target i (0-8) on G1; refuses unless PREREG is committed
r4_ml_aggregate.sh    Phase 3-4  ml_report / predictions / figures (G1) + rev4_tables.py -> results_rev4/rev4_*
r4_downstream.sh      Phase 3-3  GEOM=g1: analyze_extra.py + evaluate_pairs.py        -> results_rev4/downstream_g1/
r4_compare_prep.sh    Phase 4    compare_espley.py prep (Espley side + G1)
r4_compare_train.sh   Phase 4    0-6 ours (G1 x 7 targets), 7-8 Espley role d1/d2 (their protocol / our pipeline)
r4_compare_plot.sh    Phase 4    intersection scoring (3,327 rows), tables, figures   -> results_rev4/espley_compare_*
r4_ablation.sh        Phase 4-3  geometry-information ablation, arms A, C, C0, B_g1, O_g1 (ABL_ROWS=common|own)
r4_ablation_aggregate.sh                                                             -> results_rev4/espley_geometry_ablation*
r4_bath_am1.sh        Phase 4-4  start structure of Espley's AM1 TS optimisations (HTTP range reads of BATH-01480)
r4_pack.sh            run several elements of an r4 array script in one allocation (one queue slot)
r5_phaseA.sh          rev 5 A    (r5_phase_a.py, list and pattern in r5_phaseA_discard.txt) discard record + scratch
                                 deletion, G1-only row check, rev 4 tables / Espley comparison / ablation regenerated and
                                 checked against 740ba895, grep check                 -> results_rev5/A_*
r5_ext_smoke.sh       rev 5 B-0  (ext_features.py smoke) tblite 0.7.0 vs xtb 6.7.1 gates (energy, overlap block,
                                 C^T S C = I), parser self-checks, every block on SMOKE_RXNS with G1 structures
                                 + one charged rxn per (q1, q2) pattern, core-h/rxn; always from scratch;
                                 exit 3 = STOP                                         -> results_rev5/B0_smoke.json
r5_ext_array.sh       rev 5 B    (ext_features.py slice) blocks B1..B6 for the G1-ok rxns, 18 interleaved slices as
                                 9 x 2, 16 single-threaded workers, per-rxn cache $R5_SCRATCH/ext/<rid>/
                                                                                     -> $R5_SCRATCH/ext/slices/slice_XX.parquet
r5_ext_merge.sh       rev 5 B-7  (r5_ext_merge.py) merge + gates (rows_rev4 failure share > 1 %, any NaN feature in
                                 a rows_rev4 rxn, B5 δ = 0 gate); report always written
                                 -> $ESPLEY_OUT/xtb_features_ext_g1.parquet,
                                 results_rev5/{rows_rev5.csv, B7_report.json, B7_feature_quantiles.csv}
r5_lockbox.sh         rev 5 C-1  (select_blocks.py lockbox) 15 % lockbox / 85 % dev split of rows_rev5.csv, never
                                 rewritten                  -> results_rev5/{lockbox_ids.csv, dev_ids.csv, lockbox.json}
r5_select.sh          rev 5 C-2/3 (select_blocks.py select) dev-CV KRR of BASE, BASE+Bk, EXT_ALL (9 targets) + forward
                                 block selection on elst/Pauli/OI NMAE, one allocation, units cached in
                                 $R5_SCRATCH/select; refuses unless PREREG_REV5a.md (with the lockbox_ids.csv
                                 sha256), lockbox_ids.csv, dev_ids.csv and lockbox.json are committed
                                 -> results_rev5/C2_*, prereg_rev5b.json (EXT_SEL)
r5_lockbox_eval.sh    rev 5 D-1  (final_eval.py) dev -> lockbox, BASE / EXT_SEL / EXT_ALL x KRR (headline), Ridge,
                                 SVR, XGB x 9 targets, bootstrap CIs + paired differences; refuses unless PREREG_REV5b
                                 is committed and no Phase C cache holds a lockbox id -> results_rev5/D1_lockbox*
r5_espley.sh          rev 5 D-3/D-4 (espley_rev5.py d3a|d3b|d4|all) Espley comparison, one allocation, units in
                                 $R5_SCRATCH/espley: d3a Espley split (labelled ∩ rows_rev5), ESPLEY46 / ESPLEY73 /
                                 EXT_SEL KRR vs Espley SVR and the strongest Espley-side model, Nadeau-Bengio;
                                 d3b lockbox head-to-head vs their protocol (SVR, KRR), paired bootstrap; d4 learning
                                 curves (after d3a); refuses unless PREREG_REV5b is committed and no Phase C cache
                                 holds a lockbox id (exit 4 on a leak, 3 on a stale unit cache)
                                 -> results_rev5/{D3a_*, D3b_*, D4_learning_curves*}
r5_figures.sh         rev 5 E    (figures_rev5.py [--only figN,..] [--force]) paper figures fig1..fig8 from C2_*, D1_*,
                                 D3a_*, D3b_*, D4_* (every value read from those files and cross-checked against the
                                 finer-grained ones); a figure with a missing / inconsistent input is skipped (exit 1),
                                 an up-to-date one is not redrawn -> results_rev5/figures/<name>.{pdf,png,csv}
                                 + figures_rev5_manifest.json
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

Scratch outputs go under `/gpfs/tmp_cpu2/yeseo1ee/{espley_xtb_g1, espley_xtb, espley_rev4, espley_rev5}`.

## Compute

- **G1 geometries.** 216 core-h for 5,260 reactions.
  - ORCA OptTS+Freq takes 0.2–3 min per reaction with 2 cores (analytic xTB Hessian).
  - The references take about 1 s.
  - Concurrent ORCA runs on one node need `OMPI_MCA_rmaps_base_oversubscribe=1` and no core binding (`r4_env.sh`). 16 runs hit an MPI finalize bus error on shared nodes and were rerun unchanged.
- **Features.** About 0.25 s per reaction per core. One geometry takes about 15 min with 4–8 cores.
- **ML.** One target element takes 3 arms × 4 models × 5 seeds on 8 cores. Pin BLAS to one thread (`OMP_NUM_THREADS=1`), otherwise every GridSearchCV worker spawns one thread per allocated core.
