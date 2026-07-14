# CLAUDE.md — eda-asm-prediction

## HPC ground rules (READ FIRST — non-negotiable)

- **ALL non-trivial compute MUST run via `sbatch` on compute nodes.**
  The login node `gate1.hpc` is **read-only** for us. No `python`,
  no `xtb`, no matplotlib rendering, no MACE feature extraction on
  login. Even a 15-row Hammett plot goes through sbatch.
- **Every `sbatch` submission MUST use `#SBATCH --time=48:00:00`.**
  No shorter walltimes. If the underlying work only needs a few
  minutes, still ask for 48h — the extra time costs nothing while a
  too-tight walltime silently kills long tails.
- **All work must be idempotent on resubmit.** Each cell / shard /
  batch must (a) write its output atomically and (b) skip work whose
  output file already exists. If a 48h wall clips a partial cell,
  losing that cell is acceptable — losing the *rest* is not.
- **If a job hits the 48h wall, just re-`sbatch` the same script.**
  Idempotency + `if out_path.exists(): return` in the runner makes
  this safe.
- **We may hold up to 10 concurrent SLURM jobs.** When designing
  arrays or multi-model runs, distribute across partitions so total
  concurrency stays ≤ 10. Prefer parallel across gpu3 / gpu4 / gpu5
  over serialising on one partition.

## Project overview

EDA (Energy Decomposition Analysis) + ASM (Activation Strain Model)
proxy prediction of 5-channel decomposed activation energies. Repo
covers Stage 5 (label pipeline) + Stage 6 (Δ-learners over
MACE-OFF23 features).

Folder/distribution name: `eda-asm-prediction` (hyphenated).
Python import name: `eda_asm`.

## Repository layout (post-cleanup, 2026-07-03)

```
labels/                       789-reaction ADF + ORCA EDA-ASM labels + seed selection
V1/                           Claisen 15-substrate ASR-EDA (spec + runs + Hammett analysis)
models/                       parent for delta-learner deliverables
  m1/  m2/  m3/               Δ-learner code + frozen cells + per-model figures/results per baseline
  comparison/                 cross-model aggregates (current: comparison/v9/)
src/eda_asm/                  canonical shared package
  asr_v1/                     model / backbone / training / baseline_physics
  datasets/                   dipolar_cycloaddition, qmrxn20 loaders
  adf/                        ADF input builder + parser (legacy)
  phase1/                     Halo8 sampling + fragment definition (legacy)
  stage5a/                    ADF fragmentation pipeline (legacy)
scripts/                      pipeline utilities (asr_v1 caching + trainers,
                              screen_substituents.py, ...)
pipeline_rebuild/spec_v1/     current spec-compliant rebuild (2026-07-03)
reports/fragment_screen/      substituent decomposition (BRICS + Bemis-Murcko)
backbone_ft/                  MACE-OFF23 fine-tune experiment (gitignored)
```

## Datasets — where the geometries come from

The 789-reaction cohort spans four families. Raw data lives under
`/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/` (regenerable, gitignored).

| family | count | source | archive | local extraction |
|---|---|---|---|---|
| dipolar | 193 | Stuyver / Jorner / Coley 2023 | figshare 21707888 v5 | `dipolar_cycloaddition/extracted/full_dataset_profiles/{idx}/` |
| qmrxn20_e2 | 200 | von Rudorff 2020 | materialscloud 2020.55 (uuid `gkqvy-3vp74`) | `QMrxn20/transition-states/e2/{label}.xyz` + friends |
| qmrxn20_sn2 | 196 | von Rudorff 2020 | (same) | `QMrxn20/transition-states/sn2/{label}.xyz` |
| rgd1 | 200 | Zhao & Savoie 2023 | figshare 21066901 v6 | `rgd1/RGD1_CHNO.h5` (per-reaction R/TS/P extracted to `extracted_xyz/{rid}/{R,TS,P}.xyz`) |

Cohort membership: `labels/adf/adf_labels_v6_multifamily.parquet`
(reaction_id column). Original selection artefacts (seed=42, Morgan-r2
+ Kennard–Stone) at `labels/seed_selection/`.

## MACE-OFF23 backbone

Cached locally at `/home1/yeseo1ee/.cache/mace/MACE-OFF23_medium.model`
(also small/large). Feature extraction goes through
`src/eda_asm/asr_v1/backbone_maceoff.py:MACEOFFFeatureExtractor`.
Per-atom invariant features, 256-d, float32.

Precomputed R/TS/P features for all 789 reactions live at
`/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium/{rid}.pt`
(regenerable via `pipeline_rebuild/stage2_mace_features.py`).

## Spec-compliant model architecture (m1 / m2 / m3)

All three share the same model + training loop; they differ only in
the physics descriptor vector fed to the ridge baseline.

```
MACE-OFF23 features (256-d per atom, per state)
  → InputStandardizer  (fit on train fold R+P only, TS excluded)
  → SiLU + Dropout linear projection to 128-d
  → 4 × Cross-attention blocks (LayerNorm-stabilised, 4 heads × 32-d):
        h′(TS) = LN(h(TS) + ½ [CA_θ1(h(TS),h(R)) + CA_θ2(h(TS),h(P))])
        h′(R)  = LN(h(R)  + CA_θ3(h(R), h(TS)))
        h′(P)  = LN(h(P)  + CA_θ4(h(P), h(TS)))
  → AttentionPool per state (learned query q_s, mask padding)
  → z = [v_R || v_TS || v_P || v_TS-v_R || v_TS-v_P || v_P-v_R]  (768-d)
  → MLP 768 → SiLU → 64 → SiLU → 64 → 5   (residual δ)

Physics baseline b:  ridge (α=1) over z-score(d1..d_D) with intercept.
Prediction:          ŷ = b + δ
Loss:                mean over batch of  mean_c |ŷ_c − y_c| / σ_c
                     (σ_c = per-channel std of train-fold labels)
Optimiser:           Adam, lr = 1e-5, weight_decay = 1e-3
Regularisation:      grad-clip 5.0, dropout 0.2
Budget:              EPOCHS_MAX = 100 000, PATIENCE = 10 000, batch = 16
```

### Physics descriptors per model

| model | dim | descriptors |
|---|---|---|
| m1 | 6 | d1..d6 — Kabsch RMSD × 2, Pauli/elst/disp pair-sums at TS, n_atoms |
| m2 | 21 | d1..d6 + d7..d21 (GFN2-xTB energies, dipoles, HOMO/LUMO, fragA charge sum) |
| m3 | 24 | d1..d21 + d22 = μ²/2η (Parr ω), d23 = Σq², d24 = Σ|WBO_{a∈A,b∈B}| |

xTB descriptors come from three single-points at the TS geometry:
complex, fragA, fragB. Fragment partition uses:
- dipolar: atom-mapped SMILES + RDKit subgraph match on TS connectivity
- qmrxn20 e2/sn2: connected components on R (R & TS share atom order)
- rgd1: connected components on R (same)

## Current spec-v1 rebuild pipeline

Everything under `pipeline_rebuild/spec_v1/`. All sbatch scripts use
`--time=48:00:00`; all output goes to
`/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/`.

| stage | script | what it does | output |
|---|---|---|---|
| 1 | `stage1_download.sh` + `stage1b_rgd1.sh` + `stage1d_qmrxn20_fixed.sh` | download source archives (dipolar, RGD1, QMrxn20) | raw XYZs under `/gpfs/tmp_cpu2/.../eda_asm_raw/` |
| 2 | `stage2_mace.sh` | MACE-OFF23_medium features per (R/TS/P) per reaction | `mace_off23_medium/{rid}.pt` |
| 3 | `stage3_array.sh` (8-way shards) | fragment partition + GFN2-xTB descriptors d1..d24 | `descriptors_v1.parquet` |
| 4 | `stage4.sh` | assemble m1/m2/m3 bundles + 5-fold stratified splits | `bundles_v1/features_v6_delta_{m1,m2,m3}.pt`, `subsamples_v1/trackB_no_ood/fold*/…` |
| 5 | `stage5_train_{m1,m2,m3}.sh` | 5×5 CV arrays on gpu3/gpu4/gpu5 (%3 each = 9 concurrent) | `models/m{1,2,3}/code/trackB_lowlr_no_ood_*/m{1,2,3}_delta/fold*/member*.json` |
| 6 | `stage6_aggregate.py` | 3-way NMAE / RMSE bar + parity grid + REPORT.md | `models/comparison/spec_v1/{figures,results,REPORT.md}` |

Idempotency contract:
- Stage 2/3 shards check `_progress.jsonl` / existing rows before recompute.
- Stage 5 runners skip a cell if the target `member{M}.json` exists.
- If a 48h wall clips a shard/cell, re-`sbatch` the same script.
  Anything already written is preserved; only the interrupted cell restarts.

## Critical gotchas from the current rebuild (2026-07-03)

- **tblite must be imported before torch.** Torch ships its own
  `libgomp` without the `GOMP_5.0` symbol that tblite needs. Order:
  `from tblite.interface import Calculator` at module top, then
  everything else. Stage 3 dies with `ImportError: tblite C extension
  unimportable` if this is violated.
- **`tblite.Calculator.add()` does NOT accept "bond-orders" or "dipole".**
  Only interaction terms (`electric-field`, `alpb-solvation`, …).
  Bond orders and dipole are returned by `singlepoint()` by default.
  Calling `add("bond-orders")` silently corrupts state so that dipole
  is not computed → `TBLiteValueError: Molecular dipole was not
  calculated`.
- **QMrxn20 e2/sn2 products lose LG + proton atoms.** `d2 = kabsch_rmsd(P, TS)`
  needs matching atom counts. Fallback: substitute TS for P when
  `len(P) != len(TS)`, effectively setting d2 → 0 for those ~396
  reactions. Documented in `stage3_xtb_and_descriptors.py`.
- **SIZE_FULL = 509** is hardcoded in `runner_lowlr_trackB_m1delta.py`
  (spec convention). Stage 4 must therefore emit `size_509.json` in
  each fold dir even when the train pool is larger (subsample with a
  per-fold RNG seed).
- **Standardizer scope**: `InputStandardizer.fit_from()` on train R+P
  only. TS is excluded per spec. The restored code included TS by
  mistake — corrected in `training_delta.py` (2026-07-03).
- **Loss**: σ_c-normalised L1, not raw `F.l1_loss`. σ_c comes from
  train-fold labels, not global.

## V1 Claisen 15-substrate ASR/EDA

Independent side project under `V1/`. 15 para-substituents on a vinyl
allyl ether, wB97X-3c geometry + ZORA-BLYP-D3(BJ)/TZ2P NOCV-EDA.
Frozen 15-row parquet at `V1/outputs/v1_claisen_asr.parquet`.

Downstream Hammett analysis under `V1/analysis/`:
- `hammett_plot.py` — σₚ regressions per EDA channel + Swain–Lupton fit.
- `submit_hammett.sh` — sbatch (48h) to cpu2.
- Frozen figures + results/CSV committed.

Headline: total ΔE‡ vs σₚ has R² ≈ 0.03 (barrier flat in σₚ), but
individual EDA channels (Pauli, V_elst, strain) each correlate
strongly with σₚ (R² ≈ 0.4–0.5) with signs that cancel in the total.
This is the argument for treating EDA channels as separate features.

## MACE-OFF23 fine-tune (`backbone_ft/` — gitignored)

Live experiment fine-tuning MACE-OFF23_large on Halo8 R/TS/P frames.
Uses:
- Foundation model: `/gpfs/tmp_cpu2/yeseo1ee/halo8_ft/foundation/MACE-OFF23_large.model`
- Splits + XYZs under `/gpfs/tmp_cpu2/yeseo1ee/halo8_ft/`
- Scripts: `backbone_ft/scripts/run_ft_partial_freeze.py` (partial-freeze
  wrapper around `mace_run_train`).
- SLURM submitter: `backbone_ft/configs/slurm_ft.sh` — 48h, gpu3/4/5.

**Not part of the m1/m2/m3 deliverable.** Kept out of git via
`.gitignore` rule `/backbone_ft/`. Restart from the last MACE
checkpoint if the 48h wall trips.

## Halo8 reaction trajectories (legacy)

Source DBs deleted 2026-06-12 to free quota (memory:
`halo8_data_deleted.md`). The `data/Halo8/` symlink is broken; the
dataset is no longer available on this cluster. Fine-tune uses the
pre-processed XYZ splits at `/gpfs/tmp_cpu2/yeseo1ee/halo8_ft/`.

## Environment

- Conda env: `reactot` (Python 3.10, torch 2.2.1, mace, nequip,
  tblite, rdkit, ase, pandas, matplotlib).
- Activation: `source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh;
  conda activate reactot`. Every sbatch script does this.
- QM tools: ADF/AMS at `$HOME/ams2026.103/` (V1 only), ORCA 6.1.1
  at `$HOME/orca_6_1_1_avx2/` (V1 only), GFN2-xTB via `tblite`
  Python API (m2/m3).

## Conventions

- Random seeds always come from the config (default seed = 42 for
  fold generation and RNG in stratification).
- All energies stored in **kcal/mol** unless the column name says
  otherwise (`_Eh` = hartree, `_h` = hartree).
- Geometries in Å.
- Never commit `/gpfs/tmp_cpu2/yeseo1ee/...` outputs. Bundles + raw
  descriptors + logs live there and are gitignored.
- Fresh per-cell training outputs (`m{1,2,3}/code/trackB_*/`) are
  gitignored until a curated aggregation step promotes them.
