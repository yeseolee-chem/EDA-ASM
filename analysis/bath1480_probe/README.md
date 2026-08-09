# bath1480_probe — Espley 2024 tt subset analysis

Analysis workspace for the "three_two_cycloaddition" (tt, N=3510) subset of
the Bath 1480 dataset (Espley et al. 2024, DOI: 10.1039/D4DD00224E).

Everything visible in the git tree here mirrors — or is derived from — the
scratch workspace at `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/`
(that scratch dir is 4.3 GB, gitignored).

## Folder map

### `gh_repo/` — Espley GitHub snapshot
Downloaded from https://github.com/the-grayson-group/distortion-interaction_ML
for reproducibility. Includes:
- `diassep.py` — fragment separation from AM1 TS + imag mode
- `tt_ml/hyp_tuning.py`, `ml_analysis.py` — training pipeline
- `tt_ml/manual_tt_solvent.pkl` (1.5 MB, 46-feature ML input, 3510 rxns)
- `tt_ml/hps.pkl` — Espley's tuned hyperparameters
- `tt_ml/cleaned_results.csv` — paper's target MAE values (25-seed mean±σ)
- `tt_solvent_features.pkl` (38 MB, 300-feature pre-selection)

### `repro_results/` — Our reproduction of Espley's paper
See [`hps_comparison.md`](repro_results/hps_comparison.md) and
[`final_our_vs_paper.csv`](repro_results/final_our_vs_paper.csv).

Key result: **sklearn 100% HP match to Espley, all 16 model×target combos
match paper test MAE to 0.01 kcal/mol** (5-seed reproduction, we tuned
hyperparameters from scratch and reran ml_analysis.py with them).

- `our_hps.pkl` — our re-tuned HPs (Espley protocol, seed=23)
- `our_vs_paper.csv` — using Espley's hps (baseline validation)
- `final_our_vs_paper.csv` — using OUR hps (full reproduction test)
- `hps_comparison.md` — parameter-by-parameter Espley vs us diff
- `audit_tt.json` — 3980 total rxn → 3662 passed audit (all 5 files present)
- `our_all_results.csv` — per-seed raw test MAE per model

### `kisti_transfer/` — What to copy to KISTI
This is the **KISTI-ready package** for running our own ORCA EDA-NOCV labeling
on the 3504-reaction subset.

- **`tt_eda_kisti_package.tar.gz`** (7 MB) — the file to `scp` to KISTI
- `README.md`, `CLAUDE.md`, `VERSION.md` — same files inside the tar
  (surfaced here for easy reading before/during transfer)
- `kisti_manifest.parquet` — per-reaction metadata (rxn_id, ts_coords,
  fragment indices, imag freq, etc.) for 3,504 reactions
- `kisti_final_ids.json` — canonical rxn_id list
- `scripts/` — copies of the sbatch templates + aggregator + verifier

Read [`kisti_transfer/CLAUDE.md`](kisti_transfer/CLAUDE.md) first when a
fresh Claude session starts at KISTI — that file has the complete
workflow, HPC rules, and completion criteria.

## Non-tracked scratch workspace

`/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/`:

| Item | Size | Purpose |
|---|---|---|
| `data_archive_files.zip` | 4.1 GB | Bath 1480 original (MD5 verified) |
| `tt_eda_kisti_package.tar.gz` | 6.8 MB | KISTI transfer artifact |
| `tt_kisti_package/` | 13 MB | Staged package (3504 rxn eda.inp + docs + scripts) |
| `tt_repro/` | 57 MB | Espley reproduction workspace (5-seed ml results per seed dir) |
| `tt_hpt/` | 125 MB | Our HP re-tuning workspace (checkpoint + per-target NN pkls + final_ml_results) |
