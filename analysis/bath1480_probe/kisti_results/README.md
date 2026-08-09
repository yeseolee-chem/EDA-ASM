# tt EDA-NOCV Labeling Package

**Task**: Run ORCA 6.x EDA-NOCV calculations on 3,662 pre-built inputs for the
[3+2] cycloaddition (tt) subset of the Bath 1480 dataset (Espley et al. 2024,
DOI: 10.1039/D4DD00224E), then produce a 5-channel decomposition parquet.

If you are Claude Code / an AI agent: **read `CLAUDE.md` first** — it has the
exact workflow, HPC rules, and completion criteria.

## Quick summary

| Item | Value |
|---|---|
| N reactions | 3,662 |
| Method | B3LYP-D3(BJ) / def2-TZVP + CPCM(SMD, water) |
| Software | ORCA 6.x |
| Per-reaction wall (est.) | 20-40 min on 4-8 cores |
| Total compute (est.) | 1,500-3,000 core-hours |
| Package size | ~30 MB (inputs only) |
| Output size (est.) | ~500 MB (aggregated), ~50 GB (raw eda.out) |

## Origin

Built on companion HPC (Aug 2026). Source: 
`data_archive_files.zip` (4.0 GiB, SHA1/MD5 in `VERSION.md`) from
Bath research data archive (https://doi.org/10.15125/BATH-01480, CC-BY).

Filter applied: only reactions with all 5 essential Espley files present:
- `optimisations/am1_ts/ts_<N>.out` (AM1 TS)
- `splits/dft/ts_<N>_reactant_1.out` (fragment 1 DFT opt+freq)
- `splits/dft/ts_<N>_reactant_1_SPE.log` (fragment 1 TZVP SPE)
- `splits/dft/ts_<N>_reactant_2.out` (fragment 2 DFT opt+freq)
- `splits/dft/ts_<N>_reactant_2_SPE.log` (fragment 2 TZVP SPE)

Additional filters implicit in `data/manifest.parquet`:
- `partition_matches_espley == True` (diassep fragment count matches Espley)
- exactly 1 imaginary frequency at AM1 TS level

## Layout

See `CLAUDE.md` § 3.

## Human execution

Prerequisites:
- ORCA 6.x (or ≥ 5.0.4 for SMD support)
- OpenMPI (if running ORCA parallel)
- Python 3.10+ with `pandas`, `pyarrow`, `numpy`, `cclib` (see `env/requirements.txt`)
- SLURM cluster with sbatch

Fast path:
```bash
tar xzf tt_eda_kisti_package.tar.gz
cd tt_eda_kisti_package/
bash env/setup_env.sh              # create conda env
# Edit scripts/sbatch_run_orca.sh — set ORCA_BIN + partition + queue caps
sbatch --array=0 scripts/sbatch_run_orca.sh   # smoke test one reaction
# Wait ~30 min, verify data/rxn_0000/eda.out has "ORCA TERMINATED NORMALLY"
sbatch scripts/sbatch_run_orca.sh             # full run
# Wait days
sbatch scripts/sbatch_aggregate.sh            # aggregate 5-channel parquet
sbatch scripts/sbatch_verify.sh               # cross-check vs Espley reference
```

Outputs land in `results/`.

## Reproducibility

- `data/manifest.parquet` contains everything needed to rebuild the ORCA
  inputs from scratch (given the source Bath 1480 zip):
  - `rxn_id`, `symbols_str`, `ts_coords_{x,y,z}`, `frag_A_indices`, `frag_B_indices`
- The build scripts (`scripts/build_full_manifest.py`, kept here for
  transparency) can regenerate `data/rxn_XXXX/eda.inp` deterministically.

## License

Data (from Bath 1480): CC-BY 4.0 (see original archive).
Code (this package): MIT (or attribution to build-time repo).

## Contact

Origin: eda-asm-prediction repo, yeseolele@gmail.com.
