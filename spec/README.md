# SPEC 01-05 (m3, v9 783-rxn cohort)

Self-contained folder with the spec analyses on the current v9 m3 pipeline.

## Layout

```
spec/
  spec01_alpha/          Ridge alpha optimization (per-channel)
    code/  results/  figures/
  spec02_bdelta/         b / delta contribution decomposition
    code/  checkpoints/direct/  results/  figures/
  spec03_bmax/           Max classical baseline (ridge/lasso/enet/xgb)
    code/  results/  figures/
  spec04_descriptors/    OLS / VIF / forward-selection / lasso path / heatmap
    code/  results/  figures/
```

## Data source (shared)

- Bundle: `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt`
- Splits: `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9/fold{0..4}/`
- Labels order in bundle: `[strain, Pauli, elst (V_elst), oi (E_orb), disp]` (kcal/mol)

## Dependency chain

```
SPEC_01 (CPU, minutes)                     alpha* used by SPEC_02 & SPEC_03
      |
      v
SPEC_04 (CPU, minutes)   parallel with SPEC_01
SPEC_03 (CPU, min-hour)  parallel with SPEC_01
      |
      v
SPEC_02 direct training (GPU, hours)  needs m3 v9 bundle
      |
      v
SPEC_02 aggregate (CPU, seconds)      needs m3 v9 bdelta cells + direct cells
```

Each script is idempotent (skip-if-output-exists). All sbatch scripts use
`--time=48:00:00` per CLAUDE.md HPC rules.
