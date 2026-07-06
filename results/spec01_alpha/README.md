# SPEC_01 — Ridge α optimization (m3 physics baseline, 24-d)

Per `SPEC_01_ridge_alpha_optimization.md`. Determines the CV-optimal
ridge penalty α on the 24-d m3 descriptor matrix and quantifies how
sensitive the baseline `b` is to that choice.

## Reproduce

```bash
# From repo root (compute node via sbatch):
sbatch results/spec01_alpha/code/spec01.sh
# or, if the source-of-truth script under pipeline_rebuild/ is preferred:
sbatch pipeline_rebuild/spec_v1/spec01.sh
```

Both scripts point at the same inputs:
- **Bundle**: `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1/features_v6_delta_m3.pt`
  (also available at `pipeline_rebuild/spec_v1/artefacts/bundles/features_v6_delta_m3.pt`)
- **Fold splits**: `pipeline_rebuild/spec_v1/artefacts/subsamples_v1/trackB_no_ood/`
- Seed = 42; 5-fold CV within each fold's train pool; α ∈ logspace(−6, 4, 61)

## Contents

| file | description |
|---|---|
| `code/spec01_alpha.py` | analysis script |
| `code/spec01.sh` | SLURM submitter (cpu1/cpu2, 48h) |
| `alpha_curves.png` | 6-panel CV NMAE curves + α* marker |
| `ridge_trace.png` | β̂(α) coefficient paths per channel |
| `alpha_selection.csv` | test NMAE/RMSE/R²/slope at α ∈ {≈0, 1, α*} |
| `summary.md` | α* (CV) per channel + rank/cond |

## "Weights"

Ridge is analytic: `W*(α) = (XᵀX + α·Ĩ)⁻¹ Xᵀy` (Ĩ zeroes the intercept
diagonal). Given the CSV lists α* per fold + channel and the bundle is
committed, refitting gives byte-identical coefficients — no separate
weight artefact needed.
