# SPEC_04 — descriptor contribution + parsimony (m3, 24-d)

Per `SPEC_04_descriptor_contribution.md`. Classical inference, VIF,
forward-selection saturation, Lasso path, and cross-channel importance
heatmap on the 24 m3 descriptors.

## Reproduce

```bash
sbatch results/spec04_descriptors/code/spec04.sh
```

Uses the same bundle + fold splits as SPEC_01/02/03:
- Bundle: `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1/features_v6_delta_m3.pt`
- Splits: `pipeline_rebuild/spec_v1/artefacts/subsamples_v1/trackB_no_ood/`
- Uses only fold-0 train pool for classical inference tables.
- Seed = 42 (KFold for forward-selection).
- Note: OLS is via a scipy-only implementation (no statsmodels dep).

## Contents

| file | description |
|---|---|
| `code/spec04_descriptors.py` | inference + parsimony script |
| `code/spec04.sh` | SLURM submitter |
| `dedup_report.md` | true D vs expected + high-correlation drops |
| `vif.csv` | per-descriptor VIF with severe / moderate flags |
| `ols_tables/<channel>.csv` | β̂, SE, t, p, 95% CI per descriptor per channel |
| `saturation_curves.png` | forward-selection val NMAE vs # descriptors |
| `lasso_paths.png` | LARS-Lasso coefficient paths |
| `importance_heatmap.png` | |β̂|-normalised descriptor × channel |
| `reduced_set_proposal.md` | elbow-based core set (ΔNMAE < 0.005) |
| `summary.md` | dedup + VIF + reduced-set headline |

## "Weights"

OLS β̂ per channel (with SE / t / p / CI) are in `ols_tables/*.csv` —
these are the closed-form weights. Refitting the script reproduces them
byte-identically.
