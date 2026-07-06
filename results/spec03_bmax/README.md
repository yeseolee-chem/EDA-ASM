# SPEC_03 — physics-only baseline maximization (m3, 24-d)

Per `SPEC_03_b_only_maximization.md`. Benchmarks classical predictors
on the 24-d m3 descriptor matrix and compares the best classical against
the full neural m3 (M_bδ).

## Reproduce

```bash
sbatch results/spec03_bmax/code/spec03.sh
```

Uses the same bundle + fold splits as SPEC_01/02/04:
- Bundle: `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1/features_v6_delta_m3.pt`
- Splits: `pipeline_rebuild/spec_v1/artefacts/subsamples_v1/trackB_no_ood/`
- Seed = 42 for all stochastic tuners
- 5-fold CV grid tuning for lasso, enet, KRR, GBM, RF (ridge fixed α=1)

## Contents

| file | description |
|---|---|
| `code/spec03_bmax.py` | benchmark + comparison script |
| `code/spec03.sh` | SLURM submitter (cpu1/cpu2, 48h, 8 cpus) |
| `baseline_leaderboard.csv` | method × channel × {NMAE,RMSE,R²,slope} per fold |
| `barrier_routes.csv` | Σ-of-channels vs direct barrier per fold + method |
| `baseline_bars.png` | NMAE per channel, all 6 methods |
| `best_vs_neural.png` | best classical vs M_bδ (m3), Δ annotated |
| `summary.md` | best-method-per-channel + verdict |

## "Weights"

All tuned hyperparameters are recorded in `baseline_leaderboard.csv`
(implicitly via the metrics — the script logs the exact grid). Fitted
sklearn model objects are **not** serialised, because the script's
deterministic seed + tuned hyperparameters + tuning grid + standardised
inputs reproduce the exact fitted models on rerun. If explicit
persistence is required, augment `spec03_bmax.py:main` with a
`joblib.dump(model, ...)` inside the fit loop.
