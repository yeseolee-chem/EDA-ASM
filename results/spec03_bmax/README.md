# SPEC_03 — physics-only baseline maximization (m3, 24-d, 787-rxn cohort)

Per `SPEC_03_b_only_maximization.md`. Benchmarks **four** classical
predictors on the 24-d m3 descriptor matrix and compares the best
classical against the full neural m3 (M_bδ).

## Method set

Reduced (per user directive) to a single boosting model + three linear
regularisers:

| id | model | tuned hyperparameters | CV |
|---|---|---|---|
| `ridge` | Ridge (linear)     | α = 1 (fixed)                       | — |
| `lasso` | Lasso (linear)     | λ                                  | 5-fold |
| `enet`  | ElasticNet         | λ + l1_ratio ∈ {.1,.3,.5,.7,.9}     | 5-fold |
| `xgb`   | XGBoostRegressor   | n_estimators, max_depth, learning_rate, subsample, colsample_bytree | 5-fold GridSearch |

## Reproduce

```bash
sbatch results/spec03_bmax/code/spec03.sh
```

The sbatch script installs `xgboost` (via `pip install --user`) on
first run if the reactot env doesn't already have it. Everything else
comes from the standard reactot conda env.

Inputs (identical to SPEC_01/02/04):
- Bundle: `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1/features_v6_delta_m3.pt`
- Fold splits: `pipeline_rebuild/spec_v1/artefacts/subsamples_v1/trackB_no_ood/`
- Seeds: 42 for all stochastic components

## Contents

```
results/spec03_bmax/
├── code/
│   ├── spec03_bmax.py
│   └── spec03.sh
├── baseline_leaderboard.csv  method × channel × {NMAE,RMSE,R²,slope} per fold
├── barrier_routes.csv        Σ-of-channels vs direct barrier per fold + method
├── baseline_bars.png         NMAE per channel, 4 methods side-by-side
├── best_vs_neural.png        best classical vs M_bδ, Δ (pp) annotated
├── summary.md                headline table
└── weights/                  <— fitted models, joblib-serialised
    ├── scaler_fold{0..4}.joblib   per-fold StandardScaler
    ├── ridge/fold{0..4}/{strain,Pauli,V_elst,oi,disp,barrier_direct}.joblib
    ├── lasso/fold{0..4}/…same layout
    ├── enet/fold{0..4}/…same layout
    └── xgb/fold{0..4}/…same layout
```

## Reload a fitted model

```python
import joblib, torch, numpy as np, json
from pathlib import Path

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
bundle = torch.load(str(REPO / "pipeline_rebuild/spec_v1/artefacts/bundles/features_v6_delta_m3.pt"),
                    map_location="cpu", weights_only=False)
D = bundle["descriptors"].numpy()
Y = bundle["labels"].numpy()
rids = {r: i for i, r in enumerate(bundle["reaction_ids"])}

fold = 0
te = np.array([rids[r] for r in json.load(
    open(REPO / f"pipeline_rebuild/spec_v1/artefacts/subsamples_v1/trackB_no_ood/fold{fold}/test_rids.json")
) if r in rids])

sc = joblib.load(REPO / f"results/spec03_bmax/weights/scaler_fold{fold}.joblib")
model = joblib.load(REPO / f"results/spec03_bmax/weights/xgb/fold{fold}/strain.joblib")

Xte = sc.transform(D[te])
y_pred_strain = model.predict(Xte)
```

The scaler + model tuple is sufficient to reproduce every number in
`baseline_leaderboard.csv` byte-for-byte.

## Headline (summary.md)

XGBoost wins on every channel and the summed barrier route:

| channel | best | best NMAE | M_bδ NMAE | Δ (pp) |
|---|---|---|---|---|
| strain | xgb | 0.759 | 0.798 | −3.9 |
| Pauli  | xgb | 0.654 | 0.755 | −10.0 |
| V_elst | xgb | 0.686 | 0.765 | −7.9 |
| oi     | xgb | 0.676 | 0.797 | −12.1 |
| disp   | xgb | 0.249 | 0.286 | −3.7 |
| barrier| xgb | (see summary.md) |
