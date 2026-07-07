# SPEC_02 — b / δ decomposition (m3, 24-d, 787-rxn cohort)

Per `SPEC_02_b_delta_decomposition.md`. Compares three variants that
share arch / HP / split / seeds:

| variant | b | δ | notes |
|---|---|---|---|
| **M_b**  | ridge (α=1) on 24-d | ≡ 0 | analytic, per-fold |
| **M_δ**  | ≡ 0 | full m1-delta network | retrained (this SPEC) |
| **M_bδ** | ridge (α=1) on 24-d | m1-delta | reuses m3 checkpoints |

M_δ is the only variant requiring new training. 5 folds × 5 members =
**25 cells**, all done via `spec02_train_delta.sh` as an sbatch array.

## Reproduce

```bash
# 1) 25-cell training array (5 folds × 5 members, %3 concurrent)
sbatch results/spec02_bdelta/code/spec02_train_delta.sh
# 2) aggregate + figures + tables
sbatch results/spec02_bdelta/code/spec02_aggregate.sh
```

Inputs (identical to SPEC_01/03/04):
- Bundle: `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v1/features_v6_delta_m3.pt`
- Splits: `pipeline_rebuild/spec_v1/artefacts/subsamples_v1/trackB_no_ood/`
- Seeds: 42 (fold) + 42 + fold·100 + member·17 (RNG)
- Training budget: EPOCHS_MAX = 100 000, PATIENCE = 10 000, batch 16, lr 1e-5, wd 1e-3

Runner is idempotent — resubmitting skips any `member{M}.json` that already exists.

## Contents

```
results/spec02_bdelta/
├── code/
│   ├── spec02_delta_runner.py     M_δ retraining (b ≡ 0)
│   ├── spec02_train_delta.sh      SLURM array (%3 throttle)
│   ├── spec02_aggregate.py        3-variant metrics + figures
│   └── spec02_aggregate.sh        SLURM aggregate submitter
├── m_delta/
│   └── fold{0..4}/
│       ├── member{0..4}.json      y_true, y_pred, HP, per-channel + barrier metrics
│       └── member{0..4}.ckpt.pt   torch state_dict (only for cells trained after ckpt patch)
├── decomposition_metrics.csv      per fold × member × variant × channel
├── decomposition_summary.csv      mean ± std across cells
├── family_breakdown.csv           NMAE per family (dipolar / rgd1 / qmrxn20)
├── cancellation.csv               ρ = corr(Σb, Σδ)  per variant
├── variance_decomposition.csv     Var(y) = Var(b) + Var(δ) + 2·Cov(b,δ)
├── contribution_bars.png          NMAE stacked / grouped per channel × variant
├── parity_3models.png             barrier parity for M_b / M_δ / M_bδ
└── summary.md                     headline table
```

## Weights

- **M_b (ridge)** is analytic: `W*(1.0) = (XᵀX + Ĩ)⁻¹ Xᵀy` (Ĩ zeroes
  intercept diagonal). Rebuilt on-the-fly by `spec02_aggregate.py`;
  no separate weight artefact required.
- **M_δ (b≡0)** ckpts (state_dict) are saved to
  `m_delta/fold{F}/member{M}.ckpt.pt` for cells trained after the
  ckpt-save patch. Earlier cells still reproduce their JSON metrics
  byte-for-byte because the runner is fully seeded.
- **M_bδ (m3)** reuses the m3 checkpoints under `m3/code/…`.

## Reload an M_δ checkpoint

```python
import json, torch
from pathlib import Path
import sys

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "src"))
from eda_asm.asr_v1.models_delta import ModelM1Delta

fold, member = 3, 0
ckpt = torch.load(REPO / f"results/spec02_bdelta/m_delta/fold{fold}/member{member}.ckpt.pt",
                  map_location="cpu", weights_only=False)
model = ModelM1Delta(d_model=128, n_heads=4, head_hidden=64, dropout=0.2)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

The JSON alongside the ckpt has `y_true`, `y_pred`, per-channel + barrier
metrics — sufficient to reproduce every number in
`decomposition_metrics.csv`.

## Headline (summary.md)

Per-channel mean NMAE across the 25 cells:

| channel | M_b | M_δ | M_bδ |
|---|---|---|---|
| strain | 0.835 ± 0.032 | 0.797 ± 0.065 | 0.798 ± 0.051 |
| Pauli  | 0.879 ± 0.046 | 0.691 ± 0.049 | 0.755 ± 0.051 |
| V_elst | 0.909 ± 0.049 | 0.772 ± 0.082 | 0.765 ± 0.068 |
| oi     | 0.904 ± 0.096 | 0.705 ± 0.043 | 0.797 ± 0.051 |
| disp   | 0.286 ± 0.015 | 0.397 ± 0.218 | 0.286 ± 0.015 |
| barrier| 0.538 ± 0.031 | 0.844 ± 0.151 | 0.567 ± 0.036 |

Barrier-level cancellation ρ = corr(Σb, Σδ): M_b 0.083, M_δ 0.133, M_bδ 0.098.
