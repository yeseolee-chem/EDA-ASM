# m1 — geom6 baseline (Δ-learning, no xTB)

Track B `m1_delta` model with the **geom6** physics baseline (d1–d6
ridge features from geometry only — no xTB descriptors). Trained on the
789-reaction multi-family cohort (`labels/adf/`), no-OOD split.

## Training config

- Model: `ModelM1Delta` (cross-attention + delta-learning)
- Baseline: `geom6` (linear physics baseline; `LinearBaseline` over d1..d6)
- Pool: `trackB_no_ood` (4 OOD strain-channel outliers dropped)
- LR = 1e-5, epochs ≤ 100k, early-stop patience 10k
- 5 folds × 5 members = 25 cells (SLURM array 0..24)

## Layout — self-contained

```
m1/
├── code/
│   ├── runner_lowlr_trackB_m1delta.py    # training entrypoint (BASELINE=geom6)
│   ├── build_v2_bundles.py               # builds features_v6_delta_geom6.pt
│   ├── submit_lowlr_geom6.sh             # SLURM submitter
│   ├── eda_asm/asr_v1/                   # local copy of the shared library
│   │   ├── models_delta.py               # ModelM1Delta (cross-attention Δ head)
│   │   ├── models.py                     # shared InputStandardizer / _AttentionPool /
│   │   │                                 #   sign-constrained head
│   │   ├── backbone_maceoff.py           # frozen MACE-OFF23 feature extractor
│   │   ├── backbone.py                   # NequIP alternative backbone
│   │   ├── baseline_physics.py           # LinearBaseline (ridge over d1..d6)
│   │   ├── training_delta.py             # delta training loop
│   │   ├── data.py / data_multi.py       # dataset + ASR_COMPONENTS
│   │   └── (rtsp variants for completeness)
│   └── scripts/                          # pipeline utilities
│       ├── cache_features_maceoff_delta.py   # bundle builder used pre-training
│       ├── cache_features_maceoff.py
│       ├── train_cv_delta.py                 # historical delta CV trainer
│       ├── train_m1.py                       # original m1 trainer (pre-v2)
│       ├── learning_curve_delta.py
│       └── compare_backbones.py
└── results/
    └── foldF/memberM.json                # per-cell test-set predictions + metrics
```

The runner adds `m1/code/` and `m1/code/scripts/` to `sys.path` first,
so imports resolve to the local copy. If you delete `m1/code/eda_asm/`
it will fall back to the repo-root canonical package at
[`src/eda_asm/asr_v1/`](../src/eda_asm/asr_v1/).

## Reproduce

```bash
# from repo root
sbatch m1/code/submit_lowlr_geom6.sh
```

The fold-split cache (`outputs/asr_v1/phase3/subsamples/trackB_no_ood/`)
was never committed to git and was removed in the repo cleanup —
`results/` JSONs are the **frozen outputs** of the original run.
To re-train, regenerate the cache with `scripts/cache_features_maceoff_delta.py`
+ the fold-split builder, then re-run the submitter.
