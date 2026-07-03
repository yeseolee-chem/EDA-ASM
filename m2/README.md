# m2 — xtb_geom6 baseline (Δ-learning, xTB + d1–d6)

Track B `m1_delta` model with the **xtb_geom6** baseline
(xTB descriptors *and* d1–d6 physics features). Trained on the same
789-reaction cohort and no-OOD split as m1.

## Training config

- Model: `ModelM1Delta` (identical to m1)
- Baseline: `xtb_geom6` (LinearBaseline over xTB features + d1..d6)
- Pool: `trackB_no_ood`
- LR = 1e-5, epochs ≤ 100k, early-stop patience 10k
- 5 folds × 5 members = 25 cells

## Layout — self-contained

```
m2/
├── code/
│   ├── runner_lowlr_trackB_m1delta.py    # training entrypoint (BASELINE=xtb_geom6)
│   ├── build_v2_bundles.py               # builds features_v6_delta_xtb_geom6.pt
│   ├── build_xtb_cache.py                # extracts xTB descriptors → xtb_features.parquet
│   ├── submit_lowlr_xtbg6.sh             # SLURM submitter
│   ├── eda_asm/asr_v1/                   # local copy of the shared library
│   │                                     #   (ModelM1Delta, MACE-OFF backbone,
│   │                                     #    LinearBaseline, delta training loop)
│   └── scripts/                          # cache_features_*, train_*, learning_curve_*
└── results/
    └── foldF/memberM.json
```

m2 differs from m1 only in the `descriptors` tensor swapped into the
feature bundle (xTB + d1..d6 instead of d1..d6 only). Same model
architecture and training loop.

## Reproduce

```bash
# from repo root
sbatch m2/code/submit_lowlr_xtbg6.sh
```

Prior to the SLURM run, `xtb_features.parquet` must exist under the
bundle-builder path (produced by `build_xtb_cache.py`). The fold-split
cache used at training time was never committed and would need to be
regenerated — `results/` JSONs are the frozen outputs of the original
run.
