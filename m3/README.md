# m3 — xtb_geom6_plus_v2 baseline (Δ-learning, xTB + d1–d6 + v2 extras)

Track B `m1_delta` with the **xtb_geom6_plus_v2** baseline — xTB
descriptors, d1–d6 physics features, **plus a v2 set of xTB extras**
rebuilt with an index-based extractor. Only member 0 is trained per
fold (5 cells total), matching the m1/m2 member-0 comparison set.

## Training config

- Model: `ModelM1Delta` (identical to m1/m2)
- Baseline: `xtb_geom6_plus_v2`
- Pool: `trackB_no_ood`
- LR = 1e-5, epochs ≤ 100k, early-stop patience 10k
- 5 folds × 1 member (member 0 only) = 5 cells

## Layout — self-contained

```
m3/
├── code/
│   ├── runner_lowlr_trackB_m1delta.py    # BASELINE=xtb_geom6_plus_v2, member 0
│   ├── build_v2_bundle_m3.py             # m3-specific bundle assembler
│   ├── build_xtb_extra_cache_v2.py       # rebuilds the v2 xTB extras cache
│   ├── submit_xtb_extra_cache_v2.sh      # SLURM for cache rebuild
│   ├── submit_lowlr_m3_v2.sh             # SLURM for training
│   ├── eda_asm/asr_v1/                   # local copy of the shared library
│   └── scripts/                          # cache_features_*, train_*, learning_curve_*
└── results/
    ├── foldF/member0.json
    ├── xtb_extra_v2.parquet              # v2 xTB extras cache (differentiator vs m2)
    └── features_v6_delta_xtb_geom6_plus_v2.meta.json   # bundle manifest
```

m3 differs from m2 only in the `descriptors` tensor (v2 xTB extras
rebuilt with an index-based extractor). Same model architecture and
training loop as m1/m2.

## Reproduce

```bash
# 1) rebuild the v2 xTB extras cache (once)
sbatch m3/code/submit_xtb_extra_cache_v2.sh
# 2) rebuild the bundle
python m3/code/build_v2_bundle_m3.py
# 3) train
sbatch m3/code/submit_lowlr_m3_v2.sh
```
