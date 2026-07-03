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

## Layout

- `code/runner_lowlr_trackB_m1delta.py` — shared runner
- `code/build_v2_bundle_m3.py` — m3-specific bundle assembler
- `code/build_xtb_extra_cache_v2.py` — rebuilds the v2 xTB extras cache
- `code/submit_xtb_extra_cache_v2.sh` — SLURM for the cache rebuild
- `code/submit_lowlr_m3_v2.sh` — SLURM for training
  (`BASELINE=xtb_geom6_plus_v2`, member 0 only)
- `results/foldF/member0.json` — per-fold outputs
- `results/xtb_extra_v2.parquet` — the v2 xTB extras cache used to
  build the bundle (kept here for reproducibility, since it is the
  differentiator vs. m2).
- `results/features_v6_delta_xtb_geom6_plus_v2.meta.json` — bundle
  manifest.

## Reproduce

```bash
# 1) rebuild the v2 xTB extras cache (once)
sbatch m3/code/submit_xtb_extra_cache_v2.sh
# 2) rebuild the bundle
python m3/code/build_v2_bundle_m3.py
# 3) train
sbatch m3/code/submit_lowlr_m3_v2.sh
```
