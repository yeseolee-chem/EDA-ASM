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

## Layout

- `code/runner_lowlr_trackB_m1delta.py` — the training entrypoint (shared
  across m1/m2/m3; controlled by `BASELINE` env var).
- `code/build_v2_bundles.py` — builds `features_v6_delta_geom6.pt`.
- `code/submit_lowlr_geom6.sh` — SLURM submitter that sets
  `BASELINE=geom6`.
- `results/foldF/memberM.json` — per-cell test-set predictions +
  metrics.

## Reproduce

```bash
# from repo root
sbatch m1/code/submit_lowlr_geom6.sh
```

Note: The runner references shared modules under `src/eda_asm/asr_v1/`
and the fold-split cache under `outputs/asr_v1/phase3/subsamples/`;
these were part of the pre-cleanup pipeline. The `results/` JSONs
here are the frozen outputs of that run.
