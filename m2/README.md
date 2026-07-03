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

## Layout

- `code/runner_lowlr_trackB_m1delta.py` — shared runner
- `code/build_v2_bundles.py` — builds `features_v6_delta_xtb_geom6.pt`
- `code/build_xtb_cache.py` — extracts xTB descriptors into
  `xtb_features.parquet` (input to the bundle builder).
- `code/submit_lowlr_xtbg6.sh` — SLURM submitter (`BASELINE=xtb_geom6`).
- `results/foldF/memberM.json` — per-cell outputs.

## Reproduce

```bash
sbatch m2/code/submit_lowlr_xtbg6.sh
```

Prior to the SLURM run, `xtb_features.parquet` must exist under the
bundle-builder path (produced by `build_xtb_cache.py`).
