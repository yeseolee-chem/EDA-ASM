# m3 — spec-v1 rebuild (2026-07-04)

Fresh 5-fold × 5-member cross-validation on the 691-reaction spec-v1
cohort (98 xTB SCF non-convergences dropped from the original 789).

## Spec compliance

- Loss: per-channel σ_c-normalised L1 (train-fold label std).
- InputStandardizer: fit on train R+P features only (TS excluded).
- Optimiser: Adam, lr = 1e-5, weight_decay = 1e-3, grad-clip 5.0.
- Budget: EPOCHS_MAX = 100000, PATIENCE = 10000, batch = 16.
- Backbone: frozen MACE-OFF23_medium (256-d per-atom features).
- Model: ModelM1Delta (cross-attn + Δ-learning), ridge baseline (α=1).

## Layout

- `code/` — runner + SLURM submitter + local eda_asm.asr_v1 package + support scripts.
- `results/foldF/memberM.json` — 25 frozen test-set predictions +
  channel MAE + barrier MAE + best/final epoch metadata.

## Descriptor set

- d1..d24: geom6 + xTB + Parr ω + Σq² + Σ|WBO| across the interfragment set.

## Aggregate metrics

See `comparison/REPORT.md` (spec-v1 3-way).
