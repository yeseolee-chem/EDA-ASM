# SPEC_10 — per-family learning curve

Measures how the 5-channel EDA-ASM prediction improves as the **within-family**
training pool grows from 50 to 200 reactions. Every model here is
trained **exclusively on one family's reactions** — no cross-family
transfer. Two arms per cell:

- **xgb28_base** — pure 28-descriptor XGB (spec05 `no_sum_28d` recipe).
- **xgb28_delta** — spec06's 2-step `xgb28 + δ` learner, restricted
  to the family (same math as spec09 B1).

## Cohort

v9 bundle, 783 reactions total:

| family        | n   | 5-fold train / test (avg) |
|---------------|----:|---------------------------|
| dipolar       | 197 | 157–158 / 39–40           |
| qmrxn20_e2    | 199 | 159 / 40                  |
| qmrxn20_sn2   | 187 | 149–150 / 37–38           |
| rgd1          | 200 | 160 / 40                  |

CV splits are reused verbatim from
`spec/spec09_per_family_xgb28_delta/splits/family_folds/{family}_outer_folds.json`.

## Sizes

Target training pool per family fold:

| target | note                                     |
|-------:|------------------------------------------|
| 50     | strict                                   |
| 100    | strict                                   |
| 150    | strict for rgd1 / qmrxn20_e2 / dipolar; qmrxn20_sn2 = 149–150 (bumps against fold-train ceiling by 0–1 rxn) |

Only these three sizes are run — no full-fold cap point. If the
whole-family baseline is needed for reference, use the numbers
already produced by spec09.

## Sampling rule

Deterministic, seeded, nested per (family, fold):

1. seed = `42 + family_idx*10 + fold`
2. shuffle the family's train roster once with that seed
3. for target N, take the first `min(N, family_train_size)` rids

Because the same seed is reused for all sizes in one (family, fold),
the size-N subset is a **strict superset** of the size-(N−50) subset
— clean learning curve, no seed-swap variance between adjacent
points.

## Cells

One SLURM job = one `(family, size, fold, member)` triple. Both
arms are trained inside a single cell so they share the exact same
train / test rids and are directly comparable.

Default plan:

- 4 families × 3 sizes × 5 folds × 1 member = **60 cells**.

Members can be extended later without touching existing outputs
(runner is idempotent on `member{M}.json`).

## Files

```
spec10_family_learning_curve/
├── README.md
├── code/
│   ├── make_lc_family_splits.py     # builds splits/lc_family_splits.json
│   ├── train_lc_family_cell.py      # one (family, size, fold, member)
│   ├── submit_lc_family_cell.sh     # sbatch, 48h, gpu3/4/5
│   ├── chain_lc_family.sh           # auto-launcher, ≤ 10 jobs
│   └── aggregate_lc_family.py       # per-family curves + REPORT
├── splits/
│   └── lc_family_splits.json        # per (family, size, fold): train_rids
├── oof/
│   └── {family}/size{N}/fold{f}/member{m}.json
├── results/
│   ├── learning_curve.csv
│   ├── summary.csv
│   └── REPORT.md
└── figures/
    └── learning_curve_{family}.pdf  # one PDF per family
```

## CLAUDE.md compliance

- Every sbatch uses `#SBATCH --time=48:00:00`.
- Every cell is idempotent (skips if `member{M}.json` exists).
- Chain launcher throttles submissions so no more than **10** of my
  jobs sit in the queue at once (checks `squeue -u yeseo1ee`).
- SLURM `.out` / `.err` files go to
  `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/`.
- Partitions distributed across `gpu3,gpu4,gpu5`.
