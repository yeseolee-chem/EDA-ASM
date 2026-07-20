# SPEC_10 — whole-dataset learning curve (family-stratified sub-sampling)

Sibling to `spec10_family_learning_curve`. Here we sweep the training
**pool over all four families together**, subsampled 25-per-family per
step. Two arms per (fold, size, member) cell:

- **xgb28_base** — pure 28-descriptor XGB (spec05 `no_sum_28d` recipe).
- **xgb28_delta** — the 2-step `xgb28 + δ` learner (spec06 recipe).

## Cohort

v9 bundle: 783 reactions total (user's "786" — off by 3 vs the actual
`LOCKED_783` file).

| family        | n   |
|---------------|----:|
| rgd1          | 200 |
| qmrxn20_e2    | 199 |
| dipolar       | 197 |
| qmrxn20_sn2   | 187 |

## CV scheme

Reuses `spec/spec06_2step_xgb28_delta/splits/outer_folds.json` (whole-cohort
5-fold, family-stratified). Per fold:

- train pool ≈ 626 reactions
- test    ≈ 157 reactions
- per-family train counts: rgd1 160 / e2 159 / dipolar 157-158 / sn2 149-150

## Learning curve sizes

User request: 100, 200, 300, 400, 500, 600, 700, 786. Honoured, but note
the tail:

| target | per-family (nominal) | actual per fold | actual total |
|--------|----------------------|-----------------|--------------|
| 100    | 25                   | 25/25/25/25     | 100          |
| 200    | 50                   | 50/50/50/50     | 200          |
| 300    | 75                   | 75/75/75/75     | 300          |
| 400    | 100                  | 100/100/100/100 | 400          |
| 500    | 125                  | 125/125/125/125 | 500          |
| 600    | 150                  | 150/150/149-150/150 | 599–600  |
| 700    | 175                  | capped by sn2 → full fold train | ≈ 626 |
| 786    | 197                  | capped → full fold train        | ≈ 626 |

Once the smallest family (qmrxn20_sn2) is exhausted (~149 in fold-train),
targets 700 and 786 both collapse to the full training fold (~626 rxns).
Their metrics will therefore be numerically ≈ identical to each other
and to the natural "FULL fold" setting. They are still submitted so the
learning-curve plot ends at the exact user-requested x-axis positions.

## Sampling rule ("25/family, random within 25")

1. Per fold, seed = 42 + fold. For each family, deterministic-shuffle
   the fold's family-train roster once.
2. For target N, per family take `min(N/4, family_cap)` from the front
   of that shuffled roster (nested — size-N ⊇ size-(N−100)).
3. If any family runs out, deficit is topped up round-robin from
   families with remaining capacity, so the actual N stays close to
   target until the whole fold train is used.

Emitter: `code/make_lc_splits.py` → `splits/lc_splits.json`.

## Cell = one SLURM job

One `(size, fold, member)` triple per sbatch job. Both arms are trained
in the same cell so they share exactly the same train / test rids and
are directly comparable at every size.

Defaults: `members = [0]` → 5 folds × 8 sizes × 1 member = **40 cells**.

## Files

```
spec08_whole_dataset_learning_curve/
├── README.md
├── code/
│   ├── make_lc_splits.py          # builds splits/lc_splits.json
│   ├── train_lc_cell.py           # one (size, fold, member)
│   ├── submit_lc_cell.sh          # sbatch, 48h, gpu3/4/5
│   ├── chain_lc.sh                # auto-launcher (runs on cpu2 via sbatch)
│   ├── submit_chain_lc.sh         # sbatch wrapper that runs chain_lc.sh
│   ├── aggregate_lc.py            # metrics + PNG plots
│   └── submit_aggregate.sh        # sbatch wrapper for aggregate
├── splits/
│   └── lc_splits.json             # per (size, fold): train_rids
├── oof/
│   └── size{N}/fold{f}/member{m}.json
├── results/
│   ├── learning_curve.csv
│   ├── summary.csv
│   └── REPORT.md
└── figures/
    └── learning_curve.png
```

## CLAUDE.md compliance

- Every sbatch uses `#SBATCH --time=48:00:00`.
- Every cell is idempotent (skips if `member{M}.json` exists).
- **Launcher itself runs on cpu2 via sbatch** — no login-node processes.
- Concurrency governed by SLURM cap plus in-launcher `MAX_INFLIGHT=10`
  (running-only) and `MAX_SUBMIT=19` (safety margin under MaxSubmit=20).
- SLURM `.out` / `.err` under `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/`.
- Partitions: `gpu3,gpu4,gpu5` for cells; `cpu2` for the launcher.
