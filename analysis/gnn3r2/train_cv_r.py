#!/usr/bin/env python3
"""GNN3R Phase K (docs/specs/GNN3R.md 7): one outer fold of the GNN3 cross-validation on the GNN3R data.

    python analysis/gnn3r/train_cv_r.py --fold-from-array --out $GR/cv        (RTX 3090; fold = $SLURM_ARRAY_TASK_ID)
    python analysis/gnn3r/train_cv_r.py --fold 2 --out $GR/cv

= gnn3/scripts/train_cv.py with gr_common.dataset() instead of PairDataset(): gnn3.training.run_cross_validation_unit
tunes the 4-cell grid on the inner split, trains the 5 seeds with the chosen cell and predicts the test pairs. A
finished fold returns its json unchanged (resubmit-safe). Submit from the repository root:

    sbatch --array=0-4%5 -p gpu1 --gres=gpu:rtx3090:1 --time=48:00:00 -J gnn3r_cv \
      --output=$GR/logs/%x_%A_%a.out --export=ALL,GNN3_SCRATCH=$GR \
      gnn3/jobs/run.slurm analysis/gnn3r/train_cv_r.py --fold-from-array --out $GR/cv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402

C.import_gnn3()
from gnn3 import constants as K  # noqa: E402
from gnn3.training import run_cross_validation_unit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--fold", type=int, choices=range(K.N_FOLDS))
    which.add_argument("--fold-from-array", action="store_true", help="fold = $SLURM_ARRAY_TASK_ID")
    parser.add_argument("--out", type=Path, default=C.CV)
    parser.add_argument("--log-every", type=int, default=50, help="print the training loss every N epochs")
    args = parser.parse_args()
    fold = int(os.environ["SLURM_ARRAY_TASK_ID"]) if args.fold_from_array else args.fold
    if fold not in range(K.N_FOLDS):
        raise SystemExit(f"fold {fold} not in 0..{K.N_FOLDS - 1}")
    record = run_cross_validation_unit(C.dataset(), fold, args.out, log_every=args.log_every)
    print(f"fold {fold}: cell {record['chosen']['cell']}, {record['chosen']['best_epoch']} epochs, "
          f"validation MAE (kcal/mol) {record['val_kcal_mae']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
