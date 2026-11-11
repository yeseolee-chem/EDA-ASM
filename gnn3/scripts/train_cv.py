"""Cross-validation of GNN3: one outer fold per call (tuning on the inner split, 5 final seeds, test predictions).

    python gnn3/scripts/train_cv.py --fold 0 --out /gpfs/tmp_cpu2/$USER/gnn3/cv          (GPU, ~1 GPU-hour per fold)

Writes into --out: gnn3_fold<k>.json (chosen cell / epochs, validation curves, checks), gnn3_fold<k>.parquet (test
predictions in kcal/mol) and gnn3_fold<k>__seed<j>.pt (the 5 final networks). A finished fold is skipped on resubmit.
Run on an RTX 3090 to reproduce the published numbers bit for bit (see README "Reproducibility"):

    sbatch -p gpu1 --gres=gpu:rtx3090:1 --time=48:00:00 -J gnn3_cv0 gnn3/jobs/run.slurm \
        gnn3/scripts/train_cv.py --fold 0 --out /gpfs/tmp_cpu2/$USER/gnn3/cv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnn3 import constants as K  # noqa: E402
from gnn3.data import PairDataset  # noqa: E402
from gnn3.training import run_cross_validation_unit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, required=True, choices=range(K.N_FOLDS))
    parser.add_argument("--out", type=Path, default=K.SCRATCH / "cv")
    parser.add_argument("--log-every", type=int, default=50, help="print the training loss every N epochs")
    args = parser.parse_args()
    record = run_cross_validation_unit(PairDataset(), args.fold, args.out, log_every=args.log_every)
    print(f"fold {args.fold}: cell {record['chosen']['cell']}, {record['chosen']['best_epoch']} epochs, "
          f"validation MAE (kcal/mol) {record['val_kcal_mae']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
