"""Evaluation tables of GNN3 cross-validation predictions (CPU; no torch needed).

    python gnn3/scripts/evaluate.py --predictions /gpfs/tmp_cpu2/$USER/gnn3/cv --out gnn3/results/eval
    python gnn3/scripts/evaluate.py --published --out gnn3/results/eval       # the published GNN3 predictions

--predictions is a directory of gnn3_fold<k>.parquet files (``scripts/train_cv.py``) or one parquet with the columns
pair_id, fold, target, yhat, yhat_seed_sd. Writes:

    metrics.csv       MAE / RMSE / r / sign accuracy per set (P_main, P_sens_ref, test_all) x target x position;
                      P_main para rows carry the core-bootstrap 95 % interval of the MAE
    rank.csv          within-family barrier ranking (para and all positions)
    cause.csv         dominant-channel / cause-type agreement (pairs and cell means)
    identity.csv      |barrier - (strain + inter + solv)| of the predictions
    breakdown.csv     MAE and R² per position and per skeleton (model card)
    applicability.json   95th percentile of the seed spread per target (the out_of_range thresholds)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnn3 import constants as K  # noqa: E402
from gnn3 import metrics as M  # noqa: E402

MODEL_NAME = "GNN3"


def read_predictions(source: Path | None, published: bool) -> pd.DataFrame:
    if published:
        table = pd.read_parquet(K.ORIGINAL_RESULTS / "KM_predictions.parquet")
        table = table[(table.rung == "lf3") & (table.model == "GNN")]
    elif source.is_dir():
        files = sorted(source.glob("gnn3_fold*.parquet"))
        if len(files) != K.N_FOLDS:
            raise SystemExit(f"{source}: expected {K.N_FOLDS} gnn3_fold<k>.parquet files, found {len(files)}")
        table = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    else:
        table = pd.read_parquet(source)
    table = table[["pair_id", "fold", "target", "yhat", "yhat_seed_sd"]].copy()
    table["pair_id"] = table.pair_id.astype(str)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--published", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.published and args.predictions is None:
        parser.error("give --predictions or --published")
    predictions = read_predictions(args.predictions, args.published)
    truth, attributes = M.channel_truth()
    long = M.assemble({MODEL_NAME: predictions}, truth, attributes, M.pair_sets())
    pair_ids = sorted(long.pair_id.unique())
    args.out.mkdir(parents=True, exist_ok=True)
    M.error_table(long, pair_ids, [MODEL_NAME]).to_csv(args.out / "metrics.csv", index=False)
    M.ranking_table(long, pair_ids, [MODEL_NAME]).to_csv(args.out / "rank.csv", index=False)
    M.cause_table([M.cause_agreement(long, pair_ids, MODEL_NAME)]).to_csv(args.out / "cause.csv", index=False)
    pd.DataFrame([M.identity_residual(long, pair_ids, MODEL_NAME)]).to_csv(args.out / "identity.csv", index=False)
    M.breakdown_table(long, pair_ids, MODEL_NAME).to_csv(args.out / "breakdown.csv", index=False)
    (args.out / "applicability.json").write_text(json.dumps(dict(
        rule="95th percentile (numpy linear) of the test-pair seed spread per target, kcal/mol",
        thresholds=M.applicability_thresholds(predictions)), indent=1) + "\n")
    print(f"{len(pair_ids)} pairs evaluated -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
