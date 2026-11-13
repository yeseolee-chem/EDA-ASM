#!/usr/bin/env python3
"""GNN3R Phase E (docs/specs/GNN3R.md 8): evaluation tables of the GNN3R cross-validation predictions.

    python analysis/gnn3r/evaluate_r.py [--predictions $GR/cv]      env_d1fu2 (gnn3/jobs/run.slurm), CPU; no torch

The tables of gnn3/scripts/evaluate.py main (metrics, rank, cause, identity, breakdown, applicability); the only
differences are the two inputs: truth = metrics.channel_truth(results/F_pairs_r.csv), sets =
metrics.pair_sets(results/S_folds_r.csv). Predictions are read with evaluate.read_predictions (the 5
gnn3_fold<k>.parquet). Writes results/eval/. E-1: pairs evaluated = pairs - fold -1 pairs. Exit 0 pass, 3 = STOP.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402

C.import_gnn3()
from gnn3 import metrics as M  # noqa: E402

OUT = C.RES / "eval"


def gnn3_evaluate():
    """gnn3/scripts/evaluate.py as a module (read_predictions, MODEL_NAME); its main is not run."""
    spec = importlib.util.spec_from_file_location("gnn3_evaluate", C.GNN3_PACKAGE / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=C.CV)
    args = parser.parse_args()
    E = gnn3_evaluate()
    predictions = E.read_predictions(args.predictions, False)
    truth, attributes = M.channel_truth(C.PAIRS_R)
    long = M.assemble({E.MODEL_NAME: predictions}, truth, attributes, M.pair_sets(C.S_FOLDS_R))
    pair_ids = sorted(long.pair_id.unique())
    C.write_atomic(OUT / "metrics.csv", M.error_table(long, pair_ids, [E.MODEL_NAME]).to_csv(index=False))
    C.write_atomic(OUT / "rank.csv", M.ranking_table(long, pair_ids, [E.MODEL_NAME]).to_csv(index=False))
    C.write_atomic(OUT / "cause.csv",
                   M.cause_table([M.cause_agreement(long, pair_ids, E.MODEL_NAME)]).to_csv(index=False))
    C.write_atomic(OUT / "identity.csv",
                   pd.DataFrame([M.identity_residual(long, pair_ids, E.MODEL_NAME)]).to_csv(index=False))
    C.write_atomic(OUT / "breakdown.csv", M.breakdown_table(long, pair_ids, E.MODEL_NAME).to_csv(index=False))
    C.write_atomic(OUT / "applicability.json", json.dumps(dict(
        rule="95th percentile (numpy linear) of the test-pair seed spread per target, kcal/mol",
        thresholds=M.applicability_thresholds(predictions)), indent=1) + "\n")
    S = pd.read_csv(C.S_FOLDS_R, usecols=["pair_id", "fold_T1"])
    want = int(len(S) - (S.fold_T1 == -1).sum())
    e1 = dict(n_evaluated=len(pair_ids), n_pairs=int(len(S)), n_fold_minus1=int((S.fold_T1 == -1).sum()), want=want,
              ok=len(pair_ids) == want)
    rep = dict(spec="docs/specs/GNN3R.md 8", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), predictions=str(args.predictions),
               inputs={str(p): C.sha256(p) for p in (C.PAIRS_R, C.S_FOLDS_R)},
               outputs={n: C.sha256(OUT / n) for n in ("metrics.csv", "rank.csv", "cause.csv", "identity.csv",
                                                       "breakdown.csv", "applicability.json")},
               gates={"E-1": e1}, all_pass=e1["ok"], failed=[] if e1["ok"] else ["E-1"])
    C.write_json(OUT / "E_check.json", rep)
    print(f"{len(pair_ids)} pairs evaluated -> {OUT}; E-1 {'pass' if e1['ok'] else 'FAIL'} ({e1})")
    return 0 if e1["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
