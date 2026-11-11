"""Predict ΔΔ (strain, inter, solv, barrier; kcal/mol) for pairs of reactions with saved GNN3 networks.

    python gnn3/scripts/predict.py --models '/path/final/gnn3_final__seed*.pt' --tokens TOKENS_DIR \
        --table X182.parquet --pairs pairs.csv --out predictions.csv

pairs.csv has columns pair_id, a_rxn, b_rxn: a = the substituted reaction, b = its H reference (same dipole, skeleton
and position, X = H). Both reactions need token files made by ``scripts/prepare_inputs.py`` with the SAME procedure
(the reference must also come from the g2 TS search; mixing TS sources breaks the cancellation that ΔΔ relies on).

Output columns: pair_id, a_rxn, b_rxn, target, yhat (seed mean), yhat_sd (seed spread), out_of_range (spread above the
95th percentile seen in cross-validation: treat the value as unreliable). Pairs whose reactions fail the input checks
are written to <out>.excluded.csv with the reason.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", required=True, help="glob of the model files to average (e.g. the 5 seeds)")
    parser.add_argument("--tokens", type=Path, required=True, help="directory of <rxn_id>.npz token files")
    parser.add_argument("--table", type=Path, required=True, help="parquet with rxn_id + the 182 global features")
    parser.add_argument("--pairs", type=Path, required=True, help="csv with pair_id, a_rxn, b_rxn")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-flag", action="store_true", help="do not compute the out_of_range flag")
    args = parser.parse_args()
    from gnn3.predict import APPLICABILITY_THRESHOLDS_KCAL, predict_pairs
    files = sorted(Path(p) for p in glob.glob(args.models))
    if not files:
        raise SystemExit(f"no model files match {args.models}")
    predictions, excluded = predict_pairs(files, args.tokens, args.table, pd.read_csv(args.pairs),
                                          None if args.no_flag else APPLICABILITY_THRESHOLDS_KCAL)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out, index=False)
    excluded.to_csv(args.out.with_suffix(".excluded.csv"), index=False)
    print(f"{predictions.pair_id.nunique()} pairs predicted with {len(files)} networks, {len(excluded)} excluded "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
