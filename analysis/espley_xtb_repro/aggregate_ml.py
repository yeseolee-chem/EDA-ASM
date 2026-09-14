#!/usr/bin/env python3
"""aggregate_ml.py — merge per-target JSON/parquet into unified report + table + predictions."""
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
TARGETS_DIR = ROOT / "ml_targets"
OUT_REPORT = ROOT / "ml_report.json"
OUT_TABLE = ROOT / "ml_table_espley.csv"
OUT_PREDS = ROOT / "predictions.parquet"


def main():
    json_paths = sorted(TARGETS_DIR.glob("target_*.json"))
    if not json_paths:
        sys.exit(f"no per-target JSON files in {TARGETS_DIR}")

    per_target = [json.loads(p.read_text()) for p in json_paths]

    report = dict(n_targets=len(per_target), per_target={})
    rows = []
    for t in per_target:
        tgt = t["target"]
        report["per_target"][tgt] = t
        pre = (t.get("pre_ml") or {}).get("mae", float("nan"))
        for fs_name, A in t.get("protocol_A", {}).items():
            for mname, r in A.items():
                rows.append(dict(protocol="A", feature_set=fs_name, target=tgt, model=mname,
                                 pre_ml_mae=pre, train_mae=r["train_mae"],
                                 test_mae=r["test_mae"], test_mae_sd=r["test_mae_sd"],
                                 test_r2=r["test_r2"], test_range=r["test_range"],
                                 test_mae_pct_range=r["test_mae_pct_range"]))
        for fs_name, B in t.get("protocol_B", {}).items():
            for mname, r in B.items():
                rows.append(dict(protocol="B", feature_set=fs_name, target=tgt, model=mname,
                                 pre_ml_mae=pre, train_mae=float("nan"),
                                 test_mae=r["pooled_oof"]["mae"], test_mae_sd=float("nan"),
                                 test_r2=r["pooled_oof"]["r2"], test_range=float("nan"),
                                 test_mae_pct_range=float("nan")))

    OUT_REPORT.write_text(json.dumps(report, indent=2))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABLE, index=False)

    preds_paths = sorted(TARGETS_DIR.glob("preds_*.parquet"))
    if preds_paths:
        preds = pd.concat([pd.read_parquet(p) for p in preds_paths], ignore_index=True)
        preds.to_parquet(OUT_PREDS, index=False)
        print(f"merged {len(preds_paths)} preds parquets ({len(preds)} rows) -> {OUT_PREDS}")

    print(f"targets merged  : {len(per_target)}")
    print(f"report          : {OUT_REPORT}")
    print(f"table (rows={len(df)}): {OUT_TABLE}")


if __name__ == "__main__":
    main()
