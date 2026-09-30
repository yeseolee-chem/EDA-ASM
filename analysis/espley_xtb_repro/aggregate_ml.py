#!/usr/bin/env python3
"""aggregate_ml.py — merge per-target JSON/parquet into unified report + table + predictions.

  [ESPLEY_ML_OUT=<dir>] [ESPLEY_GEOM=g0|g1] python aggregate_ml.py
<ML_OUT>/ml_targets/ -> <ML_OUT>/ml_report.json, ml_table_espley.csv, predictions.parquet (ML_OUT default $ESPLEY_OUT).
Any target set (rev 3: 12 targets, rev 4: 9); Protocol B rows only where it was run.
Gates (exit != 0, nothing written): a preds parquet for every JSON, no duplicate target, the whole recorded target
set present (rev 4 JSONs carry targets_in_set), one geom (= ESPLEY_GEOM if set), protocols, row set and feature file
(path + sha256).
Outputs are rewritten atomically on every run (cheap and deterministic).
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
ML_OUT = Path(os.environ.get("ESPLEY_ML_OUT") or ROOT)
GEOM = os.environ.get("ESPLEY_GEOM") or None
TARGETS_DIR = ML_OUT / "ml_targets"
OUT_REPORT = ML_OUT / "ml_report.json"
OUT_TABLE = ML_OUT / "ml_table_espley.csv"
OUT_PREDS = ML_OUT / "predictions.parquet"
SHARED = ("geom", "target_set", "protocols", "features_file", "features_sha256", "rows_sha256",
          "n_rows_ml")                                                         # identical in every per-target JSON


def gates(per_target, preds_paths):
    bad = []
    tg = [t["target"] for t in per_target]
    dup = sorted({x for x in tg if tg.count(x) > 1})
    if dup:
        bad.append(f"duplicate targets {dup}")
    for key in SHARED:
        vals = sorted({str(t.get(key)) for t in per_target})
        if len(vals) > 1:
            bad.append(f"per-target JSONs disagree on {key}: {vals}")
    if GEOM and per_target[0].get("geom") != GEOM:
        bad.append(f"ESPLEY_GEOM={GEOM} but the JSONs say geom={per_target[0].get('geom')}")
    want = per_target[0].get("targets_in_set")
    if want and sorted(tg) != sorted(want):
        bad.append(f"target set {per_target[0].get('target_set')}: missing {sorted(set(want) - set(tg))},"
                   f" unexpected {sorted(set(tg) - set(want))}")
    miss = [p.name for p in preds_paths if not p.exists()]
    if miss:
        bad.append(f"preds parquet missing: {miss}")
    return bad


def main():
    json_paths = sorted(TARGETS_DIR.glob("target_*.json"))
    if not json_paths:
        sys.exit(f"no per-target JSON files in {TARGETS_DIR}")

    per_target = [json.loads(p.read_text()) for p in json_paths]
    preds_paths = [TARGETS_DIR / f"preds_{p.stem[len('target_'):]}.parquet" for p in json_paths]
    bad = gates(per_target, preds_paths)
    if bad:
        sys.exit(f"GATE ({TARGETS_DIR}): " + "; ".join(bad))

    t0 = per_target[0]
    geom = t0.get("geom")
    report = dict(n_targets=len(per_target), **{k: t0.get(k) for k in SHARED}, rows_file=t0.get("rows_file"),
                  per_target={})
    rows = []
    for t in per_target:
        tgt = t["target"]
        report["per_target"][tgt] = t
        pre_info = t.get("pre_ml") or {}
        pre_mae = pre_info.get("mae", float("nan"))
        pre_r = pre_info.get("pearson_r", float("nan"))
        for fs_name, A in t.get("protocol_A", {}).items():
            for mname, r in A.items():
                rows.append(dict(geom=geom, protocol="A", feature_set=fs_name, target=tgt, model=mname,
                                 pre_ml_mae=pre_mae, pre_ml_r=pre_r,
                                 train_mae=r["train_mae"], train_nmae=r.get("train_nmae", float("nan")),
                                 test_mae=r["test_mae"], test_mae_sd=r["test_mae_sd"],
                                 test_nmae=r.get("test_nmae", float("nan")),
                                 test_r2=r["test_r2"], test_range=r["test_range"],
                                 test_mae_pct_range=r["test_mae_pct_range"]))
        for fs_name, B in t.get("protocol_B", {}).items():
            for mname, r in B.items():
                rows.append(dict(geom=geom, protocol="B", feature_set=fs_name, target=tgt, model=mname,
                                 pre_ml_mae=pre_mae, pre_ml_r=pre_r,
                                 train_mae=float("nan"), train_nmae=float("nan"),
                                 test_mae=r["pooled_oof"]["mae"], test_mae_sd=float("nan"),
                                 test_nmae=r["pooled_oof"].get("nmae", float("nan")),
                                 test_r2=r["pooled_oof"]["r2"], test_range=float("nan"),
                                 test_mae_pct_range=float("nan")))

    df = pd.DataFrame(rows)
    preds = pd.concat([pd.read_parquet(p) for p in preds_paths], ignore_index=True)
    for out, write in ((OUT_REPORT, lambda f: f.write_text(json.dumps(report, indent=2))),
                       (OUT_TABLE, lambda f: df.to_csv(f, index=False)),
                       (OUT_PREDS, lambda f: preds.to_parquet(f, index=False))):
        tmp = out.with_name(out.name + ".tmp")
        write(tmp)
        os.replace(tmp, out)
    print(f"merged {len(preds_paths)} preds parquets ({len(preds)} rows) -> {OUT_PREDS}")

    print(f"geom            : {geom}  (target set {t0.get('target_set')}, n_rows_ml {t0.get('n_rows_ml')})")
    print(f"targets merged  : {len(per_target)}")
    print(f"report          : {OUT_REPORT}")
    print(f"table (rows={len(df)}): {OUT_TABLE}")


if __name__ == "__main__":
    main()
