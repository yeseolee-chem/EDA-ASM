#!/usr/bin/env python3
"""GNN3R Phase W (docs/specs/GNN3R.md 9): the final GNN3 on every GNN3R pair (5 seeds) -> $GR/final.

    python analysis/gnn3r/train_final_r.py        env_d1fu2, RTX 3090 (gnn3/jobs/run.slurm), after the K array

K-1 .. K-3 are re-checked first (check_cv.run_checks, no copy). setting = gnn3.training.choose_final_setting(the 5
gnn3_fold<k>.json); gnn3.training.train_final_model(gr_common.dataset(), setting, $GR/final). An existing
final_report.json with the same setting and 5 networks is not retrained. W-1: n_train_pairs = every pair and the
networks were trained on an RTX 3090 (spec 12). Only then final_report.json and the setting are copied to
results/final/. Exit 0 pass, 3 = STOP.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402
import check_cv  # noqa: E402

C.import_gnn3()
from gnn3.training import choose_final_setting, train_final_model  # noqa: E402

OUT = C.RES / "final"


def main() -> int:
    k = check_cv.run_checks(copy=False)
    if not k["all_pass"]:
        C.write_json(OUT / "W_check.json", dict(all_pass=False, failed=["K recheck"], k_failed=k["failed"]))
        print(f"STOP: K re-check failed {k['failed']}")
        return 3
    records = [json.loads((C.CV / f"gnn3_fold{f}.json").read_text()) for f in range(check_cv.N_FOLDS)]
    setting = choose_final_setting(records)
    report_path = C.FINAL / "final_report.json"
    nets = [C.FINAL / f"gnn3_final__seed{j}.pt" for j in range(check_cv.N_SEEDS)]
    t0 = time.time()
    if report_path.is_file() and all(p.is_file() for p in nets) and \
            json.loads(report_path.read_text()).get("setting") == json.loads(json.dumps(setting)):
        report = json.loads(report_path.read_text())
        print(f"final model present ({report_path}); not retrained")
    else:
        report = train_final_model(C.dataset(), setting, C.FINAL)
    n_pairs = int(len(pd.read_csv(C.PAIRS_R, usecols=["pair_id"])))
    gpu = (report.get("environment") or {}).get("gpu", "")
    w1 = dict(n_train_pairs=int(report["n_train_pairs"]), n_pairs=n_pairs, gpu=gpu, rtx3090="RTX 3090" in gpu,
              ok=int(report["n_train_pairs"]) == n_pairs and "RTX 3090" in gpu)
    if w1["ok"]:
        C.copy_atomic(report_path, OUT / "final_report.json")
        C.write_json(OUT / "final_setting.json", setting)
    rep = dict(spec="docs/specs/GNN3R.md 9", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), wall_s=round(time.time() - t0, 1), setting=setting,
               networks={str(p): C.sha256(p) for p in nets if p.is_file()}, gates={"W-1": w1},
               all_pass=w1["ok"], failed=[] if w1["ok"] else ["W-1"])
    C.write_json(OUT / "W_check.json", rep)
    print(json.dumps(dict(all_pass=w1["ok"], setting=setting, W_1=w1), indent=1, default=str))
    return 0 if w1["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
