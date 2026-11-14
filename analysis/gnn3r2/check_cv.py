#!/usr/bin/env python3
"""GNN3R Phase K checks (docs/specs/GNN3R.md 7): the 5 cross-validation units in $GR/cv -> results/cv/K_check.json.

    python analysis/gnn3r/check_cv.py            env_d1fu2 (gnn3/jobs/run.slurm), CPU; no torch needed

K-1 gnn3_fold<k>.json + .parquet for k = 0..4 and the 25 gnn3_fold<k>__seed<j>.pt;
K-2 every check stored in the json passes (antisymmetry gate_M1 and leak check of the 4 tuning fits and the 5 seeds),
    the parquet sha256 is the recorded one, the units were made from the GNN3R inputs (labels, splits, X182 sha256)
    and environment.gpu is an RTX 3090 (spec 12: K / W only on RTX 3090);
K-3 fold k parquet: pair ids = the test list of results/splits/T1__f<k>.json, the 4 targets once per pair, finite.
When all pass the json and parquet files are copied to results/cv/. Exit 0 pass, 4 = folds missing (K-1: resubmit
only those), 3 = STOP.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402

N_FOLDS, N_SEEDS = 5, 5
TARGETS = ("strain", "inter", "solv", "barrier")         # gnn3.constants.TARGETS
OUT = C.RES / "cv"


def unit_ok(rec: dict) -> dict:
    fits = [("tune", t) for t in rec.get("tune", [])] + [("final", f) for f in rec.get("final", [])]
    bad = [dict(stage=s, cell=f.get("cell"), seed=f.get("seed"), gate_M1=f.get("gate_M1"), leak=f.get("leak"))
           for s, f in fits if not ((f.get("gate_M1") or {}).get("ok") and (f.get("leak") or {}).get("ok"))]
    return dict(n_tune=len(rec.get("tune", [])), n_final=len(rec.get("final", [])), failed=bad,
                ok=len(rec.get("tune", [])) == 4 and len(rec.get("final", [])) == N_SEEDS and not bad)


def run_checks(cv: Path = C.CV, copy: bool = True) -> dict:
    files = {k: dict(json=(cv / f"gnn3_fold{k}.json").is_file(), parquet=(cv / f"gnn3_fold{k}.parquet").is_file(),
                     pt=sum((cv / f"gnn3_fold{k}__seed{j}.pt").is_file() for j in range(N_SEEDS)))
             for k in range(N_FOLDS)}
    missing = [k for k, f in files.items() if not (f["json"] and f["parquet"] and f["pt"] == N_SEEDS)]
    gates = {"K-1": dict(files=files, missing_folds=missing, ok=not missing)}
    want = dict(labels=C.sha256(C.PAIRS_R), s_splits=C.sha256(C.S_SPLITS_R), x182=C.sha256(C.X182))
    k2, k3 = {}, {}
    for k in range(N_FOLDS):
        if k in missing:
            continue
        rec = json.loads((cv / f"gnn3_fold{k}.json").read_text())
        pq = cv / f"gnn3_fold{k}.parquet"
        inputs = (rec.get("settings") or {}).get("inputs_sha256") or {}
        same_inputs = {n: inputs.get(n) == s for n, s in want.items()}
        u = unit_ok(rec)
        gpu = (rec.get("environment") or {}).get("gpu", "")
        k2[k] = dict(fits=u, parquet_sha256=rec.get("parquet_sha256") == C.sha256(pq), inputs=same_inputs,
                     fold=rec.get("settings", {}).get("fold") == k, chosen=rec.get("chosen"), gpu=gpu,
                     rtx3090="RTX 3090" in gpu,
                     ok=u["ok"] and rec.get("parquet_sha256") == C.sha256(pq) and all(same_inputs.values())
                     and rec.get("settings", {}).get("fold") == k and "RTX 3090" in gpu)
        test = json.loads((C.SPLIT_DIR / f"T1__f{k}.json").read_text())["test"]
        t = pd.read_parquet(pq)
        counts = t.groupby("pair_id").target.agg(lambda s: tuple(sorted(s)))
        vals = t[["yhat", "yhat_seed_sd"]].to_numpy(float)
        k3[k] = dict(n_rows=int(len(t)), n_pairs=int(t.pair_id.nunique()), n_test=len(test),
                     pair_ids_equal_test=set(t.pair_id.astype(str)) == set(test),
                     targets=sorted(t.target.unique().tolist()),
                     each_pair_4_targets=bool((counts == tuple(sorted(TARGETS))).all()),
                     finite=bool(np.isfinite(vals).all()))
        k3[k]["ok"] = (k3[k]["pair_ids_equal_test"] and k3[k]["targets"] == sorted(TARGETS)
                       and k3[k]["each_pair_4_targets"] and k3[k]["finite"] and len(t) == 4 * len(test))
    gates["K-2"] = dict(per_fold=k2, ok=bool(k2) and all(v["ok"] for v in k2.values()))
    gates["K-3"] = dict(per_fold=k3, ok=bool(k3) and all(v["ok"] for v in k3.values()))
    bad = [g for g, v in gates.items() if not v["ok"]]
    rep = dict(spec="docs/specs/GNN3R.md 7", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), cv=str(cv), want_inputs_sha256=want, gates=gates,
               all_pass=not bad, failed=bad)
    if not bad and copy:
        for k in range(N_FOLDS):
            for ext in ("json", "parquet"):
                C.copy_atomic(cv / f"gnn3_fold{k}.{ext}", OUT / f"gnn3_fold{k}.{ext}")
        rep["copied"] = {str(OUT / f"gnn3_fold{k}.{e}"): C.sha256(OUT / f"gnn3_fold{k}.{e}")
                         for k in range(N_FOLDS) for e in ("json", "parquet")}
    return rep


def main() -> int:
    rep = run_checks()
    C.write_json(OUT / "K_check.json", rep)
    print(json.dumps(dict(all_pass=rep["all_pass"], failed=rep["failed"],
                          missing_folds=rep["gates"]["K-1"]["missing_folds"],
                          chosen={k: v["chosen"] for k, v in rep["gates"]["K-2"]["per_fold"].items()}), indent=1))
    if rep["all_pass"]:
        return 0
    return 4 if rep["failed"] == ["K-1"] or (rep["gates"]["K-1"]["missing_folds"]
                                             and all(v["ok"] for v in rep["gates"]["K-2"]["per_fold"].values())
                                             and all(v["ok"] for v in rep["gates"]["K-3"]["per_fold"].values())) else 3


if __name__ == "__main__":
    sys.exit(main())
