#!/usr/bin/env python3
"""SPEC17rev2 Step 4 — convert R/TS/P npz into React-OT pkl format.

Output layout matches reactot/dataset/transition1x.py:
  {"reactant":         {"charges": [...], "positions": [...]},
   "transition_state": {"charges": [...], "positions": [...]},
   "product":          {"charges": [...], "positions": [...]},
   "single_fragment":  [0, ...],
   "rxn_id":           [...]}

GATE-4: conversion success ≥ 97%. rxn_id list is required for fold assignment
in Step 5 and cross-fitting audit in Step 7.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train")
RC = BASE / "data" / "reactant_complex"

Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
     "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53}


def main() -> int:
    build_csv = BASE / "artifacts" / "reactant_build.csv"
    comp_csv = BASE / "artifacts" / "composition.csv"
    if not build_csv.exists():
        print(f"[FATAL] {build_csv} missing. Run Step 2.", file=sys.stderr)
        return 1
    if not comp_csv.exists():
        print(f"[FATAL] {comp_csv} missing. Run Step 1.", file=sys.stderr)
        return 1

    build = pd.read_csv(build_csv)
    comp = pd.read_csv(comp_csv)
    keep = sorted(set(build[build.ok & build.p_order_ok].rxn_id) &
                  set(comp[(comp.status == "ok") & comp.supported].rxn_id))
    print(f"conversion candidates: {len(keep)}")

    data = {
        "reactant": {"charges": [], "positions": []},
        "transition_state": {"charges": [], "positions": []},
        "product": {"charges": [], "positions": []},
        "single_fragment": [],
        "rxn_id": [],
    }
    fails = []
    for rid in keep:
        npz_path = RC / f"rxn_{rid:04d}.npz"
        if not npz_path.exists():
            fails.append((rid, "npz missing"))
            continue
        npz = np.load(npz_path, allow_pickle=True)
        syms = [str(s) for s in npz["syms"]]
        try:
            chg = [Z[s] for s in syms]
        except KeyError as e:
            fails.append((rid, f"unknown element {e}"))
            continue
        data["reactant"]["charges"].append(chg)
        data["reactant"]["positions"].append(npz["R"].tolist())
        data["transition_state"]["charges"].append(chg)
        data["transition_state"]["positions"].append(npz["TS"].tolist())
        data["product"]["charges"].append(chg)
        data["product"]["positions"].append(npz["P"].tolist())
        data["single_fragment"].append(0)
        data["rxn_id"].append(rid)

    n_conv = len(data["rxn_id"])
    print(f"converted: {n_conv} / {len(keep)}   failures: {len(fails)}")

    out_pkl = BASE / "data" / "coley_all.pkl"
    tmp = out_pkl.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.replace(out_pkl)
    print(f"wrote {out_pkl}")

    pd.DataFrame(fails, columns=["rxn_id", "reason"]).to_csv(
        BASE / "artifacts" / "convert_failures.csv", index=False
    )

    rate = n_conv / max(1, len(keep))
    passed = rate >= 0.97
    (BASE / "artifacts" / "GATE4_STATUS.txt").write_text(
        ("PASS" if passed else "FAIL") + "\n"
        f"candidates={len(keep)}\n"
        f"converted={n_conv}\n"
        f"failed={len(fails)}\n"
        f"rate={rate:.4f}\n"
    )
    if not passed:
        print(f"[GATE-4 FAIL] conversion rate {rate:.2%} < 97%", file=sys.stderr)
        return 1
    print("=== GATE-4 PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
