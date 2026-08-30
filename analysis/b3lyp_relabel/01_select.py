#!/usr/bin/env python3
"""SPEC15 Step 1 — sample 200 rxns.

Strategy: reuse all 218 from spec14 (EDA already done, only fragment SPEs
needed) and pad with MMP-involved reactions when useful.

GATE-1: sample includes all 218 from spec14.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_relabel")
GEOM = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")

SEED = 42
N_TARGET = 200


def main():
    done = sorted(int(d.name.split("_")[1])
                  for d in (GEOM / "results_218").iterdir() if d.is_dir())
    print(f"EDA-completed rxns (spec14): {len(done)}")

    pair_path = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/delta_mae_baseline/artifacts/pairs_dedup.pkl")
    mmp_rxns = set()
    if pair_path.exists():
        P = pd.read_pickle(pair_path)
        mmp_rxns = set(int(x) for x in P["rxn_id_A"]) | set(int(x) for x in P["rxn_id_B"])
        print(f"MMP-involved rxns: {len(mmp_rxns)}")
    overlap = sorted(set(done) & mmp_rxns)
    print(f"218 ∩ MMP: {len(overlap)}")

    sel = list(done)  # start with all 218 (EDA reuse)
    extra = sorted(mmp_rxns - set(done))
    rng = np.random.default_rng(SEED)
    if len(sel) < N_TARGET and extra:
        need = min(N_TARGET - len(sel), len(extra))
        picked = list(rng.choice(extra, size=need, replace=False))
        sel += [int(x) for x in picked]

    sel = sorted(set(sel))
    # keep all 218 + any MMP extras; do NOT truncate below 218
    df = pd.DataFrame({"rxn_id": sel})
    df["eda_exists"] = df["rxn_id"].isin(done)
    df.to_csv(BASE / "artifacts" / "sample.csv", index=False)

    n = len(df)
    n_reuse = int(df["eda_exists"].sum())
    n_new = n - n_reuse

    status = "PASS" if n_reuse == 218 else "FAIL"
    with open(BASE / "artifacts" / "GATE1_STATUS.txt", "w") as f:
        f.write(f"{status} n={n} eda_reuse={n_reuse} eda_new={n_new}\n")
    print(f"\nSelected {n} rxns (EDA reuse {n_reuse}, EDA new {n_new})")
    print(f"=== GATE-1 {status} ===")


if __name__ == "__main__":
    main()
