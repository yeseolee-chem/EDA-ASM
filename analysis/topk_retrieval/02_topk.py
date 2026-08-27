#!/usr/bin/env python3
"""SPEC13 Step 2 — Top-1 / Top-3 / Spearman within each candidate group.

Uses OOF predictions from delta_mae_baseline (groupkfold_component scheme,
5 seeds pooled — see §2.2 of SPEC13).

For each channel × seed × candidate group:
  Top-1: argmin(y_true) == argmin(y_pred)                    (720 samples)
  Top-3: argmin(y_true) in argsort(y_pred)[:3]               (720 samples)
  Spearman ρ(y_true, y_pred) — computed only if len >= 4     (595 samples)

GATE-2: match SPEC §7 targets to 4 decimals.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/topk_retrieval")
SRC = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/delta_mae_baseline/artifacts")

SCHEME = "groupkfold_component"
SEEDS = [42, 43, 44, 45, 46]
MIN_SPEAR = 4
TOPK = 3
CHANNELS = ["d1_own_dft", "d2_own_dft", "elst_dft", "pauli_dft",
            "oi_dft", "disp_dft", "cpcm_dft"]

# Load
oof = pd.read_pickle(SRC / "oof_predictions.pkl")
with open(BASE / "artifacts" / "candidates.pkl", "rb") as f:
    blob = pickle.load(f)
cand, rand_t1, rand_t3 = blob["cand"], blob["rand_t1"], blob["rand_t3"]

assert set(SEEDS) <= set(oof["seed"].unique()), "seed mismatch"
assert SCHEME in set(oof["scheme"].unique()), f"scheme missing: {SCHEME}"

# Precompute per-seed lookup tables (avoid re-filtering in inner loop)
tabs = {}
for s in SEEDS:
    t = oof[(oof.scheme == SCHEME) & (oof.seed == s)].set_index("reaction_number")
    assert len(t) == 3504, f"seed {s} rows = {len(t)}"
    tabs[s] = t

rows, per_group = [], []
for ch in CHANNELS:
    t1_all, t3_all, sp_all = [], [], []
    for s in SEEDS:
        t = tabs[s]
        for key, members in cand.items():
            y_true = t.loc[members, ch + "_true"].to_numpy()
            y_pred = t.loc[members, ch + "_pred"].to_numpy()

            # Lower is better for all 7 channels → argmin picks the winner.
            best_true = int(np.argmin(y_true))
            best_pred = int(np.argmin(y_pred))
            top1 = (best_true == best_pred)
            top3 = best_true in np.argsort(y_pred)[:TOPK].tolist()

            t1_all.append(top1)
            t3_all.append(top3)
            sp = np.nan
            if len(members) >= MIN_SPEAR:
                sp = spearmanr(y_true, y_pred).statistic
                sp_all.append(sp)

            per_group.append(dict(channel=ch, seed=s, key=key,
                                  n_cand=len(members),
                                  top1=bool(top1), top3=bool(top3),
                                  spearman=sp))

    rows.append(dict(
        channel=ch,
        top1=float(np.mean(t1_all)),
        top3=float(np.mean(t3_all)),
        spearman=float(np.nanmean(sp_all)),
        n_top=len(t1_all),
        n_spearman=len(sp_all),
        random_top1=rand_t1,
        random_top3=rand_t3,
        lift_top1=float(np.mean(t1_all)) / rand_t1,
    ))

res = pd.DataFrame(rows)
pg = pd.DataFrame(per_group)
res.to_csv(BASE / "artifacts" / "topk_results.csv", index=False)
pg.to_csv(BASE / "artifacts" / "topk_per_group.csv", index=False)

pd.set_option("display.width", 140)
print(res.round(4).to_string(index=False))

# Gate check vs SPEC §7 targets (4 decimals)
expected = {
    "d1_own_dft": (0.7167, 0.9750, 0.8426),
    "d2_own_dft": (0.7208, 0.9681, 0.7990),
    "elst_dft":   (0.5694, 0.9292, 0.6844),
    "pauli_dft":  (0.6597, 0.9431, 0.7592),
    "oi_dft":     (0.6556, 0.9417, 0.7592),
    "disp_dft":   (0.6389, 0.9583, 0.7845),
    "cpcm_dft":   (0.7264, 0.9347, 0.7312),
}
TOL = 5e-4
mismatches = []
for _, r in res.iterrows():
    exp = expected[r.channel]
    for name, actual, target in zip(("top1", "top3", "spearman"),
                                     (r.top1, r.top3, r.spearman),
                                     exp):
        if abs(actual - target) > TOL:
            mismatches.append(f"{r.channel}/{name} {actual:.4f} != {target:.4f}")

n_top_ok = (res["n_top"] == 720).all()
n_spear_ok = (res["n_spearman"] == 595).all()
status = "PASS"
if mismatches or not n_top_ok or not n_spear_ok:
    status = "FAIL"
with open(BASE / "artifacts" / "GATE2_STATUS.txt", "w") as f:
    f.write(f"{status} n_top_all_720={n_top_ok} n_spear_all_595={n_spear_ok}\n")
    if mismatches:
        f.write("mismatches:\n")
        for m in mismatches:
            f.write(f"  {m}\n")

print(f"\n=== GATE-2 {status} ===")
if mismatches:
    for m in mismatches:
        print(f"  MISMATCH: {m}")
    raise SystemExit(1)
