#!/usr/bin/env python3
"""SPEC13 Step 1 — build candidate groups (same-key reactions).

MMP key groups reactions sharing the same scaffold + solvent + temp.
Only groups of size >= MIN_CAND are retained.

GATE-1 targets (SPEC §7):
  n_groups   = 144
  mean size  = 6.0069
  median     = 4
  size range = 3..18
  size >= 4  = 119
  random Top-1 expected = 0.2072
  random Top-3 expected = 0.6215
"""
import collections
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/topk_retrieval")
SRC = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/delta_mae_baseline/artifacts")
MIN_CAND = 3

pairs = pd.read_pickle(SRC / "pairs_dedup.pkl")
assert len(pairs) == 1936, f"pairs_dedup rows={len(pairs)}, expected 1936"
for col in ["key", "rxn_id_A", "rxn_id_B"]:
    assert col in pairs.columns, f"missing column: {col}"

# Group reactions by MMP key
groups = collections.defaultdict(set)
for _, r in pairs.iterrows():
    groups[r["key"]].add(int(r["rxn_id_A"]))
    groups[r["key"]].add(int(r["rxn_id_B"]))

# Keep groups of size >= MIN_CAND; sort members for deterministic order
cand = {k: sorted(v) for k, v in groups.items() if len(v) >= MIN_CAND}

sizes = np.array([len(v) for v in cand.values()])
n_groups = len(cand)
mean_sz = float(sizes.mean())
med_sz = float(np.median(sizes))
n_spear_eligible = int((sizes >= 4).sum())
rand_t1 = float(np.mean(1.0 / sizes))
rand_t3 = float(np.mean(np.minimum(3, sizes) / sizes))

print(f"n_groups        = {n_groups}")
print(f"mean size       = {mean_sz:.4f}")
print(f"median size     = {med_sz:.0f}")
print(f"size range      = {sizes.min()} .. {sizes.max()}")
print(f"size >= 4       = {n_spear_eligible}")
print(f"random Top-1    = {rand_t1:.4f}")
print(f"random Top-3    = {rand_t3:.4f}")

# Save
with open(BASE / "artifacts" / "candidates.pkl", "wb") as f:
    pickle.dump({"cand": cand, "rand_t1": rand_t1, "rand_t3": rand_t3}, f)

dist = pd.Series(collections.Counter(sizes.tolist())).sort_index()
dist.index.name = "n_candidates"
dist.name = "n_groups"
dist.to_csv(BASE / "artifacts" / "candidate_size_dist.csv")
print("\nSize distribution:")
print(dist.to_string())

# Gate assertions
expected = dict(n_groups=144, mean_size=6.0069, median_size=4,
                size_min=3, size_max=18, n_spear=119,
                rand_t1=0.2072, rand_t3=0.6215)
diagnostics = dict(
    n_groups=n_groups, mean_size=mean_sz, median_size=med_sz,
    size_min=int(sizes.min()), size_max=int(sizes.max()),
    n_spear=n_spear_eligible, rand_t1=rand_t1, rand_t3=rand_t3,
)
status = "PASS"
mismatches = []
if diagnostics["n_groups"] != expected["n_groups"]:
    status = "FAIL"; mismatches.append(f"n_groups {diagnostics['n_groups']} != 144")
if abs(diagnostics["mean_size"] - expected["mean_size"]) > 5e-4:
    status = "FAIL"; mismatches.append(
        f"mean_size {diagnostics['mean_size']:.4f} != 6.0069")
if diagnostics["n_spear"] != expected["n_spear"]:
    status = "FAIL"; mismatches.append(f"n_spear {diagnostics['n_spear']} != 119")
if abs(diagnostics["rand_t1"] - expected["rand_t1"]) > 5e-4:
    status = "FAIL"; mismatches.append(
        f"rand_t1 {diagnostics['rand_t1']:.4f} != 0.2072")
if abs(diagnostics["rand_t3"] - expected["rand_t3"]) > 5e-4:
    status = "FAIL"; mismatches.append(
        f"rand_t3 {diagnostics['rand_t3']:.4f} != 0.6215")

with open(BASE / "artifacts" / "GATE1_STATUS.txt", "w") as f:
    f.write(f"{status} " + " ".join(f"{k}={v}" for k, v in diagnostics.items()))
    if mismatches:
        f.write("\nmismatches: " + "; ".join(mismatches))
    f.write("\n")
print(f"\n=== GATE-1 {status} ===")
if mismatches:
    for m in mismatches:
        print(f"  MISMATCH: {m}")
    raise SystemExit(1)
