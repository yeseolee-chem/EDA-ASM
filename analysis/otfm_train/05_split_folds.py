#!/usr/bin/env python3
"""SPEC17rev2 Step 5 — scaffold-grouped 5-fold split.

GATE-5: scaffold spanning more than one fold count == 0.
        fold size max/min ratio < 1.5.

Scaffold key = (dot-joined reactant SMILES with map numbers stripped)
             | (elements on smallest product ring covering both new bonds)
             | (frozenset of formed-bond pairs)
"""
from __future__ import annotations

import collections
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis/otfm_train"
CSV_PATH = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")

SEED = 42
N_FOLDS = 5


def scaffold_key(rxn_smiles: str):
    try:
        reac, prod = rxn_smiles.split(">>")
    except ValueError:
        return None
    R = Chem.MolFromSmiles(reac)
    P = Chem.MolFromSmiles(prod)
    if R is None or P is None:
        return None

    def bondset(m):
        s = set()
        for b in m.GetBonds():
            i = b.GetBeginAtom().GetAtomMapNum()
            j = b.GetEndAtom().GetAtomMapNum()
            if i and j:
                s.add((min(i, j), max(i, j)))
        return s

    formed = bondset(P) - bondset(R)
    if len(formed) != 2:
        return None
    m2i = {a.GetAtomMapNum(): a.GetIdx() for a in P.GetAtoms()}
    fidx = {frozenset((m2i[i], m2i[j])) for i, j in formed}
    best = None
    for ring in P.GetRingInfo().AtomRings():
        rs = set(ring)
        if all(all(x in rs for x in fb) for fb in fidx):
            if best is None or len(ring) < len(best):
                best = ring
    if best is None:
        return None
    sig = "".join(sorted(P.GetAtomWithIdx(i).GetSymbol() for i in best))

    def strip(s):
        m = Chem.MolFromSmiles(s)
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        return Chem.MolToSmiles(m)

    parts = sorted(strip(p) for p in reac.split("."))
    return f"{'.'.join(parts)}|{sig}|{sorted(formed)}"


def main() -> int:
    coley_pkl = BASE / "data" / "coley_all.pkl"
    if not coley_pkl.exists():
        print(f"[FATAL] {coley_pkl} missing. Run Step 4.", file=sys.stderr)
        return 1
    if not CSV_PATH.exists():
        print(f"[FATAL] CSV missing: {CSV_PATH}", file=sys.stderr)
        return 1

    ds = pd.read_csv(CSV_PATH).set_index("rxn_id")
    with open(coley_pkl, "rb") as f:
        data = pickle.load(f)
    rxn_ids = list(data["rxn_id"])
    print(f"assigning folds for {len(rxn_ids)} rxns")

    keys = {rid: scaffold_key(ds.at[rid, "rxn_smiles"]) for rid in rxn_ids}
    groups = collections.defaultdict(list)
    for rid, k in keys.items():
        groups[k if k is not None else f"__nokey_{rid}"].append(rid)
    print(f"scaffolds: {len(groups)}   rxns: {len(rxn_ids)}")

    # greedy: largest scaffold first, into currently-smallest fold
    fold_of, sizes = {}, [0] * N_FOLDS
    gs = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for k, members in gs:
        f_idx = int(np.argmin(sizes))
        for rid in members:
            fold_of[rid] = f_idx
        sizes[f_idx] += len(members)

    S = pd.DataFrame({
        "rxn_id": rxn_ids,
        "fold": [fold_of[r] for r in rxn_ids],
        "scaffold": [keys[r] for r in rxn_ids],
    })
    S.to_csv(BASE / "artifacts" / "folds.csv", index=False)
    counts = S.fold.value_counts().sort_index()
    print("fold sizes:")
    print(counts.to_string())

    leak = S.groupby("scaffold").fold.nunique()
    leak = leak[leak > 1]
    print(f"\nscaffolds spanning multiple folds: {len(leak)}  (must be 0)")

    max_ratio = counts.max() / max(1, counts.min())
    passed = (len(leak) == 0) and (max_ratio < 1.5)
    (BASE / "artifacts" / "GATE5_STATUS.txt").write_text(
        ("PASS" if passed else "FAIL") + "\n"
        f"n_scaffolds={len(groups)}\n"
        f"leak_scaffolds={len(leak)}\n"
        f"fold_sizes={list(counts.values)}\n"
        f"max_min_ratio={max_ratio:.3f}\n"
    )
    if not passed:
        print("[GATE-5 FAIL] leak or imbalanced folds", file=sys.stderr)
        return 1
    print("=== GATE-5 PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
