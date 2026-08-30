#!/usr/bin/env python3
"""SPEC15 Step 6 — validate strain calculation methodology.

Four checks (GATE-5, gate to full 5269-rxn run):
  1. Internal consistency: (dist − rel) × 627.5095 == d_b3lyp exactly
  2. Existing strain.json obeys the same formula (sanity vs old pipeline)
  3. No negative distortion energies (would indicate reactant mismatch)
  4. d1/d2 correlate between AM1 and B3LYP labels (r > 0.5 required)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_relabel")
COH = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")
HARTREE_TO_KCAL = 627.5094740631


def main():
    new = pd.read_csv(BASE / "artifacts" / "labels_b3lyp.csv").set_index("rxn_id")

    # (1) internal consistency
    lhs1 = (new["e_frag1_dist_eh"] - new["e_frag1_rel_eh"]) * HARTREE_TO_KCAL
    lhs2 = (new["e_frag2_dist_eh"] - new["e_frag2_rel_eh"]) * HARTREE_TO_KCAL
    max1 = float((lhs1 - new["d1_b3lyp"]).abs().max())
    max2 = float((lhs2 - new["d2_b3lyp"]).abs().max())
    check1 = max(max1, max2) < 1e-6
    print(f"[1] internal consistency: max|Δ| = {max(max1, max2):.2e}  {'OK' if check1 else 'FAIL'}")

    # (2) existing strain.json obeys same formula
    sample = list(new.index)[:20]
    n_ok = 0; n_total = 0
    for rid in sample:
        p = COH / "reactions" / f"rxn_{int(rid):04d}" / "strain.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        recomputed = (j["e_frag1_dist_tzvp_eh"] - j["e_frag1_rel_tzvp_eh"]) * HARTREE_TO_KCAL
        n_total += 1
        if abs(recomputed - j["distortion_energy_1_dft"]) < 1e-6:
            n_ok += 1
    check2 = (n_total > 0 and n_ok == n_total)
    print(f"[2] existing strain.json formula sanity: {n_ok}/{n_total}  {'OK' if check2 else 'FAIL'}")

    # (3) no negative distortion energies
    neg = new[(new["d1_b3lyp"] < 0) | (new["d2_b3lyp"] < 0)]
    check3 = len(neg) == 0
    print(f"[3] negative distortions: {len(neg)}  {'OK' if check3 else 'FAIL'}")
    if len(neg):
        print(f"    first 10 rxn_ids: {list(neg.index)[:10]}")

    # (4) AM1 vs B3LYP d1/d2 correlation
    old = pd.read_pickle(COH / "phase5_dataset_v2.pkl")
    old["reaction_number"] = old["reaction_number"].astype(int)
    old = old.set_index("reaction_number")
    idx = new.index.intersection(old.index)

    print(f"\n=== d1, d2 shift (AM1 → B3LYP, n={len(idx)}) ===")
    check4_ok = True
    for c_new, c_old in [("d1_b3lyp", "d1_own_dft"), ("d2_b3lyp", "d2_own_dft")]:
        a = old.loc[idx, c_old].values.astype(float)
        b = new.loc[idx, c_new].values.astype(float)
        m = ~(np.isnan(a) | np.isnan(b))
        a, b = a[m], b[m]
        r = float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 else float("nan")
        mae = float(np.abs(b - a).mean())
        print(f"  {c_old:>12}: AM1 mean {a.mean():7.2f}  B3LYP mean {b.mean():7.2f}  "
              f"MAE {mae:6.2f}  r={r:.3f}")
        if r <= 0.5:
            check4_ok = False
    print(f"[4] d1/d2 correlation > 0.5: {'OK' if check4_ok else 'FAIL'}")

    status = "PASS" if (check1 and check2 and check3 and check4_ok) else "FAIL"
    with open(BASE / "artifacts" / "GATE5_STATUS.txt", "w") as f:
        f.write(f"{status} internal={check1} sanity={check2} nonneg={check3} corr={check4_ok}\n")
    print(f"\n=== GATE-5 {status} ===")


if __name__ == "__main__":
    main()
