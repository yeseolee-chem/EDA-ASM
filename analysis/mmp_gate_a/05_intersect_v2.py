#!/usr/bin/env python3
"""Step 5 (v2): label intersection on v2 pairs → GATE-4-rev + Q1 answer."""
from pathlib import Path
from collections import Counter
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a")
OUT = BASE / "artifacts"

CHANNELS = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds"]
COL = {"d1": "d1_own", "d2": "d2_own",
       "elst": "elst_dft", "pauli": "pauli_dft", "oi": "oi_dft",
       "disp": "disp_dft", "cpcm": "cpcm_dft", "cds": "cds_dft"}


def main():
    cohort = pd.read_pickle(OUT / "cohort_join.pkl").set_index("reaction_number")
    pairs = pd.read_pickle(OUT / "mmp_pairs_v2.pkl")
    print(f"MMP pairs (v2): {len(pairs)}")

    for ch, col in COL.items():
        pairs[f"has_{ch}"] = (pairs["rxn_id_A"].map(lambda x: pd.notna(cohort.loc[x, col])) &
                              pairs["rxn_id_B"].map(lambda x: pd.notna(cohort.loc[x, col])))
        pairs[f"{ch}_A"] = pairs["rxn_id_A"].map(cohort[col])
        pairs[f"{ch}_B"] = pairs["rxn_id_B"].map(cohort[col])

    per_ch = {ch: int(pairs[f"has_{ch}"].sum()) for ch in CHANNELS}
    all8 = int(pairs[[f"has_{ch}" for ch in CHANNELS]].all(axis=1).sum())

    print("Per-channel coverage:")
    for ch, n in per_ch.items():
        print(f"  {ch:6s}: {n}")
    print(f"All-8-channel pairs: {all8}")

    trans = Counter(zip(pairs["sub_A"], pairs["sub_B"]))
    print("Top 15 substituent transitions (v2):")
    for (a, b), n in sorted(trans.items(), key=lambda x: -x[1])[:15]:
        print(f"  {n:5d}  {a}  ->  {b}")

    with open(OUT / "Q1_ANSWER_v2.txt", "w") as f:
        f.write(f"Q1 (v2): 8-channel-complete MMP pairs = {all8}\n")
        f.write(f"Total pairs: {len(pairs)}\n")
        for ch, n in per_ch.items():
            f.write(f"  {ch}: {n}\n")

    if all8 >= 300:
        stat = "PASS"
    elif all8 >= 100:
        stat = "CONDITIONAL"
    else:
        stat = "FAIL"
    with open(OUT / "GATE4_STATUS_v2.txt", "w") as f:
        f.write(f"{stat} n_all8={all8}\n")
    pairs.to_pickle(OUT / "mmp_pairs_labeled_v2.pkl")
    print(f"=== GATE-4-rev: {stat} (n_all8={all8}) ===")


if __name__ == "__main__":
    main()
