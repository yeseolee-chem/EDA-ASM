#!/usr/bin/env python3
"""Step 5: label intersection - GATE-4.

For each MMP pair, check that both reactions carry each of the 8 channels.
Report Q1: N pairs with all-8-channel coverage.
"""
from pathlib import Path
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a")
OUT = BASE / "artifacts"

CHANNELS = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds"]
COL = {"d1": "d1_own", "d2": "d2_own",
       "elst": "elst_dft", "pauli": "pauli_dft", "oi": "oi_dft",
       "disp": "disp_dft", "cpcm": "cpcm_dft", "cds": "cds_dft"}


def main():
    cohort = pd.read_pickle(OUT / "cohort_join.pkl").set_index("reaction_number")
    pairs = pd.read_pickle(OUT / "mmp_pairs.pkl")
    print(f"MMP pairs: {len(pairs)}  cohort rows: {len(cohort)}")

    for ch, col in COL.items():
        aok = pairs["rxn_id_A"].map(lambda x: pd.notna(cohort.loc[x, col]))
        bok = pairs["rxn_id_B"].map(lambda x: pd.notna(cohort.loc[x, col]))
        pairs[f"has_{ch}"] = aok & bok
        # Also copy the channel values for downstream
        pairs[f"{ch}_A"] = pairs["rxn_id_A"].map(cohort[col])
        pairs[f"{ch}_B"] = pairs["rxn_id_B"].map(cohort[col])

    # Overall
    per_ch = {ch: int(pairs[f"has_{ch}"].sum()) for ch in CHANNELS}
    all8 = int(pairs[[f"has_{ch}" for ch in CHANNELS]].all(axis=1).sum())

    print("Per-channel pair coverage:")
    for ch, n in per_ch.items():
        pct = n / len(pairs) * 100 if len(pairs) else 0
        print(f"  {ch:6s}: {n:6d} ({pct:5.1f}%)")
    print(f"All-8-channel pairs: {all8}")

    # top substituent transitions
    from collections import Counter
    trans = Counter(zip(pairs["sub_A"], pairs["sub_B"]))
    print("Top 20 substituent transitions:")
    for (a, b), n in sorted(trans.items(), key=lambda x: -x[1])[:20]:
        print(f"  {n:5d}  {a}  ->  {b}")

    # Q1 answer
    with open(OUT / "Q1_ANSWER.txt", "w") as f:
        f.write(f"Q1: 8-channel-complete MMP pairs = {all8}\n")
        f.write(f"Total pairs: {len(pairs)}\n")
        for ch, n in per_ch.items():
            f.write(f"  {ch}: {n}\n")

    pairs.to_pickle(OUT / "mmp_pairs_labeled.pkl")
    print(f"\n=== Q1 ANSWER: {all8} pairs with all 8 channels ===")

    if all8 >= 300:
        gate4 = "PASS"
    elif all8 >= 100:
        gate4 = "CONDITIONAL"
    else:
        gate4 = "FAIL"
    with open(OUT / "GATE4_STATUS.txt", "w") as f:
        f.write(f"{gate4} n_all8={all8}\n")
    print(f"=== GATE-4: {gate4} (n_all8={all8}) ===")


if __name__ == "__main__":
    main()
