#!/usr/bin/env python3
"""Step 2 (v2 patch): library hit rate — report-only + core-adjacent GATE-1b.

§4 of rev SPEC:
  - GATE-1 (>=95% total hit) demoted to REPORT (was mis-designed)
  - GATE-1b: hit rate on core-ADJACENT cuts (sub bonded directly to core atom)
             must be >= 95%
"""
from pathlib import Path
import pandas as pd
from rdkit import Chem

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a")
OUT = BASE / "artifacts"
SUB_MAX_HEAVY = 8

KNOWN_SUBS_RAW = [
    ("H",         "[*][H]"),
    ("CH3",       "[*]C"),
    ("CN",        "[*]C#N"),
    ("CO2Me",     "[*]C(=O)OC"),
    ("C(=O)Me",   "[*]C(C)=O"),
    ("C(=O)NHMe", "[*]C(=O)NC"),
    ("Ph",        "[*]c1ccccc1"),
    ("F",         "[*]F"),
    ("Cl",        "[*]Cl"),
    ("Br",        "[*]Br"),
    ("OMe",       "[*]OC"),
    ("CF3",       "[*]C(F)(F)F"),
]


def canon_dummy(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    for a in m.GetAtoms():
        if a.GetSymbol() == "*":
            a.SetAtomMapNum(0); a.SetIsotope(0)
    return Chem.MolToSmiles(m, canonical=True)


def main():
    frags = pd.read_pickle(OUT / "fragments_v2.pkl")
    print(f"Fragments (v2): {len(frags)}")

    known = {canon_dummy(smi): name for name, smi in KNOWN_SUBS_RAW}
    print(f"Known library: {len(known)} unique canonical R-groups")

    # Consider the smaller side as the substituent (by heavy count)
    def smaller(row):
        if row["frag_a_heavy"] <= row["frag_b_heavy"]:
            return row["frag_a_smiles"], row["frag_a_heavy"], row["frag_a_has_core"], row["frag_b_has_core"]
        return row["frag_b_smiles"], row["frag_b_heavy"], row["frag_b_has_core"], row["frag_a_has_core"]

    tmp = frags.apply(smaller, axis=1, result_type="expand")
    tmp.columns = ["sub_smi", "sub_heavy", "sub_has_core", "other_has_core"]
    df = pd.concat([frags[["rxn_id"]].reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)
    df["sub_canon"] = df["sub_smi"].apply(canon_dummy)
    df["is_known"] = df["sub_canon"].isin(known)
    df["is_core_adjacent"] = (~df["sub_has_core"]) & (df["other_has_core"])

    plaus = df[df["sub_heavy"] <= SUB_MAX_HEAVY]
    total_hit = plaus["is_known"].mean() * 100 if len(plaus) else 0.0
    print(f"[REPORT] Total-hit rate (sub_heavy<={SUB_MAX_HEAVY}): {total_hit:.2f}%  (n={len(plaus)})")
    print(f"  (rev SPEC §4: this is a report-only metric — mid-substituent cuts pollute it)")

    core_adj = plaus[plaus["is_core_adjacent"]]
    if len(core_adj):
        adj_hit = core_adj["is_known"].mean() * 100
    else:
        adj_hit = 0.0
    print(f"[GATE-1b] Core-adjacent hit rate: {adj_hit:.2f}%  (n={len(core_adj)})")

    inv = plaus.groupby("sub_canon").size().reset_index(name="n")
    inv["known_name"] = inv["sub_canon"].map(known).fillna("")
    core_adj_counts = core_adj.groupby("sub_canon").size().reset_index(name="n_core_adjacent")
    inv = inv.merge(core_adj_counts, on="sub_canon", how="left").fillna({"n_core_adjacent": 0})
    inv = inv.sort_values("n", ascending=False)
    inv.to_csv(OUT / "substituent_inventory_v2.csv", index=False)

    unknown = plaus[~plaus["is_known"]].groupby("sub_canon").size().reset_index(name="n").sort_values("n", ascending=False)
    unknown.to_csv(OUT / "unknown_subs_v2.csv", index=False)
    print(f"Top 15 unknown subs (any position):")
    print(unknown.head(15).to_string(index=False))

    with open(OUT / "GATE1_STATUS_v2.txt", "w") as f:
        f.write(f"REPORT total_hit_rate={total_hit:.2f}%\n")
    if adj_hit >= 95.0:
        stat = "PASS"
    else:
        stat = "FAIL"
    with open(OUT / "GATE1b_STATUS.txt", "w") as f:
        f.write(f"{stat} core_adjacent_hit_rate={adj_hit:.2f}% n={len(core_adj)}\n")
    print(f"\n=== GATE-1b: {stat} core-adjacent hit={adj_hit:.2f}% ===")


if __name__ == "__main__":
    main()
