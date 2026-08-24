#!/usr/bin/env python3
"""Step 2: cross-validate substituents against known ds3 library (GATE-1).

Ground-truth library from coleygroup/dipolar_cycloaddition_dataset construction
notebooks (dipole termini + dipolarophile subs). 12 unique canonical R-groups.
"""
from pathlib import Path
from collections import Counter
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


def canon(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    return Chem.MolToSmiles(m, canonical=True)


def canon_dummy(smi: str) -> str:
    """Canonical SMILES with dummy [*] retained."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    for a in m.GetAtoms():
        if a.GetSymbol() == "*":
            a.SetAtomMapNum(0)
            a.SetIsotope(0)
    return Chem.MolToSmiles(m, canonical=True)


def main():
    frags = pd.read_pickle(OUT / "fragments.pkl")
    print(f"Fragments: {len(frags)} rows")

    known = {canon_dummy(smi): name for name, smi in KNOWN_SUBS_RAW}
    print(f"Known library: {len(known)} unique canonical R-groups")
    for smi, name in known.items():
        print(f"  {name:12s} -> {smi}")

    # For each fragment cut, take the smaller side as the "substituent candidate"
    def smaller_side(row):
        if row["frag_a_heavy"] <= row["frag_b_heavy"]:
            return row["frag_a_smiles"], row["frag_a_heavy"]
        return row["frag_b_smiles"], row["frag_b_heavy"]

    sides = frags.apply(smaller_side, axis=1, result_type="expand")
    sides.columns = ["sub_smi", "sub_heavy"]
    frags2 = pd.concat([frags[["rxn_id"]].reset_index(drop=True), sides.reset_index(drop=True)], axis=1)
    frags2["sub_canon"] = frags2["sub_smi"].apply(canon_dummy)
    frags2["is_known"] = frags2["sub_canon"].isin(known)

    # Filter to plausible substituents (heavy atom count <= SUB_MAX_HEAVY)
    plaus = frags2[frags2["sub_heavy"] <= SUB_MAX_HEAVY].copy()
    print(f"Plausible substituents (heavy <= {SUB_MAX_HEAVY}): {len(plaus)} of {len(frags2)}")
    hit_rate = plaus["is_known"].mean() * 100
    print(f"Known-library hit rate: {hit_rate:.2f}%")

    inv = plaus.groupby("sub_canon").size().reset_index(name="n")
    inv["known_name"] = inv["sub_canon"].map(known).fillna("")
    inv = inv.sort_values("n", ascending=False)
    inv.to_csv(OUT / "substituent_inventory.csv", index=False)
    print(f"Substituent inventory saved: {len(inv)} unique R-groups")

    unknown = plaus[~plaus["is_known"]].groupby("sub_canon").size().reset_index(name="n")
    unknown = unknown.sort_values("n", ascending=False)
    unknown.to_csv(OUT / "unknown_subs.csv", index=False)
    print(f"Unknown substituents (in library-plausible range): {len(unknown)} unique, top:")
    print(unknown.head(20).to_string(index=False))

    if hit_rate < 95.0:
        print(f"\n=== GATE-1 FAIL: hit_rate={hit_rate:.2f}% < 95% ===")
        print("Review artifacts/unknown_subs.csv before continuing.")
        # Don't hard-fail; report and let orchestrator decide
        with open(OUT / "GATE1_STATUS.txt", "w") as f:
            f.write(f"FAIL hit_rate={hit_rate:.2f}%\n")
    else:
        print(f"\n=== GATE-1 PASS: hit_rate={hit_rate:.2f}% >= 95% ===")
        with open(OUT / "GATE1_STATUS.txt", "w") as f:
            f.write(f"PASS hit_rate={hit_rate:.2f}%\n")


if __name__ == "__main__":
    main()
