#!/usr/bin/env python3
"""Step 1: enumerate all single-cut fragments per reactant molecule.

Chosen rule (§3.3): CUT_MODE = all_acyclic_single, single cut only, INCLUDE_H_SUBS = True.
Reason: default RDKit SMARTS excludes hypervalent atoms + multi-bonded C -> misses
our C#N, C(=O)Me, CO2Me, etc. All acyclic single bonds (including C-H, N-H) captured.

Input:  artifacts/cohort_join.pkl
Output: artifacts/fragments.pkl
        rows: (rxn_id, mol_idx, bond_idx, frag_a_smiles, frag_b_smiles,
               frag_a_heavy, frag_b_heavy, is_ch_or_nh)
"""
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a")
OUT = BASE / "artifacts"


def strip_map(mol: Chem.Mol) -> Chem.Mol:
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return mol


def canon(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    return Chem.MolToSmiles(m, canonical=True)


def enumerate_cuts(mol_no_h: Chem.Mol):
    """Yield (bond_idx_in_h_mol, frag_a_smi_canon, frag_b_smi_canon, is_ch_or_nh).

    Explicitly adds H so C-H/N-H bonds become cut candidates.
    """
    mol = Chem.AddHs(mol_no_h)
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        s1, s2 = a1.GetSymbol(), a2.GetSymbol()
        # Skip H-H bonds (there aren't any but safe)
        if s1 == "H" and s2 == "H":
            continue
        is_ch_or_nh = (s1 == "H") or (s2 == "H")
        try:
            frag_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True,
                                             dummyLabels=[(1, 2)])
        except Exception:
            continue
        try:
            frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
        except Exception:
            continue
        if len(frags) != 2:
            continue
        try:
            smis = []
            for f in frags:
                # Reset dummy labels to plain [*]
                for a in f.GetAtoms():
                    if a.GetSymbol() == "*":
                        a.SetIsotope(0)
                        a.SetAtomMapNum(0)
                # Remove explicit H display, but keep dummy
                Chem.SanitizeMol(f, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
                smi = Chem.MolToSmiles(f, canonical=True)
                smis.append(smi)
        except Exception:
            continue
        yield (bond.GetIdx(), smis[0], smis[1], is_ch_or_nh)


def heavy_atom_count(smi: str) -> int:
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return -1
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return sum(1 for a in m.GetAtoms() if a.GetSymbol() not in ("H", "*"))


def main():
    cohort = pd.read_pickle(OUT / "cohort_join.pkl")
    print(f"Cohort: {len(cohort)} reactions")

    rows = []
    stats_no_cut = 0
    for _, r in cohort.iterrows():
        rxn = int(r["reaction_number"])
        smi = r["rxn_smiles"].split(">>")[0]
        parts = smi.split(".")
        n_cut_total = 0
        for mol_idx, part in enumerate(parts):
            m = Chem.MolFromSmiles(part)
            if m is None:
                continue
            m = strip_map(m)
            for bond_idx, a_smi, b_smi, is_h in enumerate_cuts(m):
                rows.append({
                    "rxn_id": rxn, "mol_idx": mol_idx, "bond_idx": bond_idx,
                    "frag_a_smiles": a_smi, "frag_b_smiles": b_smi,
                    "frag_a_heavy": heavy_atom_count(a_smi),
                    "frag_b_heavy": heavy_atom_count(b_smi),
                    "is_ch_or_nh": is_h,
                })
                n_cut_total += 1
        if n_cut_total == 0:
            stats_no_cut += 1

    df = pd.DataFrame(rows)
    df.to_pickle(OUT / "fragments.pkl")
    print(f"Fragments: {len(df)} rows over {df['rxn_id'].nunique()} reactions")
    print(f"Reactions with 0 cuts: {stats_no_cut}")
    print(f"Mean cuts / reaction: {len(df)/cohort.shape[0]:.1f}")
    print(f"C-H / N-H cut share: {df['is_ch_or_nh'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
