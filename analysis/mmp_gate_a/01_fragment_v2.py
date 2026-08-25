#!/usr/bin/env python3
"""Step 1 (v2 patch): enumerate cuts + tag core atom presence per fragment.

Patch over 01_fragment.py:
  - Defer strip_map() until AFTER fragmentation
  - Compute core atom maps via reactant/product bond diff → smallest ring
    containing both formed bonds (§2 of rev SPEC)
  - Tag each fragment with has_core + core_signature
"""
from pathlib import Path
import pandas as pd
from rdkit import Chem

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a")
OUT = BASE / "artifacts"


def core_atom_maps(rxn_smiles):
    """Return set of atom map numbers forming the reaction core (smallest ring
    in product containing both newly formed bonds). None on failure.
    """
    reac, prod = rxn_smiles.split(">>")
    R, P = Chem.MolFromSmiles(reac), Chem.MolFromSmiles(prod)
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

    m2i = {a.GetAtomMapNum(): a.GetIdx() for a in P.GetAtoms() if a.GetAtomMapNum() > 0}
    try:
        fidx = {frozenset((m2i[i], m2i[j])) for i, j in formed}
    except KeyError:
        return None

    best = None
    for ring in P.GetRingInfo().AtomRings():
        rs = set(ring)
        if all(all(x in rs for x in fb) for fb in fidx):
            if best is None or len(ring) < len(best):
                best = ring
    if best is None:
        return None

    i2m = {a.GetIdx(): a.GetAtomMapNum() for a in P.GetAtoms()}
    return {i2m[i] for i in best}


def core_signature(rxn_smiles, core_maps):
    """Sorted atom-symbol string of core atoms (e.g., 'CCCCN')."""
    P = Chem.MolFromSmiles(rxn_smiles.split(">>")[1])
    syms = sorted(a.GetSymbol() for a in P.GetAtoms() if a.GetAtomMapNum() in core_maps)
    return "".join(syms)


def enumerate_cuts(mol_with_map, core_maps):
    """Yield dicts per single-cut with core-presence tag.

    mol_with_map: reactant molecule with atom maps preserved
    core_maps: set of atom map numbers of the reaction core
    """
    mol = Chem.AddHs(mol_with_map)  # AddHs preserves atom maps on heavy atoms
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        s1, s2 = a1.GetSymbol(), a2.GetSymbol()
        if s1 == "H" and s2 == "H":
            continue
        is_ch_or_nh = (s1 == "H") or (s2 == "H")
        try:
            frag_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True,
                                             dummyLabels=[(1, 2)])
            frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
        except Exception:
            continue
        if len(frags) != 2:
            continue

        frag_info = []
        ok = True
        for f in frags:
            # Core detection BEFORE stripping map
            f_maps = set(a.GetAtomMapNum() for a in f.GetAtoms()
                         if a.GetAtomMapNum() > 0 and a.GetSymbol() != "*")
            has_core = bool(f_maps & core_maps)
            heavy = sum(1 for a in f.GetAtoms() if a.GetSymbol() not in ("H", "*"))
            # Now strip map + canonicalize
            for a in f.GetAtoms():
                a.SetAtomMapNum(0)
                if a.GetSymbol() == "*":
                    a.SetIsotope(0)
            try:
                Chem.SanitizeMol(f, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
                smi = Chem.MolToSmiles(f, canonical=True)
            except Exception:
                ok = False
                break
            frag_info.append({"smi": smi, "has_core": has_core, "heavy": heavy})
        if not ok:
            continue
        yield {
            "bond_idx": bond.GetIdx(),
            "frag_a_smiles": frag_info[0]["smi"],
            "frag_b_smiles": frag_info[1]["smi"],
            "frag_a_has_core": frag_info[0]["has_core"],
            "frag_b_has_core": frag_info[1]["has_core"],
            "frag_a_heavy": frag_info[0]["heavy"],
            "frag_b_heavy": frag_info[1]["heavy"],
            "is_ch_or_nh": is_ch_or_nh,
        }


def main():
    cohort = pd.read_pickle(OUT / "cohort_join.pkl")
    print(f"Cohort: {len(cohort)} reactions")

    # GATE-1a: core detection
    core_info = []
    for _, r in cohort.iterrows():
        rxn = int(r["reaction_number"])
        cmaps = core_atom_maps(r["rxn_smiles"])
        sig = core_signature(r["rxn_smiles"], cmaps) if cmaps else None
        core_info.append({"rxn_id": rxn, "core_maps": cmaps,
                          "core_size": len(cmaps) if cmaps else 0,
                          "core_signature": sig})
    core_df = pd.DataFrame(core_info)
    n_ok = (core_df["core_size"] == 5).sum()
    n_total = len(core_df)
    print(f"GATE-1a: core detected 5-membered ring in {n_ok}/{n_total} reactions")
    sizes = core_df["core_size"].value_counts().sort_index()
    print("core_size distribution:")
    print(sizes.to_string())
    with open(OUT / "GATE1a_STATUS.txt", "w") as f:
        if n_ok == n_total:
            f.write(f"PASS {n_ok}/{n_total} 5-membered cores\n")
        else:
            f.write(f"FAIL {n_ok}/{n_total} 5-membered cores\n")
    assert n_ok == n_total, f"GATE-1a FAIL: {n_ok}/{n_total} — cannot proceed"

    core_map = {r["rxn_id"]: (r["core_maps"], r["core_signature"]) for _, r in core_df.iterrows()}
    core_df.drop(columns=["core_maps"]).to_pickle(OUT / "core_v2.pkl")

    # Enumerate cuts (map-preserving)
    rows = []
    stats_no_cut = 0
    for _, r in cohort.iterrows():
        rxn = int(r["reaction_number"])
        cmaps, csig = core_map[rxn]
        # Split reactant side (keep atom maps intact)
        reac = r["rxn_smiles"].split(">>")[0]
        parts = reac.split(".")
        n_this = 0
        for mol_idx, part in enumerate(parts):
            m = Chem.MolFromSmiles(part)  # keep atom maps
            if m is None:
                continue
            for cut in enumerate_cuts(m, cmaps):
                rows.append({
                    "rxn_id": rxn, "mol_idx": mol_idx,
                    "core_signature": csig,
                    **cut,
                })
                n_this += 1
        if n_this == 0:
            stats_no_cut += 1

    df = pd.DataFrame(rows)
    df.to_pickle(OUT / "fragments_v2.pkl")
    print(f"Fragments (v2): {len(df)} rows over {df['rxn_id'].nunique()} reactions")
    print(f"Reactions with 0 cuts: {stats_no_cut}")
    print(f"Mean cuts / reaction: {len(df)/n_total:.1f}")
    both_core = ((df["frag_a_has_core"] & df["frag_b_has_core"])).sum()
    neither = ((~df["frag_a_has_core"] & ~df["frag_b_has_core"])).sum()
    only_one = len(df) - both_core - neither
    print(f"Cuts by core presence: only-one={only_one}  both={both_core}  neither={neither}")


if __name__ == "__main__":
    main()
