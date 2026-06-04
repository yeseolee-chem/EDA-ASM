"""Discover cap sites where each fragment needs an H atom.

For ASM-EDA we replace the bond crossing the fragment boundary with an H atom
on each side; this capping module returns the list of (anchor, partner)
pairs but does not place the H atoms in 3-D — that step belongs to the
geometric pipeline (Stage 5) which has the actual coordinates.
"""
from __future__ import annotations

from collections.abc import Iterable

from rdkit import Chem


def find_cap_sites(
    fragments: list[set[int]],
    mol_R: Chem.Mol,
) -> dict[int, list[tuple[int, int]]]:
    """Return {fragment_idx: [(anchor_in_fragment, partner_in_other), ...]}.

    Walks every R-bond once. If the bond crosses a fragment boundary, both
    atoms are recorded as cap-site anchors (one per fragment). Bond order is
    preserved by emitting *order* duplicate cap entries so that double bonds
    yield two H caps each.
    """
    out: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(fragments))}

    def fragment_of(atom: int) -> int | None:
        for idx, frag in enumerate(fragments):
            if atom in frag:
                return idx
        return None

    for bond in mol_R.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        fi = fragment_of(i)
        fj = fragment_of(j)
        if fi is None or fj is None or fi == fj:
            continue
        order = int(bond.GetBondTypeAsDouble())
        for _ in range(max(order, 1)):
            out[fi].append((i, j))
            out[fj].append((j, i))
    return out


def fragment_smiles(
    fragment: Iterable[int],
    mol_R: Chem.Mol,
    cap_sites_for_fragment: list[tuple[int, int]],
) -> str | None:
    """Best-effort SMILES for *fragment* with H caps applied.

    Returns ``None`` if RDKit cannot sanitize the resulting structure.
    """
    rw = Chem.RWMol()
    sub = sorted(fragment)
    sub_set = set(sub)
    idx_map: dict[int, int] = {}
    for old in sub:
        atom = mol_R.GetAtomWithIdx(old)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        idx_map[old] = rw.AddAtom(new_atom)
    # Internal bonds
    for bond in mol_R.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in sub_set and b in sub_set:
            rw.AddBond(idx_map[a], idx_map[b], bond.GetBondType())
    # H caps
    for anchor, _partner in cap_sites_for_fragment:
        if anchor not in idx_map:
            continue
        h_idx = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(idx_map[anchor], h_idx, Chem.BondType.SINGLE)
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None
