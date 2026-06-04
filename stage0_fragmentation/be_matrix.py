"""Bond–Electron (Ugi-Dugundji) matrix construction and validation.

Convention (matches Joung et al. 2025, *Nature*, DOI: 10.1038/s41586-025-09426-9):

- ``B[i, j]`` for ``i != j`` is the bond order between atoms i and j (0/1/2/3).
- ``B[i, i]`` is the **lone-pair electron count** of atom i (electrons, not pairs).
- Aromatic bonds are not allowed: callers must Kekulize first. ``build_be_matrix``
  performs this Kekulization internally to be safe.
- Row sum equals (valence_e[i] − formal_charge[i]); this is enforced in
  ``validate_be_matrix``.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem

# Periodic-table valence electron counts for atoms relevant to organic /
# halogen chemistry. Group numbers are the standard main-group lookups.
VALENCE_ELECTRONS: dict[int, int] = {
    1: 1,                       # H
    2: 2,                       # He
    3: 1, 11: 1, 19: 1,         # Li, Na, K
    4: 2, 12: 2, 20: 2,         # Be, Mg, Ca
    5: 3, 13: 3,                # B, Al
    6: 4, 14: 4,                # C, Si
    7: 5, 15: 5,                # N, P
    8: 6, 16: 6, 34: 6,         # O, S, Se
    9: 7, 17: 7, 35: 7, 53: 7,  # F, Cl, Br, I
    10: 8, 18: 8,               # Ne, Ar
}


class RadicalNotSupportedError(ValueError):
    """Raised when the molecule contains unpaired electrons (radicals)."""


def _ensure_kekulized(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of *mol* with all aromatic bonds resolved to single/double."""
    out = Chem.Mol(mol)
    Chem.Kekulize(out, clearAromaticFlags=True)
    return out


def _bond_order(bond: Chem.Bond) -> int:
    """Bond order as a small integer. Raises if aromatic or fractional."""
    btype = bond.GetBondType()
    if btype == Chem.BondType.SINGLE:
        return 1
    if btype == Chem.BondType.DOUBLE:
        return 2
    if btype == Chem.BondType.TRIPLE:
        return 3
    if btype == Chem.BondType.QUADRUPLE:
        return 4
    if btype == Chem.BondType.ZERO:
        return 0
    if btype == Chem.BondType.AROMATIC:
        raise ValueError("aromatic bond encountered; molecule must be Kekulized first")
    raise ValueError(f"unsupported bond type: {btype}")


def build_be_matrix(mol: Chem.Mol) -> np.ndarray:
    """Build the BE matrix for *mol*.

    The molecule is Kekulized internally and any radicals raise
    :class:`RadicalNotSupportedError` (consistent with v1 scope in the spec).

    Parameters
    ----------
    mol : Chem.Mol
        Atom-mapped, explicit-H RDKit molecule.

    Returns
    -------
    np.ndarray
        Shape (N, N), int dtype.
    """
    if mol is None:
        raise ValueError("mol is None")
    mol = _ensure_kekulized(mol)
    n = mol.GetNumAtoms()
    B = np.zeros((n, n), dtype=int)

    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            raise RadicalNotSupportedError(
                f"atom {atom.GetIdx()} ({atom.GetSymbol()}) has "
                f"{atom.GetNumRadicalElectrons()} unpaired electron(s); "
                "BE-matrix-based fragmentation does not yet support radicals"
            )

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        order = _bond_order(bond)
        B[i, j] = order
        B[j, i] = order

    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        z = atom.GetAtomicNum()
        if z not in VALENCE_ELECTRONS:
            raise ValueError(
                f"no valence-electron table entry for atomic number {z}; "
                f"extend VALENCE_ELECTRONS to add support"
            )
        bond_sum = int(B[i].sum())  # off-diagonal sum (diagonal still 0 here)
        available = VALENCE_ELECTRONS[z] - atom.GetFormalCharge()
        lone_pair_e = available - bond_sum
        if lone_pair_e < 0:
            raise ValueError(
                f"atom {i} ({atom.GetSymbol()}) has more bonds ({bond_sum}) than "
                f"available valence electrons ({available}); check formal charge / valence"
            )
        B[i, i] = lone_pair_e

    return B


def validate_be_matrix(B: np.ndarray, mol: Chem.Mol) -> None:
    """Sanity-check a BE matrix against its source molecule.

    Raises ValueError on any inconsistency. No return value.
    """
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"B must be square 2-D; got shape {B.shape}")
    n = B.shape[0]
    if n != mol.GetNumAtoms():
        raise ValueError(
            f"B has {n} rows but mol has {mol.GetNumAtoms()} atoms"
        )
    if not np.array_equal(B, B.T):
        raise ValueError("B is not symmetric")
    if (B < 0).any():
        raise ValueError("B contains negative entries")

    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        z = atom.GetAtomicNum()
        expected_row_sum = VALENCE_ELECTRONS[z] - atom.GetFormalCharge()
        actual_row_sum = int(B[i].sum())
        if actual_row_sum != expected_row_sum:
            raise ValueError(
                f"row {i} ({atom.GetSymbol()}) sum is {actual_row_sum} but "
                f"valence_electrons - formal_charge = {expected_row_sum}"
            )
        # Aromatic bonds should have been Kekulized away.
        for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.AROMATIC:
                raise ValueError("found AROMATIC bond; molecule must be Kekulized")
