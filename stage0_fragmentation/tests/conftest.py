"""Shared fixtures and helpers for the Stage-0 fragmentation tests."""
from __future__ import annotations

from rdkit import Chem


def smiles_to_mapped_mol(smiles: str) -> Chem.Mol:
    """Parse a SMILES string, add explicit Hs, and assign a stable atom map.

    The map number for each atom is set to ``atom.GetIdx() + 1`` so that the
    map ID encodes the canonical RDKit atom index. Tests can then reference
    atoms by index without worrying about RDKit's parsing order.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol


def build_pair_with_indices(
    elements: list[str],
    bonds_R: list[tuple[int, int, int]],
    bonds_P: list[tuple[int, int, int]],
    formal_charges_R: dict[int, int] | None = None,
    formal_charges_P: dict[int, int] | None = None,
) -> tuple[Chem.Mol, Chem.Mol]:
    """Construct an R/P pair from atom indices and bond lists.

    Parameters
    ----------
    elements : list[str]
        Element symbols for each atom (in index order).
    bonds_R, bonds_P : list[(i, j, order)]
        Bond lists for R and P. Order is 1, 2, or 3.
    formal_charges_R, formal_charges_P : dict[int, int], optional
        Map atom index → formal charge. Atoms not listed default to 0.

    Returns
    -------
    (mol_R, mol_P) : tuple[Chem.Mol, Chem.Mol]
        Two RDKit molecules with identical atom ordering, suitable for
        passing to ``run_fragmentation``.
    """
    bond_order_to_type = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    def build(
        bonds: list[tuple[int, int, int]],
        charges: dict[int, int] | None,
    ) -> Chem.Mol:
        rw = Chem.RWMol()
        for i, sym in enumerate(elements):
            atom = Chem.Atom(sym)
            atom.SetNoImplicit(True)
            if charges and i in charges:
                atom.SetFormalCharge(charges[i])
            rw.AddAtom(atom)
        for i, j, order in bonds:
            rw.AddBond(i, j, bond_order_to_type[order])
        mol = rw.GetMol()
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        return mol

    return (
        build(bonds_R, formal_charges_R),
        build(bonds_P, formal_charges_P),
    )
