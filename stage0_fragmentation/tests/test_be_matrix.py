"""BE matrix unit tests."""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from stage0_fragmentation.be_matrix import (
    RadicalNotSupportedError,
    build_be_matrix,
    validate_be_matrix,
)
from .conftest import smiles_to_mapped_mol


def test_methane_be_matrix():
    mol = smiles_to_mapped_mol("C")
    B = build_be_matrix(mol)
    assert B.shape == (5, 5)
    # Symmetry
    assert np.array_equal(B, B.T)
    # C-H bonds = 4 single bonds
    c_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C")
    h_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H"]
    for h in h_idxs:
        assert B[c_idx, h] == 1
    assert B[c_idx, c_idx] == 0  # C has no lone pair electrons in CH4
    for h in h_idxs:
        assert B[h, h] == 0


def test_water_be_matrix_lone_pairs():
    mol = smiles_to_mapped_mol("O")
    B = build_be_matrix(mol)
    o_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
    assert B[o_idx, o_idx] == 4  # 2 lone pairs = 4 electrons
    validate_be_matrix(B, mol)


def test_ammonia_be_matrix_lone_pairs():
    mol = smiles_to_mapped_mol("N")
    B = build_be_matrix(mol)
    n_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    assert B[n_idx, n_idx] == 2  # 1 lone pair = 2 electrons
    validate_be_matrix(B, mol)


def test_aromatic_is_kekulized():
    """Benzene rows should sum to 4 per C (3 bonds + 1 H)."""
    mol = smiles_to_mapped_mol("c1ccccc1")
    B = build_be_matrix(mol)
    validate_be_matrix(B, mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C":
            assert B[atom.GetIdx()].sum() == 4


def test_charged_chloride_anion():
    mol = smiles_to_mapped_mol("[Cl-]")
    B = build_be_matrix(mol)
    cl_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl")
    # Cl- has 8 lone pair electrons (4 lone pairs); row sum = 8
    assert B[cl_idx, cl_idx] == 8
    assert B[cl_idx].sum() == 8
    validate_be_matrix(B, mol)


def test_radical_rejected():
    """Methyl radical should raise."""
    mol = Chem.MolFromSmiles("[CH3]")
    mol = Chem.AddHs(mol)
    with pytest.raises(RadicalNotSupportedError):
        build_be_matrix(mol)


def test_validation_catches_asymmetry():
    mol = smiles_to_mapped_mol("C")
    B = build_be_matrix(mol)
    B[0, 1] += 1  # break symmetry
    with pytest.raises(ValueError, match="symmetric"):
        validate_be_matrix(B, mol)


def test_acetylene_triple_bond():
    mol = smiles_to_mapped_mol("C#C")
    B = build_be_matrix(mol)
    c_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"]
    assert B[c_idxs[0], c_idxs[1]] == 3
    validate_be_matrix(B, mol)


def test_validate_be_matrix_catches_row_sum_mismatch():
    mol = smiles_to_mapped_mol("C")
    B = build_be_matrix(mol)
    # Add a phantom electron to the C diagonal — row sum no longer = valence_e.
    B[0, 0] += 1
    with pytest.raises(ValueError, match="row .* sum"):
        validate_be_matrix(B, mol)


def test_unsupported_atom_raises():
    """An element absent from VALENCE_ELECTRONS should raise."""
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom("Po"))  # Polonium — not in the table
    mol = rw.GetMol()
    with pytest.raises(ValueError, match="valence-electron"):
        build_be_matrix(mol)
