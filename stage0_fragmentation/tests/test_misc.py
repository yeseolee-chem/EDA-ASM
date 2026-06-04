"""Additional unit tests to lift coverage on the core functions."""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from stage0_fragmentation.be_matrix import build_be_matrix, validate_be_matrix
from stage0_fragmentation.capping import find_cap_sites, fragment_smiles
from stage0_fragmentation.migration import (
    detect_migrating_atoms,
    reactive_bonds_from_delta,
)
from stage0_fragmentation.partition import (
    bonds_from_be,
    connected_component_analysis,
    route_migrations,
)
from stage0_fragmentation.rearrangement import (
    migration_clustering,
    split_by_user_hint,
)
from stage0_fragmentation.types import FragmentationResult


def test_build_be_matrix_rejects_none():
    with pytest.raises(ValueError, match="None"):
        build_be_matrix(None)  # type: ignore[arg-type]


def test_validate_be_matrix_rejects_wrong_shape():
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    B = np.zeros((3, 3), dtype=int)
    with pytest.raises(ValueError):
        validate_be_matrix(B, mol)


def test_validate_be_matrix_rejects_negative():
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    B = build_be_matrix(mol)
    B[0, 1] = -1
    B[1, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        validate_be_matrix(B, mol)


def test_detect_migrating_atoms_shape_mismatch():
    with pytest.raises(ValueError):
        detect_migrating_atoms(np.zeros((3, 3)), np.zeros((4, 4)))


def test_reactive_bonds_from_delta_shape_check():
    with pytest.raises(ValueError):
        reactive_bonds_from_delta(np.zeros((3, 4)))


def test_route_migrations_handles_empty_components():
    assert route_migrations([], [], np.zeros((1, 1))) == ([], [])


def test_route_migrations_no_destination_in_components():
    """If a migrating atom's destinations aren't in any seed component, the
    routing logs a note and leaves the atom unassigned."""
    components = [{0, 1}]  # only one fragment as seed
    migrating = [{"atom": 5, "from": [0], "to": [9], "loss": 1, "gain": 1}]
    delta = np.zeros((10, 10), dtype=int)
    delta[5, 9] = 1
    new_components, notes = route_migrations(components, migrating, delta)
    assert new_components == [{0, 1}]  # untouched
    assert any("no destination found" in n for n in notes)


def test_validate_fragmentation_rejects_size_mismatch():
    from stage0_fragmentation.partition import validate_fragmentation

    mol_R = Chem.AddHs(Chem.MolFromSmiles("C"))
    mol_P = Chem.AddHs(Chem.MolFromSmiles("CC"))
    result = FragmentationResult(fragments=[set(range(mol_R.GetNumAtoms()))])
    with pytest.raises(ValueError, match="atom counts"):
        validate_fragmentation(result, mol_R, mol_P)


def test_validate_fragmentation_rejects_overlap():
    from stage0_fragmentation.partition import validate_fragmentation

    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    n = mol.GetNumAtoms()
    result = FragmentationResult(fragments=[set(range(n)), {0}])  # overlap on atom 0
    with pytest.raises(ValueError, match="overlap"):
        validate_fragmentation(result, mol, mol)


def test_validate_fragmentation_rejects_missing_atom():
    from stage0_fragmentation.partition import validate_fragmentation

    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    result = FragmentationResult(fragments=[{0, 1}])  # most atoms missing
    with pytest.raises(ValueError, match="missing"):
        validate_fragmentation(result, mol, mol)


def test_connected_component_analysis_no_bonds():
    comps = connected_component_analysis(3, [], [], [])
    assert {len(c) for c in comps} == {1}


def test_bonds_from_be_extracts_uppertriangle():
    B = np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]])
    assert bonds_from_be(B) == [(0, 1), (1, 2)]


def test_split_by_user_hint_basic():
    bonds_R = [(0, 1), (1, 2), (2, 3)]
    out = split_by_user_hint(4, bonds_R, (1, 2))
    assert out is not None
    assert {tuple(sorted(c)) for c in out} == {(0, 1), (2, 3)}


def test_split_by_user_hint_returns_none_when_no_split():
    # Removing this bond doesn't disconnect (cycle).
    bonds_R = [(0, 1), (1, 2), (2, 0)]
    out = split_by_user_hint(3, bonds_R, (0, 1))
    assert out is None


def test_migration_clustering_empty():
    assert migration_clustering(5, [], [], []) is None


def test_migration_clustering_two_clusters():
    migrating = [
        {"atom": 0, "from": [3], "to": [4]},
        {"atom": 1, "from": [3], "to": [4]},  # shares partners with #0
        {"atom": 5, "from": [6], "to": [7]},  # disjoint
    ]
    out = migration_clustering(8, [], [], migrating)
    assert out is not None
    moving, skeleton = out
    assert {0, 1}.issubset(moving)


def test_fragment_smiles_methane():
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    smi = fragment_smiles(set(range(mol.GetNumAtoms())), mol, [])
    # Explicit-H molecule yields "[H]C([H])([H])[H]"; collapse to canonical.
    assert smi is not None
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    assert canon == "C"


def test_find_cap_sites_returns_empty_dict_for_one_fragment():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    out = find_cap_sites([set(range(mol.GetNumAtoms()))], mol)
    assert out == {0: []}


def test_find_cap_sites_records_cross_bonds():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    # Split C-C: fragments {C0 + its H's} and {C1 + its H's}.
    c_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"]
    h_for = {c: [b.GetOtherAtomIdx(c) for b in mol.GetAtomWithIdx(c).GetBonds() if b.GetOtherAtom(mol.GetAtomWithIdx(c)).GetSymbol() == "H"] for c in c_idxs}
    frag_a = {c_idxs[0]} | set(h_for[c_idxs[0]])
    frag_b = {c_idxs[1]} | set(h_for[c_idxs[1]])
    sites = find_cap_sites([frag_a, frag_b], mol)
    # Each fragment should record one cap site (the C-C bond once).
    assert len(sites[0]) == 1
    assert len(sites[1]) == 1


def test_fragmentation_result_defaults():
    r = FragmentationResult(fragments=[{0, 1}, {2}])
    assert r.migrating_atoms == []
    assert r.reactive_bonds == []
    assert r.cap_sites == {}
    assert r.fallback_strategy is None
    assert r.notes == []
