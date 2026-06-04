"""SN2: Cl- + CH3-Br → Cl-CH3 + Br-.

Reactive bonds: (Cl-C) formed, (C-Br) broken.

The central carbon has *one fully broken* bond (C-Br) and *one fully formed*
bond (C-Cl), so it qualifies as a migrating atom under the strict definition
(see migration.py docstring). After removing it, the H atoms are isolated;
routing returns C to the {Cl} component, and the strays-merge step pulls the
Hs together. Br ends up alone in the second fragment.
"""
from __future__ import annotations

from stage0_fragmentation.api import run_fragmentation
from .conftest import build_pair_with_indices


def _build_sn2():
    # indices: 0=Cl, 1=C, 2=H, 3=H, 4=H, 5=Br
    elements = ["Cl", "C", "H", "H", "H", "Br"]
    bonds_R = [(1, 2, 1), (1, 3, 1), (1, 4, 1), (1, 5, 1)]
    bonds_P = [(0, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1)]
    return build_pair_with_indices(
        elements,
        bonds_R,
        bonds_P,
        formal_charges_R={0: -1},
        formal_charges_P={5: -1},
    )


def test_sn2_reactive_bonds():
    mol_R, mol_P = _build_sn2()
    result = run_fragmentation(mol_R, mol_P)
    assert (0, 1) in result.reactive_bonds
    assert (1, 5) in result.reactive_bonds


def test_sn2_carbon_is_migrating():
    mol_R, mol_P = _build_sn2()
    result = run_fragmentation(mol_R, mol_P)
    migrating_atoms = {m["atom"] for m in result.migrating_atoms}
    assert 1 in migrating_atoms  # central carbon
    # Cl and Br should NOT be migrating (each only loses or only gains, not both)
    assert 0 not in migrating_atoms
    assert 5 not in migrating_atoms


def test_sn2_clean_partition():
    mol_R, mol_P = _build_sn2()
    result = run_fragmentation(mol_R, mol_P)
    # Two fragments, every atom assigned exactly once.
    assert len(result.fragments) == 2
    union = set().union(*result.fragments)
    assert union == {0, 1, 2, 3, 4, 5}
    # Br should be alone in its fragment (it has no other connections in R).
    br_frag = next(f for f in result.fragments if 5 in f)
    assert br_frag == {5}


def test_sn2_not_pure_rearrangement():
    mol_R, mol_P = _build_sn2()
    result = run_fragmentation(mol_R, mol_P)
    assert result.is_pure_rearrangement is False
    assert result.fallback_strategy is None
