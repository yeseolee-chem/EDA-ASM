"""Diels-Alder: butadiene + ethylene → cyclohexene.

Atom indices:
    0..3  butadiene carbons (C1=C2-C3=C4)
    4, 5  ethylene carbons (C5=C6)
    H atoms appended thereafter.

Reactive bonds (R → P):
    (0, 1): 2 → 1   (C1=C2 reduces order)
    (1, 2): 1 → 2   (C2-C3 becomes C2=C3)
    (2, 3): 2 → 1   (C3=C4 reduces order)
    (4, 5): 2 → 1   (C5=C6 reduces order)
    (3, 4): 0 → 1   (new σ bond)
    (0, 5): 0 → 1   (new σ bond)

Under the strict migrating-atom definition (full-break + full-form), no DA
atom qualifies — all bond changes are mere order shifts at preserved
neighbours. So we expect ``migrating_atoms == []`` and the connected-
component analysis to cleanly split into diene and dienophile.
"""
from __future__ import annotations

from stage0_fragmentation.api import run_fragmentation
from .conftest import build_pair_with_indices


def _build_diels_alder():
    # 0..3 = butadiene carbons; 4, 5 = ethylene carbons
    elements = (
        ["C"] * 6
        + ["H"] * 6  # 2 H on C1, 1 on C2, 1 on C3, 2 on C4
        + ["H"] * 4  # 2 on each ethylene C
    )
    # Butadiene (R): C0=C1-C2=C3, with 2/1/1/2 H respectively
    # Ethylene  (R): C4=C5, with 2/2 H respectively
    bonds_R = [
        (0, 1, 2), (1, 2, 1), (2, 3, 2),
        (4, 5, 2),
        # H labels: 6,7=H on C0; 8=H on C1; 9=H on C2; 10,11=H on C3
        (0, 6, 1), (0, 7, 1), (1, 8, 1), (2, 9, 1), (3, 10, 1), (3, 11, 1),
        # H 12,13 on C4; 14,15 on C5
        (4, 12, 1), (4, 13, 1), (5, 14, 1), (5, 15, 1),
    ]
    # Cyclohexene (P): C0-C1=C2-C3-C4-C5 ring; new σ at C3-C4 and C0-C5
    bonds_P = [
        (0, 1, 1), (1, 2, 2), (2, 3, 1),
        (3, 4, 1), (4, 5, 1), (0, 5, 1),
        (0, 6, 1), (0, 7, 1), (1, 8, 1), (2, 9, 1), (3, 10, 1), (3, 11, 1),
        (4, 12, 1), (4, 13, 1), (5, 14, 1), (5, 15, 1),
    ]
    return build_pair_with_indices(elements, bonds_R, bonds_P)


def test_diels_alder_reactive_bonds():
    mol_R, mol_P = _build_diels_alder()
    result = run_fragmentation(mol_R, mol_P)
    expected = {(0, 1), (1, 2), (2, 3), (4, 5), (3, 4), (0, 5)}
    assert set(result.reactive_bonds) == expected


def test_diels_alder_no_migration():
    mol_R, mol_P = _build_diels_alder()
    result = run_fragmentation(mol_R, mol_P)
    # Strict definition: all DA atoms keep their original neighbours, just
    # change bond orders, so none should be flagged migrating.
    assert result.migrating_atoms == []


def test_diels_alder_partition():
    mol_R, mol_P = _build_diels_alder()
    result = run_fragmentation(mol_R, mol_P)
    assert result.is_pure_rearrangement is False
    assert len(result.fragments) == 2

    # Diene atoms (0..3 + their H's) and dienophile atoms (4, 5 + theirs)
    # should land in different fragments.
    diene_carbons = {0, 1, 2, 3}
    dienophile_carbons = {4, 5}
    frags = result.fragments
    f_diene = next(f for f in frags if diene_carbons & f)
    f_dieno = next(f for f in frags if dienophile_carbons & f)
    assert f_diene != f_dieno
    assert diene_carbons.issubset(f_diene)
    assert dienophile_carbons.issubset(f_dieno)
    # H atoms follow their parent C (we only check a few key ones).
    assert 6 in f_diene  # H on C0
    assert 12 in f_dieno  # H on C4
