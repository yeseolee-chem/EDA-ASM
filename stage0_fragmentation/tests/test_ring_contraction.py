"""1,3-H shift (keto–enol tautomerization of acetaldehyde).

This is the cleanest small-molecule case where exactly one hydrogen migrates
between two heavy atoms without other connectivity changes. It exercises the
*migrating atom* path of the Stage-0 algorithm, which is the same machinery
the spec calls out for ring contraction with hydrogen migration.

Atom indexing (16 atoms total counting all explicit H's):
    0 = C (methyl)        in R: H3, H4, H5 attached
    1 = C (carbonyl)      in R: H6 attached, double bond to O2
    2 = O                  in R: bonded to C1 (double)
    3, 4, 5 = H on C0
    6 = H on C1

Reactive bonds (R → P):
    (0, 1): 1 → 2   (C–C becomes C=C)
    (1, 2): 2 → 1   (C=O becomes C–O)
    (0, 5): 1 → 0   (C–H broken — fully)
    (2, 5): 0 → 1   (O–H formed — fully)

Atom 5 has a fully-broken loss (C–H) and a fully-formed gain (O–H), so it is
the migrating atom under the strict definition.
"""
from __future__ import annotations

from stage0_fragmentation.api import run_fragmentation
from .conftest import build_pair_with_indices


def _build_keto_enol():
    elements = ["C", "C", "O", "H", "H", "H", "H"]
    bonds_R = [
        (0, 1, 1), (1, 2, 2),
        (0, 3, 1), (0, 4, 1), (0, 5, 1),
        (1, 6, 1),
    ]
    bonds_P = [
        (0, 1, 2), (1, 2, 1),
        (2, 5, 1),                       # H5 migrated to O2
        (0, 3, 1), (0, 4, 1),
        (1, 6, 1),
    ]
    return build_pair_with_indices(elements, bonds_R, bonds_P)


def test_keto_enol_reactive_bonds():
    mol_R, mol_P = _build_keto_enol()
    result = run_fragmentation(mol_R, mol_P)
    rb = set(result.reactive_bonds)
    expected = {(0, 1), (1, 2), (0, 5), (2, 5)}
    assert expected.issubset(rb)


def test_keto_enol_h5_is_migrating():
    mol_R, mol_P = _build_keto_enol()
    result = run_fragmentation(mol_R, mol_P)
    migrating_atoms = {m["atom"] for m in result.migrating_atoms}
    assert 5 in migrating_atoms
    record = next(m for m in result.migrating_atoms if m["atom"] == 5)
    assert 0 in record["from"]
    assert 2 in record["to"]


def test_keto_enol_partition_covers_atoms():
    mol_R, mol_P = _build_keto_enol()
    result = run_fragmentation(mol_R, mol_P)
    union = set().union(*result.fragments)
    assert union == set(range(7))
    assert sum(len(f) for f in result.fragments) == 7
    # The migrating H should land in whichever fragment contains its destination O.
    migrating_h = 5
    for frag in result.fragments:
        if 2 in frag:  # the O atom
            assert migrating_h in frag
            break
    else:
        raise AssertionError("oxygen not in any fragment")


def test_keto_enol_not_pure_rearrangement():
    mol_R, mol_P = _build_keto_enol()
    result = run_fragmentation(mol_R, mol_P)
    assert result.is_pure_rearrangement is False
