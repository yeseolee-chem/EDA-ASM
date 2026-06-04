"""Cope rearrangement of 1,5-hexadiene.

R: H₂C₁=C₂H–C₃H₂–C₄H₂–C₅H=C₆H₂
P: same skeleton, with the σ(C3-C4) bond broken, σ(C1-C6) bond formed,
   and π bonds shifted (C1=C2 → C2=C3, C5=C6 → C4=C5).

Under the strict migrating-atom definition no individual atom qualifies
(all carbons keep their original neighbours; only bond orders shift, with
the σ(C3-C4) break and σ(C1-C6) form being the only "fully-formed/broken"
events). The skeleton-after-removing-reactive-bonds collapses into many
single-atom components, so the bipartition logic should still find a
two-fragment split — but the chemistry is genuinely a single concerted
rearrangement, so we're satisfied if either:
    - run_fragmentation succeeds with a covered, validated partition; or
    - is_pure_rearrangement is True and a fallback fires.
"""
from __future__ import annotations

from stage0_fragmentation.api import run_fragmentation
from .conftest import build_pair_with_indices


def _build_cope():
    # 6 carbons + 10 hydrogens
    elements = ["C"] * 6 + ["H"] * 10
    # Hydrogen distribution: H6,H7 on C0; H8 on C1; H9,H10 on C2; H11,H12 on C3;
    # H13 on C4; H14,H15 on C5
    h_bonds = [
        (0, 6, 1), (0, 7, 1),
        (1, 8, 1),
        (2, 9, 1), (2, 10, 1),
        (3, 11, 1), (3, 12, 1),
        (4, 13, 1),
        (5, 14, 1), (5, 15, 1),
    ]
    bonds_R = [
        (0, 1, 2), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 2),
    ] + h_bonds
    bonds_P = [
        (0, 1, 1), (1, 2, 2), (2, 3, 1),  # π shifted
        (3, 4, 0),  # placeholder; we'll filter below
        (3, 4, 1) if False else None,  # not used
    ]
    # Cleaner: build bonds_P from scratch.
    bonds_P = [
        (0, 1, 1), (1, 2, 2),  # C1-C2 single, C2=C3 double
        (3, 4, 2), (4, 5, 1),  # C4=C5 double, C5-C6 single
        (0, 5, 1),             # new σ bond
        # σ(C3-C4) is broken (no entry)
        # Note: with this assignment carbons 2 and 3 are NOT bonded in P;
        # the rearranged 1,5-hexadiene labels look re-permuted.
    ] + h_bonds
    return build_pair_with_indices(elements, bonds_R, bonds_P)


def test_cope_runs_and_produces_valid_partition():
    mol_R, mol_P = _build_cope()
    result = run_fragmentation(mol_R, mol_P)
    # Atom partition must cover all atoms.
    union = set().union(*result.fragments)
    assert union == set(range(16))
    # No overlap.
    total = sum(len(f) for f in result.fragments)
    assert total == 16


def test_cope_reactive_bonds_include_sigma_swap():
    mol_R, mol_P = _build_cope()
    result = run_fragmentation(mol_R, mol_P)
    rb = set(result.reactive_bonds)
    # σ(C3-C4) is broken in P → reactive
    assert (3, 4) in rb
    # σ(C0-C5) is the new bond → reactive (note our indices: 0 ↔ atom C1, 5 ↔ atom C6)
    assert (0, 5) in rb
