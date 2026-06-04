"""Connected-component partitioning + migration routing for the BE-matrix workflow."""
from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
from rdkit import Chem

from .types import FragmentationResult


def _skeleton_graph(
    n_atoms: int,
    bonds_R: Iterable[tuple[int, int]],
    reactive_bonds: Iterable[tuple[int, int]],
    excluded_atoms: Iterable[int] = (),
) -> nx.Graph:
    """Build the R-graph minus reactive bonds and minus excluded atoms."""
    g = nx.Graph()
    g.add_nodes_from(range(n_atoms))
    react_set = {tuple(sorted(b)) for b in reactive_bonds}
    for i, j in bonds_R:
        key = tuple(sorted((i, j)))
        if key in react_set:
            continue
        g.add_edge(i, j)
    for k in excluded_atoms:
        if g.has_node(k):
            g.remove_node(k)
    return g


def bonds_from_be(B: np.ndarray) -> list[tuple[int, int]]:
    """Return all (i, j) (i < j) where B[i, j] > 0."""
    n = B.shape[0]
    out: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if B[i, j] > 0:
                out.append((i, j))
    return out


def connected_component_analysis(
    n_atoms: int,
    bonds_R: list[tuple[int, int]],
    reactive_bonds: list[tuple[int, int]],
    migrating_atoms: list[dict],
) -> list[set[int]]:
    """Split the R skeleton (sans reactive bonds and migrating atoms) into
    connected components, sorted largest-first."""
    excluded = [m["atom"] for m in migrating_atoms]
    g = _skeleton_graph(n_atoms, bonds_R, reactive_bonds, excluded_atoms=excluded)
    comps = [set(c) for c in nx.connected_components(g)]
    comps.sort(key=lambda c: (-len(c), min(c) if c else 0))
    return comps


def route_migrations(
    components: list[set[int]],
    migrating_atoms: list[dict],
    delta_be: np.ndarray,
) -> tuple[list[set[int]], list[str]]:
    """Assign each migrating atom to whichever component contains its dominant
    destination. Returns (new_components, notes)."""
    notes: list[str] = []
    if not components:
        return components, notes
    new = [set(c) for c in components]

    def find_component(atom: int) -> int | None:
        for idx, comp in enumerate(new):
            if atom in comp:
                return idx
        return None

    for m in migrating_atoms:
        k = m["atom"]
        # Score each candidate destination by the bond-order it gains.
        scores: dict[int, int] = {}
        for j in m["to"]:
            comp_idx = find_component(j)
            if comp_idx is None:
                continue
            gain = int(delta_be[k, j])
            scores[comp_idx] = scores.get(comp_idx, 0) + gain
        if not scores:
            notes.append(
                f"migrating atom {k}: no destination found in any component; "
                "leaving unassigned"
            )
            continue
        # Tie-break: highest gain wins; on tie, lowest atom index in component.
        best = sorted(
            scores.items(),
            key=lambda kv: (-kv[1], min(new[kv[0]]) if new[kv[0]] else 1 << 30),
        )[0][0]
        new[best].add(k)

    return new, notes


def validate_fragmentation(
    result: FragmentationResult,
    mol_R: Chem.Mol,
    mol_P: Chem.Mol,
) -> None:
    """Cross-check a FragmentationResult against R and P.

    Raises ValueError on any inconsistency.
    """
    if mol_R.GetNumAtoms() != mol_P.GetNumAtoms():
        raise ValueError("mol_R and mol_P have different atom counts")
    n = mol_R.GetNumAtoms()
    union: set[int] = set()
    for frag in result.fragments:
        if union & frag:
            raise ValueError(f"fragments overlap on atoms {sorted(union & frag)}")
        union |= frag
    expected = set(range(n))
    if union != expected:
        missing = sorted(expected - union)
        extra = sorted(union - expected)
        if missing:
            raise ValueError(f"missing atoms: {missing}")
        if extra:
            raise ValueError(f"out-of-range atom indices: {extra}")
    # Migrating atoms must end up assigned to a fragment.
    for m in result.migrating_atoms:
        if not any(m["atom"] in frag for frag in result.fragments):
            raise ValueError(
                f"migrating atom {m['atom']} not assigned to any fragment"
            )
    # Cap sites should reference (anchor in fragment, partner in OTHER fragment).
    for frag_idx, sites in result.cap_sites.items():
        if frag_idx < 0 or frag_idx >= len(result.fragments):
            raise ValueError(f"cap_sites refers to unknown fragment index {frag_idx}")
        frag = result.fragments[frag_idx]
        for anchor, partner in sites:
            if anchor not in frag:
                raise ValueError(
                    f"cap site anchor {anchor} not in fragment {frag_idx}"
                )
            if partner in frag:
                raise ValueError(
                    f"cap site partner {partner} is in the same fragment {frag_idx}"
                )
