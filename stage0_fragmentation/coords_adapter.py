"""Connectivity-only fragmentation driver for the Halo8 use case.

Stage 0's full pipeline (`api.run_fragmentation`) requires atom-mapped RDKit
``Mol`` objects with valence-consistent bond orders. Halo8 trajectories give
us 3-D coordinates and atomic numbers; building a strictly valence-correct
``Mol`` from that requires bond-order perception that often fails on
TS-like or radical-like geometries.

For ASM-EDA fragment partitioning, however, **connectivity-level information
is sufficient**: which atoms are bonded changes between R and P, and that
delta determines which atoms migrate and where the cut should land.

This module exposes :func:`fragments_from_coords` which:
1. Builds binary connectivity matrices ``B_R`` and ``B_P`` from
   distance-based bond detection (Cordero radii, shared with the rest of
   the pipeline).
2. Reuses the same migration / partitioning / routing logic from
   ``api.run_fragmentation`` but operates directly on the connectivity
   matrices.

The resulting ``FragmentationResult`` is fully compatible with the rest of
the project (Phase 1.5 review tool, downstream Stage 5).
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np

from eda_asm.phase1.bonds import detect_bonds

from .migration import detect_migrating_atoms, reactive_bonds_from_delta
from .partition import connected_component_analysis
from .rearrangement import migration_clustering, split_by_user_hint
from .types import FragmentationResult


def _connectivity_matrix(numbers: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    bonds = sorted(detect_bonds(numbers, coords))
    n = len(numbers)
    B = np.zeros((n, n), dtype=int)
    for i, j in bonds:
        B[i, j] = 1
        B[j, i] = 1
    return B, bonds


def fragments_from_coords(
    numbers: list[int] | np.ndarray,
    coords_R: list[list[float]] | np.ndarray,
    coords_P: list[list[float]] | np.ndarray,
    user_hint: dict | None = None,
) -> FragmentationResult:
    """Run the Stage-0 fragmentation algorithm directly on coordinates.

    Returns a :class:`FragmentationResult` compatible with the RDKit-based
    :func:`run_fragmentation`. Bond perception uses Cordero radii; bond
    orders are not inferred (every bond is treated as a single edge).
    """
    numbers_np = np.asarray(numbers, dtype=int)
    coords_R_np = np.asarray(coords_R, dtype=float)
    coords_P_np = np.asarray(coords_P, dtype=float)

    if coords_R_np.shape != coords_P_np.shape:
        raise ValueError(
            f"coords_R shape {coords_R_np.shape} != coords_P shape {coords_P_np.shape}"
        )
    n = len(numbers_np)
    if coords_R_np.shape[0] != n:
        raise ValueError("number of atoms doesn't match coords")

    B_R, bonds_R = _connectivity_matrix(numbers_np, coords_R_np)
    B_P, _bonds_P = _connectivity_matrix(numbers_np, coords_P_np)

    delta = B_P - B_R
    reactive_bonds = reactive_bonds_from_delta(delta)
    migrating = detect_migrating_atoms(B_R, B_P)

    components = connected_component_analysis(n, bonds_R, reactive_bonds, migrating)

    notes: list[str] = []
    is_pure_rearrangement = len(components) < 2
    fallback: str | None = None

    if not is_pure_rearrangement:
        seeds, ranked, used = _seed_selection(
            components, B_R, B_P, reactive_bonds, migrating
        )
        fragments, route_notes = _route_migrations_with_scores(
            seeds, migrating, B_P - B_R
        )
        notes.extend(route_notes)
        fragments = _merge_strays(fragments, ranked, bonds_R, reactive_bonds, notes)
    else:
        notes.append("no clean 2-component split; entering rearrangement fallback")
        fragments_opt: list[set[int]] | None = None
        if user_hint and "split_bond" in user_hint:
            split = split_by_user_hint(n, bonds_R, tuple(user_hint["split_bond"]))
            if split is not None:
                fragments_opt = split
                fallback = "user_hint"
                notes.append(f"used user_hint split_bond={user_hint['split_bond']}")
        if fragments_opt is None:
            split = migration_clustering(n, bonds_R, reactive_bonds, migrating)
            if split is not None:
                fragments_opt = split
                fallback = "migration_clustering"
                notes.append("used migration_clustering")
        if fragments_opt is None:
            fragments_opt = [set(range(n))]
            fallback = "strain_only"
            notes.append("no fallback succeeded; flagging strain_only")
        fragments = fragments_opt

    cap_sites = _cap_sites_from_bonds(fragments, bonds_R)

    return FragmentationResult(
        fragments=fragments,
        migrating_atoms=migrating,
        reactive_bonds=reactive_bonds,
        cap_sites=cap_sites,
        is_pure_rearrangement=is_pure_rearrangement,
        fallback_strategy=fallback,
        notes=notes,
    )


def _seed_selection(
    components: list[set[int]],
    B_R: np.ndarray,
    B_P: np.ndarray,
    reactive_bonds: list[tuple[int, int]],
    migrating: list[dict],
) -> tuple[list[set[int]], list[set[int]], set[int]]:
    """Bipartite-coloring-based seed selection (mirrors api.run_fragmentation)."""
    key_bonds = [
        (i, j)
        for (i, j) in reactive_bonds
        if (B_R[i, j] == 0 and B_P[i, j] > 0) or (B_R[i, j] > 0 and B_P[i, j] == 0)
    ]
    comp_of: dict[int, int] = {}
    for idx, comp in enumerate(components):
        for atom in comp:
            comp_of[atom] = idx

    cg = nx.Graph()
    cg.add_nodes_from(range(len(components)))
    for i, j in key_bonds:
        ci, cj = comp_of.get(i), comp_of.get(j)
        if ci is not None and cj is not None and ci != cj:
            cg.add_edge(ci, cj)
    for m in migrating:
        for f in m["from"]:
            for t in m["to"]:
                cf, ct = comp_of.get(f), comp_of.get(t)
                if cf is not None and ct is not None and cf != ct:
                    cg.add_edge(cf, ct)

    seed_a: set[int] = set()
    seed_b: set[int] = set()
    used: set[int] = set()
    for cc in nx.connected_components(cg):
        sub = cg.subgraph(cc)
        if sub.number_of_edges() == 0 or not nx.is_bipartite(sub):
            continue
        coloring = nx.bipartite.color(sub)
        for ci, color in coloring.items():
            (seed_a if color == 0 else seed_b).update(components[ci])
            used.add(ci)

    if seed_a and seed_b:
        seeds = [seed_a, seed_b]
        ranked = [components[i] for i in range(len(components)) if i not in used]
        return seeds, ranked, used

    # Fallback: top-2 by endpoint touch + size
    endpoint_atoms: set[int] = set()
    for m in migrating:
        endpoint_atoms.update(m["from"])
        endpoint_atoms.update(m["to"])
    for i, j in reactive_bonds:
        endpoint_atoms.add(i)
        endpoint_atoms.add(j)

    def _score(comp: set[int]) -> tuple[int, int, int]:
        touch = sum(1 for a in comp if a in endpoint_atoms)
        return (touch, len(comp), -min(comp) if comp else 0)

    ranked_full = sorted(components, key=_score, reverse=True)
    seeds = [ranked_full[0], ranked_full[1] if len(ranked_full) > 1 else set()]
    return seeds, ranked_full[2:], set()


def _route_migrations_with_scores(
    components: list[set[int]],
    migrating_atoms: list[dict],
    delta_be: np.ndarray,
) -> tuple[list[set[int]], list[str]]:
    """Inline copy of partition.route_migrations to avoid double-import games."""
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
        scores: dict[int, int] = {}
        for j in m["to"]:
            comp_idx = find_component(j)
            if comp_idx is None:
                continue
            gain = int(delta_be[k, j])
            scores[comp_idx] = scores.get(comp_idx, 0) + gain
        if not scores:
            notes.append(
                f"migrating atom {k}: no destination found in any component"
            )
            continue
        best = sorted(
            scores.items(),
            key=lambda kv: (-kv[1], min(new[kv[0]]) if new[kv[0]] else 1 << 30),
        )[0][0]
        new[best].add(k)
    return new, notes


def _merge_strays(
    fragments: list[set[int]],
    ranked: list[set[int]],
    bonds_R: list[tuple[int, int]],
    reactive_bonds: Iterable[tuple[int, int]],
    notes: list[str],
) -> list[set[int]]:
    """Merge any atom not in a seed fragment using R-graph neighbour majority."""
    assigned: set[int] = set().union(*fragments) if fragments else set()
    strays = [a for comp in ranked for a in comp if a not in assigned]

    reactive_set = {tuple(sorted(b)) for b in reactive_bonds}
    n_atoms = max(
        max((max(b) for b in bonds_R), default=-1),
        max(assigned, default=-1),
        max(strays, default=-1),
    ) + 1
    neighbors: dict[int, set[int]] = {a: set() for a in range(n_atoms)}
    for i, j in bonds_R:
        if tuple(sorted((i, j))) in reactive_set:
            continue
        neighbors[i].add(j)
        neighbors[j].add(i)

    progress = True
    while strays and progress:
        progress = False
        remaining: list[int] = []
        for atom in strays:
            votes: dict[int, int] = {}
            for nb in neighbors[atom]:
                for idx, frag in enumerate(fragments):
                    if nb in frag:
                        votes[idx] = votes.get(idx, 0) + 1
            if votes:
                ranked_idx = sorted(
                    votes.keys(),
                    key=lambda k: (votes[k], -len(fragments[k])),
                    reverse=True,
                )
                chosen = int(ranked_idx[0])
                fragments[chosen].add(atom)
                assigned.add(atom)
                progress = True
                notes.append(
                    f"merged stray atom {atom} into fragment {chosen} "
                    f"by R-neighbor majority"
                )
            else:
                remaining.append(atom)
        strays = remaining

    for atom in strays:
        sizes = [(len(frag), idx) for idx, frag in enumerate(fragments)]
        sizes.sort()
        chosen = int(sizes[0][1])
        fragments[chosen].add(atom)
        assigned.add(atom)
        notes.append(
            f"merged isolated stray atom {atom} into smallest fragment {chosen}"
        )
    return fragments


def _cap_sites_from_bonds(
    fragments: list[set[int]],
    bonds_R: list[tuple[int, int]],
) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(fragments))}

    def fragment_of(atom: int) -> int | None:
        for idx, frag in enumerate(fragments):
            if atom in frag:
                return idx
        return None

    for i, j in bonds_R:
        fi = fragment_of(i)
        fj = fragment_of(j)
        if fi is None or fj is None or fi == fj:
            continue
        out[fi].append((i, j))
        out[fj].append((j, i))
    return out
