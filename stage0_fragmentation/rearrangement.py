"""Fallback strategies for "pure rearrangement" cases.

Triggered when removing reactive bonds + migrating atoms still leaves the
molecular graph connected (i.e. no clean two-component split exists).
"""
from __future__ import annotations

import networkx as nx

from .partition import _skeleton_graph


def migration_clustering(
    n_atoms: int,
    bonds_R: list[tuple[int, int]],
    reactive_bonds: list[tuple[int, int]],
    migrating_atoms: list[dict],
) -> list[set[int]] | None:
    """Group migrating atoms into "moving units" by their migration graph.

    Two migrating atoms are linked if they share at least one ``from`` or
    ``to`` partner. The remaining atoms (the skeleton) form the other
    fragment. Returns ``[migrating_cluster, skeleton]`` or ``None`` if the
    migration graph itself splits into ≠ 1 cluster (handled elsewhere).
    """
    if not migrating_atoms:
        return None

    g = nx.Graph()
    for m in migrating_atoms:
        g.add_node(m["atom"])
    for i, mi in enumerate(migrating_atoms):
        for j in range(i + 1, len(migrating_atoms)):
            mj = migrating_atoms[j]
            shared = set(mi["from"] + mi["to"]) & set(mj["from"] + mj["to"])
            if shared:
                g.add_edge(mi["atom"], mj["atom"])

    clusters = [set(c) for c in nx.connected_components(g)]
    if not clusters:
        return None

    # Take the biggest cluster as the "moving unit".
    clusters.sort(key=lambda c: -len(c))
    moving = clusters[0]
    skeleton = set(range(n_atoms)) - moving
    if not moving or not skeleton:
        return None
    return [moving, skeleton]


def split_by_user_hint(
    n_atoms: int,
    bonds_R: list[tuple[int, int]],
    split_bond: tuple[int, int],
) -> list[set[int]] | None:
    """Force a split at ``split_bond`` and return the resulting components."""
    a, b = sorted(split_bond)
    g = _skeleton_graph(n_atoms, bonds_R, [(a, b)])
    comps = [set(c) for c in nx.connected_components(g)]
    comps.sort(key=lambda c: (-len(c), min(c) if c else 0))
    if len(comps) < 2:
        return None
    return comps[:2]
