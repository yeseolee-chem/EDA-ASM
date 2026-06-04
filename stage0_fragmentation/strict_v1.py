"""Strict fragment definition v1 — implements the decision tree from
``strict_fragment_definition_v1.md``.

Operates on connectivity matrices (no bond orders). Bond perception comes
from Cordero radii via :func:`eda_asm.phase1.bonds.detect_bonds`. Suitable
for the Halo8 Phase-1 use case where R/P come from MD frames rather than
fully sanitised RDKit ``Mol`` objects.

Decision tree (per spec Part 3):

    Q1: R has ≥ 2 components?           → Case A (bimolecular reactant)
    Q2: P has ≥ 2 components?           → Case B (dissociation)
    Q3: any migrating atoms?            → branch
    Q4: |M| == 0  AND reactive graph bipartite → Case C1
    Q5: |M|/|V| ≤ 0.30  AND breaking graph bipartite → Case C2
    Q6: migrating atoms form a cluster? → Case D1 (clustered)
                                         else Case D2 (concerted, strain-only)

The result reuses :class:`stage0_fragmentation.types.FragmentationResult`,
adding the case label to ``notes`` and the strict-spec-specific fields via
``fallback_strategy`` (e.g. ``"strain_only"`` for Case D2).
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np

from .bond_detection import detect_bonds_strict
from .types import FragmentationResult

MIGRATING_RATIO_THRESHOLD = 0.30


def _connectivity(numbers: np.ndarray, coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted(b)) for b in detect_bonds_strict(numbers, coords)}


def _graph(n: int, edges: Iterable[tuple[int, int]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    return g


def _detect_strict_migrating(
    bonds_R: set[tuple[int, int]],
    bonds_P: set[tuple[int, int]],
    n: int,
) -> list[dict]:
    """Definition D6 (strict) on connectivity matrices.

    Atom k qualifies iff:
      - k loses at least one full bond (partner only present in R)
      - k gains at least one full bond to a *different* partner (only in P)
      - |loss| == |gain|  (balanced — same number of partners swapped)

    The lone-pair clause (|ΔB_kk| ≤ 1) is automatically satisfied here
    because we don't track lone-pair populations on the diagonal.
    """
    breaking = bonds_R - bonds_P
    forming = bonds_P - bonds_R
    out: list[dict] = []
    for k in range(n):
        lost: list[int] = []
        for i, j in breaking:
            if i == k:
                lost.append(j)
            elif j == k:
                lost.append(i)
        gained: list[int] = []
        for i, j in forming:
            if i == k:
                gained.append(j)
            elif j == k:
                gained.append(i)
        if not lost or not gained:
            continue
        if set(lost) & set(gained):
            continue  # shouldn't happen at connectivity level
        if len(lost) != len(gained):
            continue
        out.append({
            "atom": k,
            "from": sorted(lost),
            "to": sorted(gained),
            "loss": len(lost),
            "gain": len(gained),
        })
    return out


def _h_only_check(numbers: np.ndarray, fragment: set[int]) -> bool:
    """AP1: True if the fragment contains no heavy atoms (Z ≥ 2)."""
    return all(int(numbers[a]) < 2 for a in fragment)


def _ensure_heavy(
    fragments: list[set[int]],
    numbers: np.ndarray,
    notes: list[str],
) -> list[set[int]]:
    """If any fragment is H-only (AP1), move its H atoms into the other
    fragment AND signal failure (return [{ALL}] so the caller can decide
    whether to keep or fall back to strain_only).
    """
    if len(fragments) != 2:
        return fragments
    a, b = fragments
    if _h_only_check(numbers, a) and a:
        notes.append("AP1: fragment A was H-only → merged into B; partition collapsed")
        return [a | b]
    if _h_only_check(numbers, b) and b:
        notes.append("AP1: fragment B was H-only → merged into A; partition collapsed")
        return [a | b]
    return fragments


def _connected_after(
    fragment: set[int],
    bonds: set[tuple[int, int]],
) -> bool:
    """A7: True if the fragment forms a single connected component."""
    if not fragment:
        return True
    g = nx.Graph()
    g.add_nodes_from(fragment)
    for i, j in bonds:
        if i in fragment and j in fragment:
            g.add_edge(i, j)
    return nx.is_connected(g)


def _repair_disconnected(
    fragments: list[set[int]],
    bonds: set[tuple[int, int]],
    notes: list[str],
) -> list[set[int]]:
    """Move any orphan atom (no neighbour in its own fragment) to the
    fragment where its neighbours actually live. Repeats until stable."""
    if len(fragments) != 2:
        return fragments
    neigh: dict[int, set[int]] = {}
    for i, j in bonds:
        neigh.setdefault(i, set()).add(j)
        neigh.setdefault(j, set()).add(i)

    progress = True
    iters = 0
    while progress and iters < 10:
        progress = False
        iters += 1
        for idx, frag in enumerate(fragments):
            other = fragments[1 - idx]
            for atom in list(frag):
                own = sum(1 for nb in neigh.get(atom, set()) if nb in frag)
                across = sum(1 for nb in neigh.get(atom, set()) if nb in other)
                if own == 0 and across > 0:
                    frag.discard(atom)
                    other.add(atom)
                    notes.append(
                        f"repair: moved atom {atom} from fragment {idx} (no own neighbours) "
                        f"to fragment {1 - idx}"
                    )
                    progress = True
    return fragments


def _route_non_reactive(
    fragments: list[set[int]],
    n: int,
    bonds_preserved: set[tuple[int, int]],
    notes: list[str],
) -> list[set[int]]:
    """Assign each not-yet-allocated atom by majority vote among its
    preserved-bond neighbours; ties broken by smaller fragment first."""
    assigned = set().union(*fragments) if fragments else set()
    pending = [a for a in range(n) if a not in assigned]
    if not pending:
        return fragments

    neigh: dict[int, set[int]] = {a: set() for a in range(n)}
    for i, j in bonds_preserved:
        neigh[i].add(j)
        neigh[j].add(i)

    progress = True
    while pending and progress:
        progress = False
        leftover: list[int] = []
        for atom in pending:
            votes: dict[int, int] = {}
            for nb in neigh[atom]:
                for idx, frag in enumerate(fragments):
                    if nb in frag:
                        votes[idx] = votes.get(idx, 0) + 1
            if votes:
                best = sorted(
                    votes.keys(),
                    key=lambda k: (votes[k], -len(fragments[k])),
                    reverse=True,
                )[0]
                fragments[best].add(atom)
                progress = True
            else:
                leftover.append(atom)
        pending = leftover

    # Drop remaining isolates into the smaller fragment.
    for atom in pending:
        target = sorted(range(len(fragments)), key=lambda k: len(fragments[k]))[0]
        fragments[target].add(atom)
        notes.append(f"isolated atom {atom} → fragment {target} (smallest)")
    return fragments


def _strain_only(n: int, notes: list[str]) -> FragmentationResult:
    notes.append("Case D2: concerted full-skeleton — strain_only fallback")
    return FragmentationResult(
        fragments=[set(range(n))],
        migrating_atoms=[],
        reactive_bonds=[],
        cap_sites={0: []},
        is_pure_rearrangement=True,
        fallback_strategy="strain_only",
        notes=notes,
    )


def _bipartite_split_seeds(
    breaking_or_reactive_edges: set[tuple[int, int]],
    exclude_atoms: set[int] | None = None,
) -> tuple[set[int], set[int]] | None:
    """2-colour the given graph; return the two atom sets if bipartite.

    ``exclude_atoms`` (optional) drops those nodes from the colouring so
    they can be routed afterwards (e.g. migrating atoms are placed by
    destination, not by bipartite colour).
    """
    if not breaking_or_reactive_edges:
        return None
    g = nx.Graph()
    for i, j in breaking_or_reactive_edges:
        g.add_edge(i, j)
    if exclude_atoms:
        g.remove_nodes_from(a for a in exclude_atoms if g.has_node(a))
    if g.number_of_nodes() == 0 or not nx.is_bipartite(g):
        return None
    seed_a: set[int] = set()
    seed_b: set[int] = set()
    for cc in nx.connected_components(g):
        sub = g.subgraph(cc)
        if sub.number_of_edges() == 0:
            continue
        coloring = nx.bipartite.color(sub)
        for atom, color in coloring.items():
            (seed_a if color == 0 else seed_b).add(atom)
    if not seed_a or not seed_b:
        return None
    return seed_a, seed_b


def _route_migrating(
    fragments: list[set[int]],
    migrating: list[dict],
    notes: list[str],
) -> list[set[int]]:
    """A6: each migrating atom k joins the fragment of its destination atom.

    Tie-break: prefer destination atom with the most "to" partners; if still
    tied, lower fragment index.
    """
    for m in migrating:
        atom = m["atom"]
        if any(atom in f for f in fragments):
            continue
        votes: dict[int, int] = {}
        for dest in m["to"]:
            for idx, frag in enumerate(fragments):
                if dest in frag:
                    votes[idx] = votes.get(idx, 0) + 1
        if votes:
            best = sorted(votes.keys(), key=lambda k: (-votes[k], k))[0]
            fragments[best].add(atom)
            notes.append(f"A6: migrating atom {atom} → fragment {best}")
        else:
            fragments[0].add(atom)
            notes.append(
                f"A6 fallback: migrating atom {atom} → fragment 0 "
                f"(destinations {m['to']} not in seeds)"
            )
    return fragments


def fragmentation_strict_v1(
    numbers: list[int] | np.ndarray,
    coords_R: list[list[float]] | np.ndarray,
    coords_P: list[list[float]] | np.ndarray,
) -> FragmentationResult:
    """Top-level entry point. Implements the spec's Part 3 decision tree."""
    numbers_np = np.asarray(numbers, dtype=int)
    coords_R_np = np.asarray(coords_R, dtype=float)
    coords_P_np = np.asarray(coords_P, dtype=float)
    n = len(numbers_np)
    notes: list[str] = []

    bonds_R = _connectivity(numbers_np, coords_R_np)
    bonds_P = _connectivity(numbers_np, coords_P_np)
    breaking = bonds_R - bonds_P
    forming = bonds_P - bonds_R
    preserved = bonds_R & bonds_P

    reactive_bonds = sorted(breaking | forming)

    G_R = _graph(n, bonds_R)
    G_P = _graph(n, bonds_P)
    R_comps = list(nx.connected_components(G_R))
    P_comps = list(nx.connected_components(G_P))

    # ------------------------------------------------------------------ Q1
    if len(R_comps) >= 2:
        notes.append(f"Q1: R has {len(R_comps)} components → Case A")
        fragments = _case_A(R_comps, P_comps, numbers_np, notes)
    # ------------------------------------------------------------------ Q2
    elif len(P_comps) >= 2:
        notes.append(f"Q2: P has {len(P_comps)} components → Case B")
        fragments = _case_B(P_comps, numbers_np, notes)
    else:
        # ------------------------------------------------------------- Q3+
        migrating = _detect_strict_migrating(bonds_R, bonds_P, n)
        ratio = len(migrating) / n if n > 0 else 0.0
        notes.append(
            f"Q3: |migrating|={len(migrating)}, |V|={n}, ratio={ratio:.2%}"
        )

        if not migrating:
            # Q4
            split = _bipartite_split_seeds(breaking | forming)
            if split:
                notes.append("Q4: bipartite reactive graph → Case C1")
                fragments = _case_C(split, n, preserved, [], notes)
            else:
                notes.append("Q4: non-bipartite reactive graph and no migrating atoms → escalate (E1)")
                return _strain_only(n, notes)
        else:
            # Q5
            G_break = _graph(n, breaking)
            break_bipartite = nx.is_bipartite(G_break)
            if ratio <= MIGRATING_RATIO_THRESHOLD and break_bipartite:
                notes.append(
                    f"Q5: ratio ≤ {MIGRATING_RATIO_THRESHOLD:.0%} and breaking graph bipartite → Case C2"
                )
                fragments = _case_C2(
                    n, numbers_np, bonds_R, breaking, forming, preserved, migrating, notes
                )
                if fragments is None:
                    return _strain_only(n, notes)
            else:
                # Q6
                mig_atoms = [m["atom"] for m in migrating]
                G_mig = G_R.subgraph(mig_atoms)
                cluster = nx.is_connected(G_mig) if mig_atoms else False
                if cluster and len(mig_atoms) < n:
                    notes.append("Q6: migrating cluster → Case D1")
                    fragments = _case_D1(set(mig_atoms), n, preserved, notes)
                    # If D1 produces a disconnected fragment, fall back to
                    # C2-style strategies which redistribute atoms across the
                    # bipartite/path cut.
                    if not all(_connected_after(f, bonds_R | bonds_P) for f in fragments):
                        notes.append("D1 produced disconnected fragment → trying C2 strategies")
                        c2 = _case_C2(n, numbers_np, bonds_R, breaking, forming, preserved, migrating, notes)
                        if c2 is not None:
                            fragments = c2
                else:
                    notes.append("Q6: distributed migration → Case D2 (strain_only)")
                    return _strain_only(n, notes)

    fragments = _ensure_heavy(fragments, numbers_np, notes)
    fragments = [f for f in fragments if f]  # drop empties
    if len(fragments) < 2:
        notes.append("Final partition collapsed to one fragment → strain_only")
        return _strain_only(n, notes)

    fragments = _repair_disconnected(fragments, bonds_R | bonds_P, notes)
    if len(fragments) < 2 or any(not f for f in fragments):
        notes.append("Repair pass collapsed partition → strain_only")
        return _strain_only(n, notes)

    cap_sites = _cap_sites(fragments, bonds_R)
    if not all(_connected_after(f, bonds_R | bonds_P) for f in fragments):
        notes.append("AP5: a fragment is disconnected after H-cap → strain_only")
        return _strain_only(n, notes)

    return FragmentationResult(
        fragments=fragments,
        migrating_atoms=_detect_strict_migrating(bonds_R, bonds_P, n),
        reactive_bonds=reactive_bonds,
        cap_sites=cap_sites,
        is_pure_rearrangement=False,
        fallback_strategy=None,
        notes=notes,
    )


def _case_A(
    R_comps: list[set[int]],
    P_comps: list[set[int]],
    numbers: np.ndarray,
    notes: list[str],
) -> list[set[int]]:
    """Case A: each R component is a fragment. Verify P agrees (axiom A2)."""
    R_comps_sorted = sorted(R_comps, key=lambda c: (-len(c), min(c) if c else 0))
    if len(R_comps_sorted) > 2:
        notes.append(f"Case A: {len(R_comps_sorted)} R components — merging extras into smaller seed")
    primary = R_comps_sorted[0]
    secondary: set[int] = set()
    for c in R_comps_sorted[1:]:
        secondary |= c
    fragments: list[set[int]] = [set(primary), secondary]
    if len(P_comps) >= 2:
        # Optional sanity: warn if the partition disagrees.
        primary_atoms = primary
        for pc in P_comps:
            if pc & primary_atoms and not pc.issubset(primary_atoms) and not pc.isdisjoint(primary_atoms):
                notes.append("AP3 warning: R/P component assignment disagrees")
                break
    return fragments


def _case_B(
    P_comps: list[set[int]],
    numbers: np.ndarray,
    notes: list[str],
) -> list[set[int]]:
    """Case B: use P components to partition R atoms (same atom indices)."""
    P_comps_sorted = sorted(P_comps, key=lambda c: (-len(c), min(c) if c else 0))
    if len(P_comps_sorted) > 2:
        notes.append(f"Case B: {len(P_comps_sorted)} P components — merging extras into smaller seed")
    primary = P_comps_sorted[0]
    secondary: set[int] = set()
    for c in P_comps_sorted[1:]:
        secondary |= c
    return [set(primary), secondary]


def _case_C(
    split: tuple[set[int], set[int]],
    n: int,
    preserved: set[tuple[int, int]],
    migrating: list[dict],
    notes: list[str],
) -> list[set[int]]:
    seed_a, seed_b = split
    fragments = [set(seed_a), set(seed_b)]
    if migrating:
        # Migrating atoms might already be in seeds because they touch the
        # cut bond; if so, leave them; if not, route by destination.
        fragments = _route_migrating(fragments, migrating, notes)
    fragments = _route_non_reactive(fragments, n, preserved, notes)
    return fragments


def _case_C2(
    n: int,
    numbers: np.ndarray,
    bonds_R: set[tuple[int, int]],
    breaking: set[tuple[int, int]],
    forming: set[tuple[int, int]],
    preserved: set[tuple[int, int]],
    migrating: list[dict],
    notes: list[str],
) -> list[set[int]] | None:
    """Case C2 with multiple split strategies; returns None if all fail.

    Strategy 1 — bipartite of breaking bonds, migrating atoms removed from seeds.
    Strategy 2 — same but using the breaking ∪ forming graph.
    Strategy 3 — for a single migrating atom, cut the preserved-bond path
        between its origin and destination at the heaviest-balance edge.
    """
    mig_set = {m["atom"] for m in migrating}

    # Strategies 1, 2 — bipartite-based.
    for label, edges in (("breaking", breaking), ("breaking∪forming", breaking | forming)):
        split = _bipartite_split_seeds(edges)
        if split is None:
            continue
        seed_a, seed_b = split[0] - mig_set, split[1] - mig_set
        if not seed_a or not seed_b:
            continue
        notes.append(f"C2-strategy: bipartite {label}, migrating excluded from seeds")
        return _case_C((seed_a, seed_b), n, preserved, migrating, notes)

    # Strategy 3 — path cut. Useful for 1,n-shifts where the bipartite seed
    # collapses (e.g. only one migrating H bridges two heavy fragments).
    for m in migrating:
        atom = m["atom"]
        for origin in m["from"]:
            for dest in m["to"]:
                if origin == dest:
                    continue
                cut = _path_cut(n, preserved, origin, dest)
                if cut is None:
                    continue
                seed_a, seed_b = cut
                if mig_set & seed_a:
                    seed_a -= mig_set
                if mig_set & seed_b:
                    seed_b -= mig_set
                if not seed_a or not seed_b:
                    continue
                notes.append(
                    f"C2-strategy: path-cut between origin {origin} and destination {dest}"
                )
                return _case_C((seed_a, seed_b), n, preserved, migrating, notes)

    # Strategy 4 — try every preserved bond as a candidate cut. Pick the
    # cut that gives a connected 2-partition with the best balance of:
    # (a) a heavy atom on each side, (b) reactive bonds crossing the cut.
    cut_candidates: list[tuple[float, set[int], set[int]]] = []
    for cut_bond in preserved:
        a, b = cut_bond
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for i, j in preserved:
            if (i, j) == cut_bond or (j, i) == cut_bond:
                continue
            g.add_edge(i, j)
        comps = [set(c) for c in nx.connected_components(g)]
        if len(comps) != 2:
            continue
        f1, f2 = comps
        # Heavy-atom requirement (AP1) — both sides need at least one Z ≥ 2.
        if all(int(numbers[a]) < 2 for a in f1) or all(int(numbers[a]) < 2 for a in f2):
            continue
        # Score: prefer balanced sizes and reactive bonds that cross.
        balance = min(len(f1), len(f2)) / max(len(f1), len(f2))
        crossing = sum(
            1 for (i, j) in (breaking | forming)
            if (i in f1 and j in f2) or (i in f2 and j in f1)
        )
        score = crossing * 2 + balance
        cut_candidates.append((score, f1, f2))
    if cut_candidates:
        cut_candidates.sort(key=lambda x: -x[0])
        score, f1, f2 = cut_candidates[0]
        notes.append(
            f"C2-strategy: preserved-bond brute-cut (score={score:.2f}, "
            f"sizes={len(f1)}/{len(f2)})"
        )
        f1 = f1 - mig_set
        f2 = f2 - mig_set
        if f1 and f2:
            return _case_C((f1, f2), n, preserved, migrating, notes)

    notes.append("Case C2: no strategy produced a viable 2-fragment split")
    return None


def _path_cut(
    n: int,
    preserved: set[tuple[int, int]],
    origin: int,
    dest: int,
) -> tuple[set[int], set[int]] | None:
    """Find a preserved-bond path from origin → dest, cut its midpoint, and
    return a 2-partition of all atoms.

    Atoms with no preserved-bond connection (isolated in the preserved graph,
    typically migrating H atoms) are merged into whichever side their full-
    graph neighbours dominate.
    """
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i, j in preserved:
        g.add_edge(i, j)
    if not nx.has_path(g, origin, dest):
        return None
    path = nx.shortest_path(g, origin, dest)
    if len(path) < 3:
        return None
    mid = len(path) // 2
    a, b = path[mid - 1], path[mid]
    g.remove_edge(a, b)
    comps = [set(c) for c in nx.connected_components(g)]
    # Find the two components containing origin and dest respectively.
    side_origin: set[int] = set()
    side_dest: set[int] = set()
    leftovers: list[set[int]] = []
    for c in comps:
        if origin in c:
            side_origin = c
        elif dest in c:
            side_dest = c
        else:
            leftovers.append(c)
    if not side_origin or not side_dest:
        return None
    # Distribute leftover (isolated) atoms — pick whichever side has more
    # of their neighbours in the *full* (incl. reactive) graph.
    if leftovers:
        # Build a neighbour map from the original preserved set + a synthetic
        # bond between every atom and its closest preserved-bond cluster.
        # For now, just split leftovers by atom-by-atom heuristic:
        for c in leftovers:
            # Pick the side whose seed atoms are closer to any atom of c.
            dists_o = sum(_min_path(g, a, side_origin) for a in c) / max(len(c), 1)
            dists_d = sum(_min_path(g, a, side_dest) for a in c) / max(len(c), 1)
            if dists_o <= dists_d:
                side_origin |= c
            else:
                side_dest |= c
    return side_origin, side_dest


def _min_path(g: "nx.Graph", a: int, target_set: set[int]) -> float:
    """Return the smallest shortest-path distance from a to any atom in
    target_set. ``inf`` if no path exists. Nodes that aren't in g are
    treated as inf."""
    if a not in g:
        return float("inf")
    best = float("inf")
    for t in target_set:
        if t in g and nx.has_path(g, a, t):
            d = nx.shortest_path_length(g, a, t)
            if d < best:
                best = d
    return best


def _case_D1(
    cluster: set[int],
    n: int,
    preserved: set[tuple[int, int]],
    notes: list[str],
) -> list[set[int]]:
    """Cluster vs framework split."""
    fragments: list[set[int]] = [set(cluster), set(range(n)) - cluster]
    return fragments


def _cap_sites(
    fragments: list[set[int]],
    bonds_R: set[tuple[int, int]],
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
