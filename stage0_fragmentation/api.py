"""Top-level ``run_fragmentation`` entry point."""
from __future__ import annotations

import logging
from typing import Any

from rdkit import Chem

from .be_matrix import build_be_matrix, validate_be_matrix
from .capping import find_cap_sites
from .migration import detect_migrating_atoms, reactive_bonds_from_delta
from .partition import (
    bonds_from_be,
    connected_component_analysis,
    route_migrations,
    validate_fragmentation,
)
from .rearrangement import migration_clustering, split_by_user_hint
from .types import FragmentationResult

log = logging.getLogger(__name__)


def run_fragmentation(
    mol_R: Chem.Mol,
    mol_P: Chem.Mol,
    user_hint: dict[str, Any] | None = None,
    verbose: bool = False,
) -> FragmentationResult:
    """Build the fragment partition for ASM-EDA from atom-mapped R and P.

    See ``stage0_fragmentation_spec.md`` for the algorithm.
    """
    if mol_R.GetNumAtoms() != mol_P.GetNumAtoms():
        raise ValueError(
            f"mol_R has {mol_R.GetNumAtoms()} atoms but mol_P has {mol_P.GetNumAtoms()}"
        )
    n = mol_R.GetNumAtoms()
    notes: list[str] = []

    # 1. BE matrices.
    B_R = build_be_matrix(mol_R)
    B_P = build_be_matrix(mol_P)
    validate_be_matrix(B_R, mol_R)
    validate_be_matrix(B_P, mol_P)

    delta = B_P - B_R

    # 2. Reactive bonds (off-diagonal nonzero entries).
    reactive_bonds = reactive_bonds_from_delta(delta)
    if verbose:
        log.info("reactive bonds: %s", reactive_bonds)

    # 3. Migrating atoms.
    migrating = detect_migrating_atoms(B_R, B_P)
    if verbose:
        log.info("migrating atoms: %s", migrating)

    # 4. Connected components after removing reactive bonds + migrating atoms.
    bonds_R_full = bonds_from_be(B_R)
    components = connected_component_analysis(
        n, bonds_R_full, reactive_bonds, migrating
    )

    is_pure_rearrangement = len(components) < 2

    fallback: str | None = None
    if not is_pure_rearrangement:
        # 5. Pick fragment seeds. Strategy:
        #    a. Build a "key reactive bond" graph over components, where
        #       edges come from bonds that are *fully* formed or broken
        #       (B_R=0 ↔ B_P>0) and from migrating-atom from/to pairs.
        #    b. If the graph is bipartite, the two colors give the seed
        #       groups (this works for SN2, DA, ring contraction…).
        #    c. Otherwise, fall back to ranking components by endpoint-touch
        #       and size.
        import networkx as nx  # local import keeps the optional dep cost low

        key_bonds = [
            (i, j)
            for (i, j) in reactive_bonds
            if (B_R[i, j] == 0 and B_P[i, j] > 0)
            or (B_R[i, j] > 0 and B_P[i, j] == 0)
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
        used_indices: set[int] = set()
        for cc in nx.connected_components(cg):
            sub = cg.subgraph(cc)
            if sub.number_of_edges() == 0:
                continue
            if not nx.is_bipartite(sub):
                continue
            color = nx.bipartite.color(sub)
            for ci, c in color.items():
                target = seed_a if c == 0 else seed_b
                target.update(components[ci])
                used_indices.add(ci)

        if seed_a and seed_b:
            seeds = [seed_a, seed_b]
            ranked = [components[i] for i in range(len(components)) if i not in used_indices]
        else:
            # Fallback: rank by endpoint-touch + size.
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
            ranked = ranked_full[2:]
        # 6. Route migrations to whichever component contains their dominant
        #    destination.
        fragments, route_notes = route_migrations(seeds, migrating, delta)
        notes.extend(route_notes)
        # Atoms outside the two largest components have to land *somewhere*.
        # We assign each stray to the fragment that already contains the most
        # of its R-graph neighbors (so isolated H atoms follow the carbon they
        # were bonded to). Iterate until stable in case strays only neighbor
        # other strays.
        assigned = set().union(*fragments) if fragments else set()
        strays = [a for comp in ranked for a in comp if a not in assigned]
        # Build neighbor lookup using NON-reactive R bonds only — reactive
        # bonds are precisely those that disappear or appear, so they're not
        # a reliable signal for which fragment an atom should follow.
        reactive_set = {tuple(sorted(b)) for b in reactive_bonds}
        neighbors: dict[int, set[int]] = {a: set() for a in range(n)}
        for i, j in bonds_R_full:
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
                    chosen: int = int(ranked_idx[0])
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
        # Final fallback for atoms with no neighbor in any fragment.
        for atom in strays:
            sizes = [(len(frag), idx) for idx, frag in enumerate(fragments)]
            sizes.sort()
            chosen = int(sizes[0][1])
            fragments[chosen].add(atom)
            assigned.add(atom)
            notes.append(
                f"merged isolated stray atom {atom} into smallest fragment {chosen}"
            )
    else:
        # 7. Rearrangement fallback.
        notes.append("no clean 2-component split; entering rearrangement fallback")
        fragments_opt: list[set[int]] | None = None
        if user_hint and "split_bond" in user_hint:
            split = split_by_user_hint(n, bonds_R_full, tuple(user_hint["split_bond"]))
            if split is not None:
                fragments_opt = split
                fallback = "user_hint"
                notes.append(f"used user_hint split_bond={user_hint['split_bond']}")
        if fragments_opt is None:
            split = migration_clustering(n, bonds_R_full, reactive_bonds, migrating)
            if split is not None:
                fragments_opt = split
                fallback = "migration_clustering"
                notes.append("used migration_clustering")
        if fragments_opt is None:
            # Last resort: the entire molecule as one fragment, mark
            # strain-only — Stage 5 will handle this.
            fragments_opt = [set(range(n))]
            fallback = "strain_only"
            notes.append("no fallback succeeded; flagging strain_only")
        fragments = fragments_opt

    # 8. H-cap sites.
    cap_sites = find_cap_sites(fragments, mol_R)

    result = FragmentationResult(
        fragments=fragments,
        migrating_atoms=migrating,
        reactive_bonds=reactive_bonds,
        cap_sites=cap_sites,
        is_pure_rearrangement=is_pure_rearrangement,
        fallback_strategy=fallback,
        notes=notes,
    )
    validate_fragmentation(result, mol_R, mol_P)
    return result
