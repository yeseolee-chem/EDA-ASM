"""Per-pattern fragmentation routines (P0/P1/P2/P3) from spec §5-A.3-5.

Each fragmenter takes the pre-computed bond-change sets plus the R-frame
positions and returns a :class:`FragmentationResult`.

* **P0_BIMOL** — already two molecules; emit each component as its own
  fragment (closed-shell singlet, no caps). Strain channel still defined.
* **P1_OPEN** — single Ï€-bond formed/broken; cut along that bond, two
  doublet radical fragments, no capping.
* **P2_CLOSED** — multiple bond changes concentrated in one core; remove
  all reactive bonds and partition the remaining graph; closed-shell.
* **P3_TETHER** — two reactive components connected by a non-reactive
  tether path; emit ``reactive_A``, ``reactive_B``, ``tether`` (all
  closed-shell singlets) and cap each at the cut bonds with H atoms at
  the standard 1.09 Ã… distance (spec §5-A.5-2).
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from .types import FragmentSpec, FragmentationResult

BondSet = set[tuple[int, int]]

# Standard C–H bond length used as cap. Spec §5-A.5-2 fixes this to avoid
# length-dependent tether artefacts (Bickelhaupt & Houk 2017, §5.3).
R_CAP_H = 1.09  # Å


def _graph_without(bonds: BondSet, drop: BondSet, n: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j in bonds:
        if (i, j) in drop or (j, i) in drop:
            continue
        G.add_edge(i, j)
    return G


def _add_cap_hydrogens(
    positions: np.ndarray,
    frag_atoms: list[int],
    attachment_points: list[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray]]]:
    """Compute cap-H positions for a fragment.

    ``attachment_points`` is a list of ``(f_idx, t_idx)`` where ``f_idx``
    is an atom in the fragment and ``t_idx`` is its severed neighbour
    outside the fragment. Each cap H sits at the canonical 1.09 Å along
    the original ``f_idx → t_idx`` bond direction.

    Returns ``(cap_positions, cap_attachment)``:
    - ``cap_positions``: ``(M, 3)`` array of H positions.
    - ``cap_attachment``: ``[(t_idx, frag_local_cap_idx, h_position), ...]``
      where ``frag_local_cap_idx`` is the index of this cap H within the
      *capped fragment* (atoms appended after the original fragment atoms).
    """
    cap_positions: list[np.ndarray] = []
    cap_attachment: list[tuple[int, int, np.ndarray]] = []
    for f_idx, t_idx in attachment_points:
        vec = positions[t_idx] - positions[f_idx]
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            unit = np.array([1.0, 0.0, 0.0])
        else:
            unit = vec / norm
        h_pos = positions[f_idx] + R_CAP_H * unit
        local_cap_idx = len(frag_atoms) + len(cap_positions)
        cap_positions.append(h_pos)
        cap_attachment.append((int(t_idx), int(local_cap_idx), h_pos))
    if cap_positions:
        return np.vstack(cap_positions), cap_attachment
    return np.empty((0, 3)), cap_attachment


# --------------------------------------------------------------------------- #
# P0
# --------------------------------------------------------------------------- #

def fragment_P0(
    n_atoms: int,
    debug: dict[str, Any],
) -> FragmentationResult:
    """P0 is already two molecules — surface them as two closed-shell
    fragments. If only one connected component existed (the "no bond
    changes" fallback), emit a single whole-system fragment.
    """
    components = debug.get("reactant_components")
    if not components:
        return FragmentationResult(
            pattern="P0_BIMOL",
            fragments=[
                FragmentSpec(
                    atom_indices=np.arange(n_atoms),
                    role="whole",
                    multiplicity=1,
                )
            ],
            cap_h_positions=None,
            confidence=1.0,
            notes=debug.get("note", "no decomposition needed"),
        )
    fragments: list[FragmentSpec] = []
    for k, comp in enumerate(components):
        fragments.append(
            FragmentSpec(
                atom_indices=np.array(sorted(comp), dtype=int),
                role=f"reactive_{chr(ord('A') + k)}",
                multiplicity=1,
            )
        )
    return FragmentationResult(
        pattern="P0_BIMOL",
        fragments=fragments,
        cap_h_positions=None,
        confidence=1.0,
        notes=f"{len(components)} reactant molecules detected — standard ASM",
    )


# --------------------------------------------------------------------------- #
# P1
# --------------------------------------------------------------------------- #

def fragment_P1(
    n_atoms: int,
    bonds_R: BondSet,
    bonds_broken: BondSet,
    bonds_formed: BondSet,
) -> FragmentationResult | None:
    """Cut along the unique reactive bond → two doublet radical fragments.

    Returns ``None`` if the cut leaves the molecule connected (i.e. the
    reactive bond is inside a ring), so the caller can fall back to P2.
    """
    reactive = bonds_broken | bonds_formed
    if not reactive:
        return None
    # Prefer a formed bond (cyclisation), else a broken bond (dissociation).
    target_bond = next(iter(bonds_formed)) if bonds_formed else next(iter(bonds_broken))
    i, j = target_bond
    # Graph: R bonds minus the target bond. If the bond is formed (not in
    # R), removing it from R is a no-op — that is the right behaviour:
    # forming a bond between i and j means in R they are disjoint, and
    # cutting at the (would-be) bond simply means looking at R's two
    # components that contain i and j respectively.
    G = _graph_without(bonds_R, {target_bond}, n_atoms)

    if i not in G.nodes:
        G.add_node(i)
    if j not in G.nodes:
        G.add_node(j)
    comp_i = nx.node_connected_component(G, i)
    comp_j = nx.node_connected_component(G, j)
    if comp_i == comp_j:
        # Cutting at this bond did not disconnect — ring system, P2 handles it.
        return None

    # Any other atoms not in comp_i or comp_j (e.g. isolated H from H abstraction)
    # are absorbed into whichever fragment is nearer along R bonds.
    leftover = set(range(n_atoms)) - comp_i - comp_j
    if leftover:
        # In a single-component reactant, the only way to land in leftover is
        # if we already cut a bond — leftovers should be empty. Defensive
        # absorb-into-A.
        comp_i |= leftover

    frag_A = FragmentSpec(
        atom_indices=np.array(sorted(comp_i), dtype=int),
        role="reactive_A",
        multiplicity=2,
        cap_attachment=None,
    )
    frag_B = FragmentSpec(
        atom_indices=np.array(sorted(comp_j), dtype=int),
        role="reactive_B",
        multiplicity=2,
        cap_attachment=None,
    )
    n_react = len(bonds_broken | bonds_formed)
    return FragmentationResult(
        pattern="P1_OPEN",
        fragments=[frag_A, frag_B],
        cap_h_positions=None,
        confidence=1.0 if n_react == 1 else 0.7,
        notes=f"cut along bond {tuple(sorted(target_bond))} "
              f"(n_broken={len(bonds_broken)}, n_formed={len(bonds_formed)})",
    )


# --------------------------------------------------------------------------- #
# P2
# --------------------------------------------------------------------------- #

def fragment_P2(
    n_atoms: int,
    bonds_R: BondSet,
    bonds_broken: BondSet,
    bonds_formed: BondSet,
    positions_R: np.ndarray,
    subtype: str = "P2A_PI_CYCLO",
) -> FragmentationResult | None:
    """Remove all reactive bonds → take the two largest components as the
    fragments; absorb any stray smaller components into whichever larger
    fragment they sit closer to (spec §5-A.4-2).

    The ``subtype`` (FernÃ¡ndez & Bickelhaupt, *Chem. Soc. Rev.* 2014, §4.2)
    decides fragment multiplicity:

    * ``P2A_PI_CYCLO`` — only Ï€ bonds break: both fragments are
      closed-shell singlets (mult = 1, confidence 0.8).
    * ``P2B_SIGMA_REARRANGE`` — at least one σ-only bond breaks: both
      fragments are doublet radicals (mult = 2). σ-bond cleavage carries
      multi-reference character that single-reference KS-DFT EDA can
      only approximate, so confidence is demoted to 0.6.
    """
    reactive = bonds_broken | bonds_formed
    if not reactive:
        return None
    G = _graph_without(bonds_R, reactive, n_atoms)
    components = [set(c) for c in nx.connected_components(G)]
    components.sort(key=len, reverse=True)
    if len(components) < 2:
        return None  # cutting all reactive bonds did not disconnect

    comp_A = components[0]
    comp_B = components[1]
    extras = components[2:]
    # Absorb each extra component into whichever of A/B it is closer to in space.
    for extra in extras:
        ex_pts = positions_R[list(extra)]
        a_pts = positions_R[list(comp_A)]
        b_pts = positions_R[list(comp_B)]
        # Min pairwise distance from extra to A and B.
        d_A = np.min(np.linalg.norm(ex_pts[:, None, :] - a_pts[None, :, :], axis=-1))
        d_B = np.min(np.linalg.norm(ex_pts[:, None, :] - b_pts[None, :, :], axis=-1))
        if d_A <= d_B:
            comp_A |= extra
        else:
            comp_B |= extra

    # Sub-classification: P2A (Ï€-cycloaddition, closed-shell singlets)
    # vs P2B (σ-skeletal rearrangement, doublet radical fragments).
    if subtype == "P2B_SIGMA_REARRANGE":
        frag_mult = 2
        base_conf = 0.6  # σ-bond homolysis carries multi-reference risk
    else:
        frag_mult = 1
        base_conf = 0.8

    frag_A = FragmentSpec(
        atom_indices=np.array(sorted(comp_A), dtype=int),
        role="reactive_A",
        multiplicity=frag_mult,
        cap_attachment=None,
    )
    frag_B = FragmentSpec(
        atom_indices=np.array(sorted(comp_B), dtype=int),
        role="reactive_B",
        multiplicity=frag_mult,
        cap_attachment=None,
    )
    # Sanity: A ∪ B should cover all atoms (we absorbed every extra component).
    covered = comp_A | comp_B
    missing = set(range(n_atoms)) - covered
    if missing:
        comp_A |= missing
        frag_A = FragmentSpec(
            atom_indices=np.array(sorted(comp_A), dtype=int),
            role="reactive_A",
            multiplicity=frag_mult,
            cap_attachment=None,
        )

    confidence = base_conf if len(extras) == 0 else max(0.4, base_conf - 0.2)
    return FragmentationResult(
        pattern="P2_CLOSED",
        fragments=[frag_A, frag_B],
        cap_h_positions=None,
        confidence=confidence,
        notes=f"[{subtype}] {len(bonds_broken)} bonds broken, "
              f"{len(bonds_formed)} formed; "
              f"{len(extras)} stray components absorbed by proximity",
    )


# --------------------------------------------------------------------------- #
# P5 — simple monovalent-atom (H) 1,2-shift
# --------------------------------------------------------------------------- #


_ELEM_LABEL = {1: "H", 9: "F", 17: "Cl", 35: "Br", 53: "I"}


def fragment_P5_hshift(
    n_atoms: int,
    migrants: set[int] | list[int] | int,
    numbers: np.ndarray,
) -> FragmentationResult:
    """Split off each monovalent migrant (H or halogen) as its own
    doublet fragment, plus a scaffold with mult derived from electron
    parity.

    Accepts a single atom index (legacy P5-with-one-H call sites) or
    an arbitrary iterable of atom indices. Each migrant gets a role of
    ``migrating_<elem>`` (e.g. ``migrating_H``, ``migrating_Cl``).
    """
    if isinstance(migrants, int):
        m_atoms = [int(migrants)]
    else:
        m_atoms = sorted(int(a) for a in migrants)
    m_set = set(m_atoms)
    rest_atoms = np.array(
        [a for a in range(n_atoms) if a not in m_set], dtype=int
    )

    scaffold_electrons = int(sum(int(numbers[a]) for a in rest_atoms))
    scaffold_mult = 1 if scaffold_electrons % 2 == 0 else 2

    fragments: list[FragmentSpec] = [
        FragmentSpec(
            atom_indices=rest_atoms,
            role="scaffold",
            multiplicity=scaffold_mult,
            cap_attachment=None,
        )
    ]
    role_summary: list[str] = []
    for a in m_atoms:
        z = int(numbers[a])
        elem = _ELEM_LABEL.get(z, f"Z{z}")
        role = f"migrating_{elem}"
        role_summary.append(f"{elem}{a}")
        fragments.append(
            FragmentSpec(
                atom_indices=np.array([a], dtype=int),
                role=role,
                multiplicity=2,
                cap_attachment=None,
            )
        )

    return FragmentationResult(
        pattern="P5_HSHIFT",
        fragments=fragments,
        cap_h_positions=None,
        confidence=0.95,
        notes=(
            f"{len(m_atoms)} migrating monovalent atom(s): "
            f"{', '.join(role_summary)} each as own doublet fragment; "
            f"scaffold = {len(rest_atoms)} atoms (mult={scaffold_mult})"
        ),
    )


# --------------------------------------------------------------------------- #
# P4 — product-side dissociation
# --------------------------------------------------------------------------- #


_SMALL_MOL_MERGE_THRESHOLD = 1.5  # Å — covers H–H (0.74), H–F (0.92),
                                  # H–Cl (1.27), H–Br (1.41), F–F (1.42)


def _merge_small_fragments_by_distance(
    fragments: list[FragmentSpec],
    positions_P: np.ndarray,
    numbers: np.ndarray,
    threshold: float = _SMALL_MOL_MERGE_THRESHOLD,
) -> tuple[list[FragmentSpec], list[dict]]:
    """Patch v4 §5-A.X — recover small-molecule extrusions that the
    distance-based bond detector missed at the product geometry.

    Two single-atom product fragments sitting within ``threshold`` Å of
    each other in the product frame are merged into one closed-shell
    fragment (typical case: an H₂, HF, HCl, or HBr extruded together
    from the scaffold but whose interatomic distance just exceeded the
    perception cutoff in product geometry).
    """
    singletons = [
        (idx, f) for idx, f in enumerate(fragments)
        if len(f.atom_indices) == 1
    ]
    if len(singletons) < 2:
        return fragments, []

    merges: list[tuple[int, int]] = []
    used: set[int] = set()
    for ii in range(len(singletons)):
        i, f1 = singletons[ii]
        if i in used:
            continue
        a1 = int(f1.atom_indices[0])
        best: tuple[float, int] | None = None
        for jj in range(ii + 1, len(singletons)):
            j, f2 = singletons[jj]
            if j in used:
                continue
            a2 = int(f2.atom_indices[0])
            d = float(np.linalg.norm(positions_P[a1] - positions_P[a2]))
            if d < threshold and (best is None or d < best[0]):
                best = (d, j)
        if best is not None:
            merges.append((i, best[1]))
            used.add(i)
            used.add(best[1])

    if not merges:
        return fragments, []

    merge_log: list[dict] = []
    merged_indices: set[int] = set()
    new_fragments: list[FragmentSpec] = []
    for i, j in merges:
        f1, f2 = fragments[i], fragments[j]
        a1 = int(f1.atom_indices[0])
        a2 = int(f2.atom_indices[0])
        merged_atoms = np.array(sorted({a1, a2}), dtype=int)
        n_electrons = int(numbers[a1]) + int(numbers[a2])
        merged_mult = 1 if n_electrons % 2 == 0 else 2
        z1, z2 = int(numbers[a1]), int(numbers[a2])
        new_fragments.append(
            FragmentSpec(
                atom_indices=merged_atoms,
                role=f"small_molecule_Z{z1}Z{z2}",
                multiplicity=merged_mult,
                cap_attachment=None,
            )
        )
        merged_indices.update((i, j))
        merge_log.append({
            "merged_atoms": [a1, a2],
            "atomic_numbers": [z1, z2],
            "distance_P": float(np.linalg.norm(positions_P[a1] - positions_P[a2])),
        })

    for i, f in enumerate(fragments):
        if i not in merged_indices:
            new_fragments.append(f)

    # Largest fragment gets reactive_A, then reactive_B, etc.
    new_fragments.sort(key=lambda f: -len(f.atom_indices))
    for i, f in enumerate(new_fragments):
        if i < 26:
            f.role = f"reactive_{chr(ord('A') + i)}"
        else:
            f.role = f"frag_{i}"
    return new_fragments, merge_log


def fragment_P4(
    n_atoms: int,
    numbers: np.ndarray,
    product_components: list[set[int]],
    positions_P: np.ndarray | None = None,
) -> FragmentationResult:
    """Use each connected component of the *product* bond graph as a
    fragment (spec patch v1 §5-A.X).

    Closed-shell singlet vs doublet is assigned by electron-count parity
    (charge=0 is assumed — true for all Halo8 reactions per CLAUDE.md).
    No H caps needed: each fragment is already a closed product molecule.

    Patch v4: when ``positions_P`` is provided, single-atom fragments
    within :data:`_SMALL_MOL_MERGE_THRESHOLD` of each other are merged
    into one closed-shell small molecule (recovers H₂/HF/HCl extrusions
    whose H–H bond the distance-based perceiver missed).
    """
    components = sorted(product_components, key=len, reverse=True)
    fragments: list[FragmentSpec] = []
    for i, comp in enumerate(components):
        atoms = np.array(sorted(comp), dtype=int)
        n_electrons = int(sum(int(numbers[a]) for a in atoms))
        mult = 1 if n_electrons % 2 == 0 else 2
        role = (
            f"reactive_{chr(ord('A') + i)}" if i < 26 else f"frag_{i}"
        )
        fragments.append(
            FragmentSpec(
                atom_indices=atoms,
                role=role,
                multiplicity=mult,
                cap_attachment=None,
            )
        )

    merge_log: list[dict] = []
    if positions_P is not None:
        fragments, merge_log = _merge_small_fragments_by_distance(
            fragments, positions_P, numbers
        )

    notes = f"{len(fragments)} product molecule(s) used as fragments"
    if merge_log:
        notes += f"; merged {len(merge_log)} small-molecule pair(s): {merge_log}"
    return FragmentationResult(
        pattern="P4_DISSOC",
        fragments=fragments,
        cap_h_positions=None,
        confidence=0.95,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Hierarchical sub-fragmentation
# --------------------------------------------------------------------------- #


def fragment_hierarchical(
    n_atoms: int,
    numbers: np.ndarray,
    bonds_R: BondSet,
    bonds_broken: BondSet,
    bonds_formed: BondSet,
    migrating_atoms: list[int] | set[int] | None,
    positions_R: np.ndarray,
    pattern_label: str,
) -> FragmentationResult:
    """Asynchronous-concerted reactions: split into every chemically
    distinct piece implied by the TS bond-change set.

    Strategy: each migrating atom (H or halogen, R-degree=1) is its own
    singleton doublet fragment. The remaining R-skeleton is then cut at
    every ``bonds_broken`` edge whose endpoints are both non-migrants,
    and each resulting connected component becomes a separate fragment.
    Multiplicities are assigned by electron-count parity (charge = 0).

    This yields 3–5 fragments for reactions where two leaving groups and
    a sigma-skeletal rearrangement coincide at one TS — the previous
    "lump into two products" rule collapses too much chemistry.
    """
    m_set = {int(a) for a in (migrating_atoms or [])}
    rest = [a for a in range(n_atoms) if a not in m_set]

    # R skeleton restricted to non-migrant atoms, with bonds_broken removed.
    bb_canon = {tuple(sorted(b)) for b in bonds_broken}
    G = nx.Graph()
    G.add_nodes_from(rest)
    for i, j in bonds_R:
        if i in m_set or j in m_set:
            continue
        if tuple(sorted((int(i), int(j)))) in bb_canon:
            continue
        G.add_edge(int(i), int(j))

    components = sorted(
        (set(c) for c in nx.connected_components(G)),
        key=len,
        reverse=True,
    )

    fragments: list[FragmentSpec] = []
    for k, comp in enumerate(components):
        n_e = int(sum(int(numbers[a]) for a in comp))
        mult = 1 if n_e % 2 == 0 else 2
        role = (
            f"reactive_{chr(ord('A') + k)}" if k < 26 else f"frag_{k}"
        )
        fragments.append(
            FragmentSpec(
                atom_indices=np.array(sorted(comp), dtype=int),
                role=role,
                multiplicity=mult,
                cap_attachment=None,
            )
        )

    role_summary: list[str] = []
    for a in sorted(m_set):
        z = int(numbers[a])
        elem = _ELEM_LABEL.get(z, f"Z{z}")
        role_summary.append(f"{elem}{a}")
        fragments.append(
            FragmentSpec(
                atom_indices=np.array([a], dtype=int),
                role=f"migrating_{elem}",
                multiplicity=2,
                cap_attachment=None,
            )
        )

    base_conf = 0.75 if len(components) <= 3 else 0.6
    if len(components) == 0:
        base_conf = 0.5

    notes = (
        f"hierarchical sub-fragmentation: {len(components)} scaffold "
        f"component(s) + {len(m_set)} migrant singleton(s)"
        + (f" [{', '.join(role_summary)}]" if role_summary else "")
        + f"; {len(bonds_broken)} broken, {len(bonds_formed)} formed"
    )
    return FragmentationResult(
        pattern=pattern_label,
        fragments=fragments,
        cap_h_positions=None,
        confidence=base_conf,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# P3
# --------------------------------------------------------------------------- #

def _attachments_between(
    bonds_R: BondSet,
    comp: set[int],
    other: set[int],
) -> list[tuple[int, int]]:
    """Return ``[(atom_in_comp, atom_in_other), ...]`` for every R-bond that
    bridges ``comp`` and ``other``."""
    out: list[tuple[int, int]] = []
    for i, j in bonds_R:
        if i in comp and j in other:
            out.append((i, j))
        elif j in comp and i in other:
            out.append((j, i))
    return out


def fragment_P3(
    n_atoms: int,
    bonds_R: BondSet,
    bonds_broken: BondSet,
    bonds_formed: BondSet,
    reactive_components: list[set[int]],
    tether_atoms: set[int],
    positions_R: np.ndarray,
    numbers: np.ndarray | None = None,
) -> FragmentationResult | None:
    """Reactive component A + reactive component B + tether, with H caps
    at every severed bond. Genuine intramolecular π-cycloaddition pattern.

    σ-only broken bonds (e.g. Cope/Claisen-type [3,3]-sigmatropic) are
    routed to P2_CLOSED with P2B subtype by the classifier *before*
    reaching this fragmenter — P3 here is reserved for π-cycloaddition
    where ``bonds_broken`` contains at least one π-containing bond.

    Partitioning (v0 spec + connectivity sanity):
      * seed_A / seed_B = the core_A / core_B atoms (R-induced subgraph).
      * Cut R-skeleton at every seed-A↔seed-T and seed-B↔seed-T bond.
      * Absorb each resulting connected component into whichever seed it
        overlaps with (pendant H's and chains follow their R-bonded
        neighbour). Ring-topology cases that fail to disconnect bail to
        P2 (return None).
      * Cap H at every severed R-bond (1.09 Å along the f→t direction).
      * All fragments closed-shell singlet (mult = 1, confidence 0.9).
    """
    seed_A, seed_B = reactive_components[0], reactive_components[1]
    # Additional core components (rare) → nearest seed by spatial proximity.
    extras_core: set[int] = set()
    for c in reactive_components[2:]:
        extras_core |= c
    if extras_core:
        a_pts = positions_R[list(seed_A)]
        b_pts = positions_R[list(seed_B)]
        ex_pts = positions_R[list(extras_core)]
        d_A = np.min(np.linalg.norm(ex_pts[:, None, :] - a_pts[None, :, :], axis=-1))
        d_B = np.min(np.linalg.norm(ex_pts[:, None, :] - b_pts[None, :, :], axis=-1))
        if d_A <= d_B:
            seed_A |= extras_core
        else:
            seed_B |= extras_core

    seed_T = set(tether_atoms)

    # Severed R-bonds: every edge that crosses seed_A↔seed_T or seed_B↔seed_T.
    att_A = _attachments_between(bonds_R, seed_A, seed_T)
    att_B = _attachments_between(bonds_R, seed_B, seed_T)
    severed: BondSet = set()
    for f, t in att_A + att_B:
        severed.add((f, t) if f < t else (t, f))

    # Cut the R skeleton at those bonds and absorb every component into
    # whichever seed it touches.
    G = _graph_without(bonds_R, severed, n_atoms)

    # Connectivity sanity check — the cut must separate seed_A, seed_B,
    # and seed_T into different connected components of G.
    comp_A: set[int] = set(seed_A)
    comp_B: set[int] = set(seed_B)
    comp_T: set[int] = set(seed_T)
    unassigned: list[set[int]] = []
    for cc in nx.connected_components(G):
        cc = set(cc)
        flags = (bool(cc & seed_A), bool(cc & seed_B), bool(cc & seed_T))
        if sum(flags) > 1:
            return None  # residual ring → fall back to P2
        if flags[0]:
            comp_A |= cc
        elif flags[1]:
            comp_B |= cc
        elif flags[2]:
            comp_T |= cc
        else:
            unassigned.append(cc)

    # Fallback for orphan components: assign by spatial proximity.
    for cc in unassigned:
        ex_pts = positions_R[list(cc)]
        dists = []
        for seed_label, comp in (("A", comp_A), ("B", comp_B), ("T", comp_T)):
            if comp:
                pts = positions_R[list(comp)]
                d = float(np.min(np.linalg.norm(ex_pts[:, None, :] - pts[None, :, :], axis=-1)))
                dists.append((d, seed_label))
        dists.sort()
        target = dists[0][1] if dists else "T"
        if target == "A":
            comp_A |= cc
        elif target == "B":
            comp_B |= cc
        else:
            comp_T |= cc

    # Cap-H attachment lists.
    att_A = _attachments_between(bonds_R, comp_A, comp_T)
    att_B = _attachments_between(bonds_R, comp_B, comp_T)
    att_T = [(t, c) for (c, t) in att_A + att_B]

    caps_A_pos, cap_A_meta = _add_cap_hydrogens(positions_R, sorted(comp_A), att_A)
    caps_B_pos, cap_B_meta = _add_cap_hydrogens(positions_R, sorted(comp_B), att_B)
    caps_T_pos, cap_T_meta = _add_cap_hydrogens(positions_R, sorted(comp_T), att_T)

    frag_A = FragmentSpec(
        atom_indices=np.array(sorted(comp_A), dtype=int),
        role="reactive_A",
        multiplicity=1,
        cap_attachment=cap_A_meta,
    )
    frag_B = FragmentSpec(
        atom_indices=np.array(sorted(comp_B), dtype=int),
        role="reactive_B",
        multiplicity=1,
        cap_attachment=cap_B_meta,
    )
    frag_T = FragmentSpec(
        atom_indices=np.array(sorted(comp_T), dtype=int),
        role="tether",
        multiplicity=1,
        cap_attachment=cap_T_meta,
    )

    parts = [arr for arr in (caps_A_pos, caps_B_pos, caps_T_pos) if len(arr) > 0]
    all_caps = np.vstack(parts) if parts else np.empty((0, 3))

    return FragmentationResult(
        pattern="P3_TETHER",
        fragments=[frag_A, frag_B, frag_T],
        cap_h_positions=all_caps,
        confidence=0.9,
        notes=f"reactive A={len(comp_A)} atoms, B={len(comp_B)} atoms, "
              f"tether={len(tether_atoms)} atoms; "
              f"caps: A={len(cap_A_meta)}, B={len(cap_B_meta)}, T={len(cap_T_meta)}",
    )
