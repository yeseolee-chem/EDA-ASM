"""Stage 5-A v2 classifier — clean rewrite (supersedes patches v1–v4).

Decision tree (per ``stage_5A_v2_clean.md``):

    Step 1: R bond graph has ≥ 2 components?          → P0_BIMOL
    Step 2: P bond graph has ≥ 2 components?          → P4_DISSOC
    Step 3: any H atom in (broken ∩ formed)?          → P5_HSHIFT
    Step 4: existing P1_OPEN / P2_CLOSED / P3_TETHER  (v0 logic)

The classifier never emits ``P_STRAIN_ONLY``: any H that swaps heavy-atom
partners is admitted to P5 regardless of accompanying scaffold changes
or |broken| vs |formed| balance. Quality issues (multireference, ring
rearrangements that defeat fragment cuts) are flagged downstream by the
fragmenter via ``confidence = 0`` rather than pre-emptively excluded.

The v0 spec proposes RDKit's ``DetermineBonds`` for SMILES + bond
perception. That fails on stretched bonds, so we instead reuse the
distance-based bond detector
(``stage0_fragmentation.bond_detection.detect_bonds_strict``) at R and
at P and derive the bond-change sets directly.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from stage0_fragmentation.bond_detection import detect_bonds_strict


BondSet = set[tuple[int, int]]


def _canon(bond: tuple[int, int]) -> tuple[int, int]:
    i, j = int(bond[0]), int(bond[1])
    return (i, j) if i < j else (j, i)


def _to_canon_set(bonds) -> BondSet:
    """Normalise a list/array/set of bonds to {(i,j), i<j}."""
    out: BondSet = set()
    for b in bonds:
        try:
            i, j = int(b[0]), int(b[1])
        except (TypeError, IndexError):
            continue
        if i == j:
            continue
        out.add((i, j) if i < j else (j, i))
    return out


def detect_bond_changes(
    numbers: np.ndarray,
    positions_R: np.ndarray,
    positions_P: np.ndarray,
) -> dict[str, Any]:
    """Compute (bonds_R, bonds_P, bonds_broken, bonds_formed, core_atoms).

    Bonds are returned as canonical sets of ``(i, j)`` with ``i < j``.
    """
    bonds_R = _to_canon_set(detect_bonds_strict(numbers, positions_R))
    bonds_P = _to_canon_set(detect_bonds_strict(numbers, positions_P))
    bonds_broken = bonds_R - bonds_P
    bonds_formed = bonds_P - bonds_R
    core_atoms: set[int] = set()
    for i, j in bonds_broken | bonds_formed:
        core_atoms.add(i)
        core_atoms.add(j)
    return {
        "bonds_R": bonds_R,
        "bonds_P": bonds_P,
        "bonds_broken": bonds_broken,
        "bonds_formed": bonds_formed,
        "core_atoms": core_atoms,
    }


def _adjacency(n: int, bonds: BondSet) -> np.ndarray:
    adj = np.zeros((n, n), dtype=bool)
    for i, j in bonds:
        adj[i, j] = adj[j, i] = True
    return adj


def reactant_components(n: int, bonds_R: BondSet) -> list[set[int]]:
    """Connected components of a bond graph over ``n`` atoms."""
    adj = _adjacency(n, bonds_R)
    n_comp, labels = connected_components(csr_matrix(adj), directed=False)
    comps: list[set[int]] = [set() for _ in range(n_comp)]
    for atom_idx, lab in enumerate(labels):
        comps[lab].add(int(atom_idx))
    return comps


HALOGEN_Z: set[int] = {9, 17, 35, 53}

_ELEM_SYMBOL = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    14: "Si", 15: "P", 16: "S", 17: "Cl",
    35: "Br", 53: "I",
}


def _atom_degree_R(atom_idx: int, bonds_R: BondSet) -> int:
    return sum(1 for b in bonds_R if atom_idx in b)


def get_bond_orders_R(
    numbers: np.ndarray,
    positions_R: np.ndarray,
    *,
    charge: int = 0,
) -> tuple[dict[tuple[int, int], int], bool]:
    """Infer R-frame bond orders via RDKit's :func:`DetermineBonds`.

    Returns ``(bond_orders, fallback_used)``. ``bond_orders`` is keyed by
    canonical ``(i, j)`` with ``i < j``; values are 1 (σ-only single), 2
    (double), 3 (triple), or 2 for aromatic (treated as Ï€-containing for
    P2 sub-classification). If RDKit perception fails for any reason,
    ``fallback_used`` is True and the returned dict is empty — callers
    should treat unknown bonds as single (σ-only) by default.

    R geometry is a stable minimum (no stretched bonds), so RDKit usually
    succeeds. The function never raises.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except Exception:
        return {}, True

    natoms = int(len(numbers))
    lines = [str(natoms), "frame"]
    for z, pos in zip(numbers, positions_R):
        sym = _ELEM_SYMBOL.get(int(z), "X")
        lines.append(f"{sym} {float(pos[0]):.6f} {float(pos[1]):.6f} {float(pos[2]):.6f}")
    xyz_block = "\n".join(lines)

    try:
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            return {}, True
        rdDetermineBonds.DetermineBonds(mol, charge=int(charge))
        orders: dict[tuple[int, int], int] = {}
        for bond in mol.GetBonds():
            i = int(bond.GetBeginAtomIdx())
            j = int(bond.GetEndAtomIdx())
            bt = bond.GetBondType()
            if bt == Chem.BondType.SINGLE:
                o = 1
            elif bt == Chem.BondType.DOUBLE:
                o = 2
            elif bt == Chem.BondType.TRIPLE:
                o = 3
            elif bt == Chem.BondType.AROMATIC:
                # Aromatic bonds carry Ï€ density; treat as Ï€-containing
                # for the P2A vs P2B σ-only test (i.e. NOT σ-only).
                o = 2
            else:
                o = 1
            key = (min(i, j), max(i, j))
            orders[key] = o
        return orders, False
    except Exception:
        return {}, True


def classify_p2_subtype(
    bonds_broken: BondSet,
    bond_orders_R: dict[tuple[int, int], int],
) -> tuple[str, str]:
    """Split P2_CLOSED into P2A_PI_CYCLO vs P2B_SIGMA_REARRANGE.

    Rationale (FernÃ¡ndez & Bickelhaupt, *Chem. Soc. Rev.* **2014**,
    DOI: 10.1039/c4cs00055b, §4.2):

    * If every R-bond that breaks is Ï€-containing (order ≥ 2 or
      aromatic), the transformation is a Ï€â†’Ï€ reconfiguration (DielsâAlder,
      [3+2], etc.) — closed-shell singlet fragmentation is appropriate.
    * If at least one σ-only single bond breaks, the transformation is a
      σ-skeletal rearrangement (Cope, Claisen, electrocyclic ring
      opening, …). Bickelhaupt-style treatment uses **doublet radical
      fragments** for the broken σ-bond.

    Returns ``(subtype, rationale)``.
    """
    if not bonds_broken:
        return "P2A_PI_CYCLO", "no_broken_bonds"

    sigma_broken: list[tuple[int, int]] = []
    pi_broken: list[tuple[int, int]] = []
    for bond in bonds_broken:
        i, j = int(bond[0]), int(bond[1])
        key = (min(i, j), max(i, j))
        order = bond_orders_R.get(key, 1)  # default to single when unknown
        if order == 1:
            sigma_broken.append(key)
        else:
            pi_broken.append(key)

    if sigma_broken:
        return (
            "P2B_SIGMA_REARRANGE",
            f"sigma_broken={len(sigma_broken)};pi_broken={len(pi_broken)}",
        )
    return "P2A_PI_CYCLO", f"only_pi_broken={len(pi_broken)}"


def detect_h_migration(
    bonds_broken: BondSet,
    bonds_formed: BondSet,
    numbers: np.ndarray,
) -> set[int]:
    """Return H atoms in (broken ∩ formed). Kept for back-compat callers."""
    atoms_broken = {a for bond in bonds_broken for a in bond}
    atoms_formed = {a for bond in bonds_formed for a in bond}
    candidates = atoms_broken & atoms_formed
    return {a for a in candidates if int(numbers[int(a)]) == 1}


def detect_migrating_atoms(
    bonds_broken: BondSet,
    bonds_formed: BondSet,
    bonds_R: BondSet,
    numbers: np.ndarray,
) -> dict[str, set[int]]:
    """Categorise atoms that appear in both broken and formed bonds.

    Returns a dict with four disjoint subsets:

    * **h_migrating** — H atoms (any R-degree) swapping partners. These
      are the original P5 case.
    * **halogen_migrating** — F/Cl/Br/I atoms with R-degree = 1 whose
      single bond partner changes (e.g. 1,2-Cl shift across a C–C bond).
      Treated the same as H by the P5 fragmenter.
    * **polyvalent_migrating** — polyvalent atoms (deg ≥ 2) where ALL R
      bonds are broken (full partner swap, e.g. an O bridging two
      different carbons). Surfaced as metadata; not split out as its
      own fragment, since group-migration treatment (carrying satellite
      Hs along) is not yet implemented.
    * **rearranging** — polyvalent atoms with only a partial bond swap
      (e.g. a C that loses one neighbour and gains another while
      keeping its other three bonds). Surfaced as metadata.
    """
    atoms_broken = {a for bond in bonds_broken for a in bond}
    atoms_formed = {a for bond in bonds_formed for a in bond}
    candidates = atoms_broken & atoms_formed

    h_migrating: set[int] = set()
    halogen_migrating: set[int] = set()
    polyvalent_migrating: set[int] = set()
    rearranging: set[int] = set()

    for a in candidates:
        z = int(numbers[int(a)])
        deg = _atom_degree_R(int(a), bonds_R)
        broken_count = sum(1 for b in bonds_broken if a in b)
        if z == 1:
            h_migrating.add(int(a))
        elif z in HALOGEN_Z and deg == 1:
            halogen_migrating.add(int(a))
        elif deg >= 2 and broken_count == deg:
            polyvalent_migrating.add(int(a))
        else:
            rearranging.add(int(a))

    return {
        "h_migrating": h_migrating,
        "halogen_migrating": halogen_migrating,
        "polyvalent_migrating": polyvalent_migrating,
        "rearranging": rearranging,
    }


def find_core_components(
    bonds_R: BondSet,
    core_atoms: set[int],
) -> list[set[int]]:
    """Connected components of the induced subgraph on ``core_atoms`` using
    only edges from ``bonds_R``.

    Spec §5-A.2-2: ene-ene-yne-style 6-atom core → 1 component;
    intramolecular Diels-Alder (diene 4 + dienophile 2) → 2 components.
    """
    if not core_atoms:
        return []
    core_list = sorted(core_atoms)
    pos = {a: i for i, a in enumerate(core_list)}
    n = len(core_list)
    sub = np.zeros((n, n), dtype=bool)
    for i, j in bonds_R:
        if i in pos and j in pos:
            sub[pos[i], pos[j]] = sub[pos[j], pos[i]] = True
    n_comp, labels = connected_components(csr_matrix(sub), directed=False)
    comps: list[set[int]] = [set() for _ in range(n_comp)]
    for idx, lab in enumerate(labels):
        comps[lab].add(core_list[idx])
    return comps


def has_tether_path(
    bonds_R: BondSet,
    bonds_broken: BondSet,
    bonds_formed: BondSet,
    comp_A: set[int],
    comp_B: set[int],
) -> tuple[bool, set[int]]:
    """Return ``(has_tether, tether_atoms)``.

    Spec §5-A.2-3: a tether is a path between the two reactive components
    that uses only *non-reactive* bonds.
    """
    reactive = bonds_broken | bonds_formed
    G = nx.Graph()
    G.add_nodes_from({a for bond in bonds_R for a in bond})
    G.add_nodes_from(comp_A | comp_B)
    for i, j in bonds_R:
        if (i, j) in reactive or (j, i) in reactive:
            continue
        G.add_edge(i, j)

    best_path: list[int] | None = None
    for src in comp_A:
        if src not in G:
            continue
        for dst in comp_B:
            if dst not in G:
                continue
            try:
                path = nx.shortest_path(G, source=src, target=dst)
            except nx.NetworkXNoPath:
                continue
            if best_path is None or len(path) < len(best_path):
                best_path = path
    if best_path is None:
        return False, set()
    tether_atoms = set(best_path[1:-1]) - comp_A - comp_B
    return len(tether_atoms) > 0, tether_atoms


def classify_reaction(
    numbers: np.ndarray,
    positions_R: np.ndarray,
    positions_P: np.ndarray,
    bond_change: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return ``(pattern_name, debug_info)``.

    Patterns (v2 — six total, no P_STRAIN_ONLY):
        P0_BIMOL, P1_OPEN, P2_CLOSED, P3_TETHER, P4_DISSOC, P5_HSHIFT.
    """
    n = len(numbers)
    bc = bond_change or detect_bond_changes(numbers, positions_R, positions_P)
    bonds_R: BondSet = bc["bonds_R"]
    bonds_P: BondSet = bc["bonds_P"]
    bonds_broken: BondSet = bc["bonds_broken"]
    bonds_formed: BondSet = bc["bonds_formed"]
    core_atoms: set[int] = bc["core_atoms"]

    # Compute migration analysis up front so the metadata travels with
    # every classification branch (even P0/P4 cases where the pattern
    # itself isn't gated on migration).
    mig = detect_migrating_atoms(bonds_broken, bonds_formed, bonds_R, numbers)
    mig_metadata = {
        "migrating_H_atoms": sorted(mig["h_migrating"]),
        "migrating_halogen_atoms": sorted(mig["halogen_migrating"]),
        "polyvalent_migrating_atoms": sorted(mig["polyvalent_migrating"]),
        "rearranging_atoms": sorted(mig["rearranging"]),
        "n_H_migrating": len(mig["h_migrating"]),
        "n_halogen_migrating": len(mig["halogen_migrating"]),
    }

    # Step 1: bimolecular reactant
    r_components = reactant_components(n, bonds_R)
    if len(r_components) >= 2:
        return "P0_BIMOL", {
            **mig_metadata,
            "n_reactant_components": len(r_components),
            "reactant_components": [sorted(c) for c in r_components],
            **bc,
        }

    n_changes = len(bonds_broken) + len(bonds_formed)
    if n_changes == 0:
        return "P0_BIMOL", {
            **mig_metadata,
            "note": "no bond changes — likely conformer",
            **bc,
        }

    # Step 2: product-side dissociation. The distance-based bond
    # perceiver may miss the H–H bond of an extruded H₂ at product
    # geometry — that case still triggers this branch (≥ 2 product
    # components) and ``fragment_P4`` recovers the H₂ via distance-based
    # merging.
    p_components = reactant_components(n, bonds_P)
    if len(p_components) >= 2:
        return "P4_DISSOC", {
            **mig_metadata,
            "n_product_components": len(p_components),
            "product_components": [sorted(c) for c in p_components],
            **bc,
        }

    # Step 3: any monovalent atom (H or halogen with R-degree=1) that
    # swapped heavy-atom partners → P5_HSHIFT. Polyvalent migrating
    # atoms (e.g. an O bridging two carbons that swaps both bonds) and
    # rearranging atoms (e.g. a C that loses one bond and gains another)
    # are surfaced as metadata but do not gate the routing — handling
    # group migration cleanly requires a fragmenter rewrite.
    monovalent_migrants = mig["h_migrating"] | mig["halogen_migrating"]
    if monovalent_migrants:
        return "P5_HSHIFT", {
            **mig_metadata,
            "migrating_atoms": sorted(monovalent_migrants),
            **bc,
        }

    # Pre-compute R-frame bond orders once for any P2_CLOSED routes that
    # may follow. Only used to decide P2A vs P2B sub-classification.
    bond_orders_R, bond_order_fallback = get_bond_orders_R(numbers, positions_R)
    p2_subtype, p2_rationale = classify_p2_subtype(bonds_broken, bond_orders_R)
    p2_metadata = {
        "p2_subtype": p2_subtype,
        "p2_subtype_rationale": p2_rationale,
        "bond_order_fallback": bool(bond_order_fallback),
    }

    # Step 4: original v0 P1/P2/P3 logic. ``p2_metadata`` is attached to
    # every non-P0/P4/P5 return path so that P3-fallback-to-P2 cases in
    # the pipeline still have the correct sub-classification available.
    core_components = find_core_components(bonds_R, core_atoms)
    if len(core_components) == 1:
        if len(core_atoms) <= 2 and n_changes <= 2:
            return "P1_OPEN", {
                **mig_metadata,
                **p2_metadata,
                "core_atoms": sorted(core_atoms),
                "n_changes": n_changes,
                **bc,
            }
        return "P2_CLOSED", {
            **mig_metadata,
            **p2_metadata,
            "core_atoms": sorted(core_atoms),
            "n_changes": n_changes,
            **bc,
        }

    core_components.sort(key=len, reverse=True)
    comp_A, comp_B = core_components[0], core_components[1]

    # P3 only makes sense when both reactive components are genuine
    # multi-atom π-systems AND both actively lose bonds (otherwise one
    # side is just a new-bond acceptor, not a real reactive partner).
    broken_atoms = {a for b in bonds_broken for a in b}
    if min(len(comp_A), len(comp_B)) < 2:
        return "P2_CLOSED", {
            **mig_metadata,
            **p2_metadata,
            "core_atoms": sorted(core_atoms),
            "n_changes": n_changes,
            "note": (
                "core has multiple components but one is a singleton — "
                "not a genuine P3 cycloaddition; routed to P2"
            ),
            "reactive_components_inspected": [sorted(c) for c in core_components],
            **bc,
        }
    # Bookmark-review patch (T1x_C4H6O_rxn02981 etc.): if only one of
    # the two cores actually shares atoms with bonds_broken, the other
    # side is purely a new-bond acceptor — the reaction is closer to a
    # group migration than to a concerted cycloaddition. Defer to P2.
    if not (broken_atoms & comp_A) or not (broken_atoms & comp_B):
        return "P2_CLOSED", {
            **mig_metadata,
            **p2_metadata,
            "core_atoms": sorted(core_atoms),
            "n_changes": n_changes,
            "note": (
                "one core component has no broken-bond involvement — "
                "this is a group migration, not a concerted cycloaddition; "
                "routed to P2"
            ),
            "reactive_components_inspected": [sorted(c) for c in core_components],
            **bc,
        }

    has_t, tether_atoms = has_tether_path(
        bonds_R, bonds_broken, bonds_formed, comp_A, comp_B
    )
    if has_t:
        # **Bookmark-review round 3 (2026-05-13)**: the user's chemical
        # intuition is that "tether" as a fresh-classifier pattern does
        # not match common cases — the Halo8 reactions that fall into
        # "≥2 cores with a short tether" are predominantly σ-skeletal
        # rearrangements (Cope/Claisen-type, often in bicyclic or
        # strained-ring molecules) where the canonical Bickelhaupt
        # treatment is doublet radical fragments, not the closed-shell
        # singlet + tether scheme of intramolecular π-cycloaddition.
        # Accordingly the fresh classifier no longer emits P3_TETHER;
        # the σ/Ï‚-aware P2A/P2B sub-classification of fragment_P2
        # produces the chemically correct multiplicity (Ï€-only broken
        # → mult=1 closed-shell, σ-involving broken → mult=2 doublet).
        # P3_TETHER survives only in the locked accepted_ground_truth
        # for historical cases the reviewer previously accepted.
        return "P2_CLOSED", {
            **mig_metadata,
            **p2_metadata,
            "core_atoms": sorted(core_atoms),
            "n_changes": n_changes,
            "note": (
                "would-be P3_TETHER (≥2 core components with tether path) — "
                "fresh classifier routes to P2 because the canonical "
                "tether scheme misclassifies σ-skeletal rearrangements; "
                "σ/π sub-classification (P2A/P2B) gives correct "
                "multiplicity downstream"
            ),
            "reactive_components_inspected": [sorted(c) for c in core_components],
            "tether_atoms_proposed": sorted(tether_atoms),
            **bc,
        }
    return "P0_BIMOL", {
        **mig_metadata,
        **p2_metadata,
        "note": "disconnected core, no tether — reactant complex",
        "reactive_components": [sorted(c) for c in core_components],
        **bc,
    }
