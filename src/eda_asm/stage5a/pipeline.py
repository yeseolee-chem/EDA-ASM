"""End-to-end driver: classify one reaction and produce its fragments.

Stage 5-A v2 clean (supersedes patches v1–v4). The classifier outputs
exactly six patterns: P0_BIMOL, P1_OPEN, P2_CLOSED, P3_TETHER,
P4_DISSOC, P5_HSHIFT. When a P1/P2/P3 cut fails to disconnect the
molecule (ring-topology rearrangements), the pipeline still emits a
single whole-molecule fragment at the classifier's chosen pattern but
flags it with ``confidence = 0.0`` so the reviewer can spot and
manually triage these stragglers.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .classify import classify_reaction, detect_bond_changes
from .fragmenters import (
    fragment_P0,
    fragment_P1,
    fragment_P2,
    fragment_P3,
    fragment_P4,
    fragment_P5_hshift,
)
from .types import FragmentSpec, FragmentationResult


def _cut_failed(
    n_atoms: int,
    pattern: str,
    reason: str,
) -> FragmentationResult:
    """Fallback when fragment_P1/P2/P3 can't disconnect the molecule
    (ring rearrangement). Keeps the classifier-chosen pattern but emits
    a single whole-molecule fragment with confidence 0 so the reviewer
    can filter and re-inspect these cases.
    """
    return FragmentationResult(
        pattern=pattern,
        fragments=[
            FragmentSpec(
                atom_indices=np.arange(n_atoms),
                role="whole",
                multiplicity=1,
                cap_attachment=None,
            )
        ],
        cap_h_positions=None,
        confidence=0.0,
        notes=reason,
    )


def process_one_reaction(
    numbers: np.ndarray,
    positions_R: np.ndarray,
    positions_P: np.ndarray,
) -> tuple[FragmentationResult, dict[str, Any]]:
    """Return ``(result, serialisable_debug)`` for one reaction."""
    bc = detect_bond_changes(numbers, positions_R, positions_P)
    pattern, debug = classify_reaction(
        numbers, positions_R, positions_P, bond_change=bc
    )
    n = len(numbers)
    bonds_R = bc["bonds_R"]
    bonds_broken = bc["bonds_broken"]
    bonds_formed = bc["bonds_formed"]

    result: FragmentationResult
    if pattern == "P0_BIMOL":
        result = fragment_P0(n, debug)
    elif pattern == "P4_DISSOC":
        product_components = [set(c) for c in debug["product_components"]]
        result = fragment_P4(
            n, numbers, product_components, positions_P=positions_P
        )
        # Audit metadata for the radical-pair demotion that fragment_P4
        # may apply downstream. If every product fragment is mult ≥ 2,
        # KS-DFT EDA cannot describe the resulting open-shell partition
        # cleanly (multi-reference) — flag a recommended remedy.
        if result and result.fragments and all(
            f.multiplicity >= 2 for f in result.fragments
        ):
            debug["demoted_reason"] = "homolytic_dissociation_multireference"
            debug["recommended_method"] = "CASSCF_or_NEVPT2"
    elif pattern == "P5_HSHIFT":
        # Migrants may be H atoms and/or halogens (F/Cl/Br/I) with R-degree=1
        migrants = (
            debug.get("migrating_atoms")
            or list(set(debug.get("migrating_H_atoms", []))
                    | set(debug.get("migrating_halogen_atoms", [])))
        )
        result = fragment_P5_hshift(n, migrants, numbers)
    elif pattern == "P1_OPEN":
        r = fragment_P1(n, bonds_R, bonds_broken, bonds_formed)
        if r is None:
            r = fragment_P2(n, bonds_R, bonds_broken, bonds_formed, positions_R,
                            subtype=debug.get("p2_subtype", "P2A_PI_CYCLO"))
            if r is not None:
                r.notes = "fell back from P1; " + r.notes
        result = r if r is not None else _cut_failed(
            n,
            "P1_OPEN",
            "P1 cut did not disconnect; P2 fallback also failed "
            "(ring rearrangement)",
        )
    elif pattern == "P2_CLOSED":
        p2_subtype = debug.get("p2_subtype", "P2A_PI_CYCLO")
        r = fragment_P2(n, bonds_R, bonds_broken, bonds_formed, positions_R,
                        subtype=p2_subtype)
        if r is None:
            r = fragment_P1(n, bonds_R, bonds_broken, bonds_formed)
            if r is not None:
                r.notes = "fell back from P2; " + r.notes
        result = r if r is not None else _cut_failed(
            n,
            "P2_CLOSED",
            "neither P2 nor P1 cut disconnected the molecule "
            "(ring rearrangement)",
        )
    elif pattern == "P3_TETHER":
        reactive_components = [set(c) for c in debug["reactive_components"]]
        tether_atoms = set(debug["tether_atoms"])
        r = fragment_P3(
            n,
            bonds_R,
            bonds_broken,
            bonds_formed,
            reactive_components,
            tether_atoms,
            positions_R,
            numbers=numbers,
        )
        if r is None:
            r = fragment_P2(n, bonds_R, bonds_broken, bonds_formed, positions_R,
                            subtype=debug.get("p2_subtype", "P2A_PI_CYCLO"))
            if r is not None:
                r.notes = "fell back from P3 (tether on ring); " + r.notes
        result = r if r is not None else _cut_failed(
            n,
            "P3_TETHER",
            "P3 tether sat on ring; P2 fallback could not disconnect "
            "either (ring rearrangement)",
        )
    else:
        raise ValueError(f"unknown pattern from classifier: {pattern!r}")

    # Serialise bond-change debug as plain lists for JSON.
    serialisable_debug = {
        "pattern_from_classifier": pattern,
        "bonds_R": sorted(tuple(sorted(b)) for b in bc["bonds_R"]),
        "bonds_P": sorted(tuple(sorted(b)) for b in bc["bonds_P"]),
        "bonds_broken": sorted(tuple(sorted(b)) for b in bc["bonds_broken"]),
        "bonds_formed": sorted(tuple(sorted(b)) for b in bc["bonds_formed"]),
        "core_atoms": sorted(bc["core_atoms"]),
        "n_bond_changes": len(bc["bonds_broken"]) + len(bc["bonds_formed"]),
    }
    if "reactive_components" in debug:
        serialisable_debug["reactive_components_at_classify"] = debug["reactive_components"]
    if "tether_atoms" in debug:
        serialisable_debug["tether_atoms_at_classify"] = debug["tether_atoms"]
    if "reactant_components" in debug:
        serialisable_debug["reactant_components"] = debug["reactant_components"]
    if "product_components" in debug:
        serialisable_debug["product_components"] = debug["product_components"]
    if "migrating_H_atoms" in debug:
        serialisable_debug["migrating_H_atoms"] = debug["migrating_H_atoms"]
    if "n_H_migrating" in debug:
        serialisable_debug["n_H_migrating"] = debug["n_H_migrating"]
    if "migrating_halogen_atoms" in debug:
        serialisable_debug["migrating_halogen_atoms"] = debug["migrating_halogen_atoms"]
    if "n_halogen_migrating" in debug:
        serialisable_debug["n_halogen_migrating"] = debug["n_halogen_migrating"]
    if "polyvalent_migrating_atoms" in debug:
        serialisable_debug["polyvalent_migrating_atoms"] = debug["polyvalent_migrating_atoms"]
    if "rearranging_atoms" in debug:
        serialisable_debug["rearranging_atoms"] = debug["rearranging_atoms"]
    if "migrating_atoms" in debug:
        serialisable_debug["migrating_atoms"] = debug["migrating_atoms"]
    if "p2_subtype" in debug:
        serialisable_debug["p2_subtype"] = debug["p2_subtype"]
    if "p2_subtype_rationale" in debug:
        serialisable_debug["p2_subtype_rationale"] = debug["p2_subtype_rationale"]
    if "bond_order_fallback" in debug:
        serialisable_debug["bond_order_fallback"] = debug["bond_order_fallback"]
    if "demoted_reason" in debug:
        serialisable_debug["demoted_reason"] = debug["demoted_reason"]
    if "recommended_method" in debug:
        serialisable_debug["recommended_method"] = debug["recommended_method"]
    if "note" in debug:
        serialisable_debug["classifier_note"] = debug["note"]

    return result, serialisable_debug
