"""Stage 3.7 — Semi-auto fragment definition for Case B reactions.

Algorithm:
1. From bond-change records, identify the principal reactive bond — the
   broken/formed pair whose RâTS distance changes the most (we use the
   broken bond from R's graph; if no broken bond, fall back to the formed
   bond present in TS but absent in R).
2. Remove that bond from the R graph and check that the remaining graph
   has exactly two components. Otherwise reclassify to Case C.
3. Add H caps at the cut atoms (length 1.09 Ã along the original bond
   axis on each side).
4. Score the cut: confidence based on distortion magnitude, fragment
   balance, and SMILES validity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bonds import connected_components, detect_bonds
from .halo8_io import _formula_from_numbers
from .logging_setup import get_logger, log_header
from .paths import (
    BOND_CHANGES_JSON,
    CASE_JSON,
    FRAGMENTS_AUTO_JSON,
    TMP_DIR,
    ensure_dirs,
)
from .stage_3_6_frag_caseA import (
    _fragment_charge_multiplicity,
    _smiles_from_fragment,
    _z_to_symbol,
    _bond_count_within,
    _load_npz,
)

H_CAP_BOND_LEN = 1.09  # Ã  (typical CâH length)


def _bond_key(b: list[int] | tuple[int, int]) -> tuple[int, int]:
    a, b_ = int(b[0]), int(b[1])
    return (a, b_) if a < b_ else (b_, a)


def _principal_bond(
    bond_changes: dict,
    pos_R: np.ndarray,
    pos_TS: np.ndarray,
) -> tuple[tuple[int, int], float, str]:
    """Return ((i, j), |dR-dTS|, kind) for the bond with largest distortion."""
    candidates: list[tuple[tuple[int, int], float, str]] = []
    for kind in ("bonds_broken", "bonds_formed"):
        for b in bond_changes.get(kind, []):
            i, j = _bond_key(b)
            d_R = float(np.linalg.norm(pos_R[i] - pos_R[j]))
            d_TS = float(np.linalg.norm(pos_TS[i] - pos_TS[j]))
            candidates.append(((i, j), abs(d_R - d_TS), kind.split("_")[1]))
    if not candidates:
        raise ValueError("no reactive bonds available")
    candidates.sort(key=lambda x: -x[1])
    return candidates[0]


def _all_reactive_bonds(
    bond_changes: dict,
    pos_R: np.ndarray,
    pos_TS: np.ndarray,
) -> list[tuple[tuple[int, int], float, str]]:
    """Same as _principal_bond but returns the full sorted list (largest distortion first)."""
    out: list[tuple[tuple[int, int], float, str]] = []
    for kind in ("bonds_broken", "bonds_formed"):
        for b in bond_changes.get(kind, []):
            i, j = _bond_key(b)
            d_R = float(np.linalg.norm(pos_R[i] - pos_R[j]))
            d_TS = float(np.linalg.norm(pos_TS[i] - pos_TS[j]))
            out.append(((i, j), abs(d_R - d_TS), kind.split("_")[1]))
    out.sort(key=lambda x: -x[1])
    return out


def _add_h_cap(positions: np.ndarray, anchor_idx: int, partner_idx: int) -> np.ndarray:
    """Place an H along the (anchor -> partner) direction at H_CAP_BOND_LEN."""
    direction = positions[partner_idx] - positions[anchor_idx]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        # Degenerate; place H along +x.
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / norm
    return positions[anchor_idx] + direction * H_CAP_BOND_LEN


def _smiles_with_caps(
    numbers: np.ndarray,
    positions: np.ndarray,
    atom_idx: list[int],
    cap_anchor_pairs: list[tuple[int, int]],
    fallback_bonds: set[tuple[int, int]] | None = None,
) -> str | None:
    """Build SMILES for fragment with H caps appended.

    Tries RDKit XYZ-based bond perception first; if that fails, falls back to
    a single-bond SMILES from the supplied connectivity.
    """
    sub_z = list(numbers[atom_idx])
    sub_pos = list(positions[atom_idx])
    for anchor_idx, partner_idx in cap_anchor_pairs:
        h_pos = _add_h_cap(positions, anchor_idx, partner_idx)
        sub_z.append(1)
        sub_pos.append(h_pos)
    sub_z = np.asarray(sub_z, dtype=int)
    sub_pos = np.asarray(sub_pos)
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        xyz = [str(len(sub_z)), ""]
        for z, p in zip(sub_z, sub_pos):
            xyz.append(f"{_z_to_symbol(int(z))} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
        mol = Chem.MolFromXYZBlock("\n".join(xyz))
        if mol is not None:
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=0)
                s = Chem.MolToSmiles(mol)
                if s:
                    return s
            except Exception:
                pass
    except ImportError:
        pass
    # Connectivity-based fallback
    if fallback_bonds is not None:
        from .stage_3_6_frag_caseA import _smiles_from_graph
        return _smiles_from_graph(numbers, fallback_bonds, atom_idx)
    return None


def _confidence(distortion: float, smiles_ok: bool, balance: float, frag_min_size: int) -> float:
    score = 0.0
    # Distortion: 0.5+ Ã distortion is a clean cut, < 0.2 Ã is a borderline rearrangement.
    score += min(1.0, max(0.0, distortion / 0.6)) * 0.4
    score += 0.3 if smiles_ok else 0.0
    # Balance: 1 = perfectly equal sized, 0 = one fragment is a single atom.
    score += balance * 0.2
    score += 0.1 if frag_min_size >= 2 else 0.0
    return float(round(score, 3))


def run(
    case_json: Path | None = None,
    bond_changes_json: Path | None = None,
    output_json: Path | None = None,
) -> dict:
    ensure_dirs()
    log = get_logger("phase1.stage3_7")
    log_header(log, "3.7 Case B semi-auto fragments")
    if case_json is None:
        case_json = CASE_JSON
    if bond_changes_json is None:
        bond_changes_json = BOND_CHANGES_JSON
    if output_json is None:
        output_json = FRAGMENTS_AUTO_JSON

    cases = json.loads(case_json.read_text())
    bond_data = json.loads(bond_changes_json.read_text())

    case_b_ids = [rid for rid, info in cases.items() if info["case"] == "B"]
    case_c_ids = [rid for rid, info in cases.items() if info["case"] == "C"]
    log.info("Case B reactions: %d", len(case_b_ids))
    log.info("Case C reactions (will also try auto-split): %d", len(case_c_ids))
    case_b_ids = case_b_ids + case_c_ids  # try the same algorithm for Case C too

    out: dict[str, dict] = {}
    if output_json.exists():
        out = json.loads(output_json.read_text())

    reclass_to_C: list[str] = []
    for rid in case_b_ids:
        try:
            bundle = _load_npz(rid)
        except FileNotFoundError:
            log.warning("missing npz for %s", rid)
            continue
        numbers = np.asarray(bundle["numbers"], dtype=int)
        coords = np.asarray(bundle["coords_5pts"])
        pos_R = coords[0]
        pos_TS = coords[-1] if len(coords) < 5 else coords[-1]
        # NB: ts is at zeta=1.0 which is index 4 in our 5-pt grid.
        pos_TS = coords[4]

        bc = bond_data.get(rid)
        if not bc:
            log.warning("no bond_data for %s, skip", rid)
            continue
        try:
            candidates = _all_reactive_bonds(bc, pos_R, pos_TS)
        except ValueError as e:
            log.warning("%s: %s", rid, e)
            continue
        if not candidates:
            reclass_to_C.append(rid)
            continue

        bonds_R = detect_bonds(numbers, pos_R)
        bonds_TS = detect_bonds(numbers, pos_TS)
        # Try each reactive bond in order of decreasing distortion; pick the
        # first one whose removal from the appropriate graph yields exactly
        # two connected components.
        principal = None
        distortion = 0.0
        kind = ""
        cut_graph: set[tuple[int, int]] = set()
        anchor_pos: np.ndarray | None = None
        comps: list[list[int]] = []
        for cand_bond, cand_dist, cand_kind in candidates:
            graph = (bonds_R if cand_kind == "broken" else bonds_TS) - {cand_bond}
            cand_comps = connected_components(len(numbers), graph)
            if len(cand_comps) == 2:
                principal = cand_bond
                distortion = cand_dist
                kind = cand_kind
                cut_graph = graph
                anchor_pos = pos_R if cand_kind == "broken" else pos_TS
                comps = cand_comps
                break
        salvaged = False
        if principal is None:
            # Fallback: remove ALL broken bonds at once. If that produces 2+
            # components, take the two largest as fragments and merge any
            # smaller components into fragment 2 (the leaving group side).
            broken_pairs = {_bond_key(b) for b in bc.get("bonds_broken", [])}
            if broken_pairs:
                graph = bonds_R - broken_pairs
                cand_comps = connected_components(len(numbers), graph)
                if len(cand_comps) >= 2:
                    comps = cand_comps
                    if len(comps) > 2:
                        merged_minor = sorted([a for c in comps[1:] for a in c])
                        comps = [comps[0], merged_minor]
                    # Pick the broken bond with the largest distortion to
                    # represent the principal cut for confidence + H-cap.
                    candidates_broken = [
                        c for c in candidates if c[2] == "broken" and c[0] in broken_pairs
                    ]
                    if not candidates_broken:
                        candidates_broken = candidates
                    principal = candidates_broken[0][0]
                    distortion = candidates_broken[0][1]
                    kind = "broken"
                    cut_graph = graph
                    anchor_pos = pos_R
                    salvaged = True
                    log.info(
                        "%s: salvaged via remove-all-broken-bonds (%d -> 2 components)",
                        rid,
                        len(cand_comps),
                    )
        if principal is None:
            # No reactive bond cleanly partitions the molecule -> Case C.
            log.info("%s: no reactive bond cut produces 2 components, reclassify to C", rid)
            reclass_to_C.append(rid)
            continue
        frag1, frag2 = comps  # already sorted by size (largest first)

        # H caps: each fragment needs an H where the principal bond used to be.
        i, j = principal
        # Determine which fragment each end belongs to.
        if i in frag1:
            anchor1, partner1 = i, j
        else:
            anchor1, partner1 = i, j  # placeholder; recompute below
        if i in set(frag1):
            anchor1, partner1 = i, j
            anchor2, partner2 = j, i
        else:
            anchor1, partner1 = j, i
            anchor2, partner2 = i, j

        b_within_1 = _bond_count_within(cut_graph, set(frag1))
        b_within_2 = _bond_count_within(cut_graph, set(frag2))
        c1, m1 = _fragment_charge_multiplicity(numbers, frag1, b_within_1)
        c2, m2 = _fragment_charge_multiplicity(numbers, frag2, b_within_2)

        smiles1 = _smiles_with_caps(numbers, anchor_pos, frag1, [(anchor1, partner1)], fallback_bonds=cut_graph)
        smiles2 = _smiles_with_caps(numbers, anchor_pos, frag2, [(anchor2, partner2)], fallback_bonds=cut_graph)
        smiles_ok = smiles1 is not None and smiles2 is not None

        n1, n2 = len(frag1), len(frag2)
        balance = min(n1, n2) / max(n1, n2)
        conf = _confidence(distortion, smiles_ok, balance, min(n1, n2))

        h_caps = []
        for anchor, partner in [(anchor1, partner1), (anchor2, partner2)]:
            h_pos = _add_h_cap(anchor_pos, anchor, partner)
            h_caps.append(
                {
                    "attached_to_atom": int(anchor),
                    "h_position": h_pos.tolist(),
                    "from_broken_bond": [int(principal[0]), int(principal[1])],
                }
            )

        review_status = (
            "auto_accepted"
            if conf >= 0.65
            else ("needs_review" if conf >= 0.3 else "low_confidence")
        )
        # Preserve original case label (B or C) — this lets downstream stages
        # know that a successful auto-split was found even for what the
        # n_bond_changes >= 4 heuristic flagged as "concerted".
        original_case = cases.get(rid, {}).get("case", "B")
        if salvaged:
            review_status = "needs_review"  # salvage-derived cuts are flaggable
        record = {
            "case": original_case,
            "salvage_used": salvaged,
            "method": "auto_principal_bond_cut",
            "frag1_atoms": list(frag1),
            "frag2_atoms": list(frag2),
            "h_caps": h_caps,
            "frag1_charge": c1,
            "frag2_charge": c2,
            "frag1_multiplicity": m1,
            "frag2_multiplicity": m2,
            "frag1_smiles": smiles1,
            "frag2_smiles": smiles2,
            "frag1_formula": _formula_from_numbers(numbers[frag1]),
            "frag2_formula": _formula_from_numbers(numbers[frag2]),
            "principal_bond": [int(principal[0]), int(principal[1])],
            "principal_bond_kind": kind,
            "principal_bond_distortion_angstroms": round(distortion, 4),
            "auto_confidence": conf,
            "review_status": review_status,
            "rationale": f"Case B: cut at most-distorted {kind} bond ({principal[0]}, {principal[1]})",
        }
        if conf < 0.3:
            log.info("%s: very low confidence %.2f, demoting to Case C", rid, conf)
            reclass_to_C.append(rid)
            continue
        out[rid] = record

    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)
    log.info(
        "Wrote %s; Case B processed=%d, reclassified to C=%d",
        output_json,
        len(case_b_ids) - len(reclass_to_C),
        len(reclass_to_C),
    )

    # Update case_classification.json with the demotions.
    if reclass_to_C:
        for rid in reclass_to_C:
            cases[rid]["case"] = "C"
            cases[rid]["rationale"] = cases[rid].get("rationale", "") + " (demoted from B by Stage 3.7)"
        case_json.write_text(json.dumps(cases, indent=2))
        log.info("Updated %s with %d demotions", case_json, len(reclass_to_C))
    return {"processed": len(case_b_ids) - len(reclass_to_C), "reclassified": reclass_to_C}
