"""Validation + SMILES generation for fragment definitions submitted by reviewers.

This intentionally stays self-contained (no Phase 1 imports beyond the bond
detection helpers) so the tool runs even if Phase 1 source layout shifts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eda_asm.phase1.bonds import detect_bonds
from eda_asm.phase1.stage_3_6_frag_caseA import _smiles_from_xyz, _smiles_from_graph

H_CAP_BOND_LEN = 1.09  # Å


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    h_caps: list[dict] = field(default_factory=list)
    frag1_smiles: str | None = None
    frag2_smiles: str | None = None
    frag1_formula: str | None = None
    frag2_formula: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "h_caps": self.h_caps,
            "frag1_smiles": self.frag1_smiles,
            "frag2_smiles": self.frag2_smiles,
            "frag1_formula": self.frag1_formula,
            "frag2_formula": self.frag2_formula,
        }


_ELEMS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P",
          16: "S", 17: "Cl", 35: "Br", 53: "I"}


def _formula(numbers: list[int]) -> str:
    counts: dict[str, int] = {}
    for z in numbers:
        sym = _ELEMS.get(int(z), f"Z{z}")
        counts[sym] = counts.get(sym, 0) + 1
    out = []
    for sym in ("C", "H"):
        if sym in counts:
            n = counts.pop(sym)
            out.append(sym + (str(n) if n > 1 else ""))
    for sym in sorted(counts):
        n = counts[sym]
        out.append(sym + (str(n) if n > 1 else ""))
    return "".join(out)


def _add_h_cap(positions: np.ndarray, anchor_idx: int, partner_idx: int) -> list[float]:
    direction = positions[partner_idx] - positions[anchor_idx]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / norm
    return (positions[anchor_idx] + direction * H_CAP_BOND_LEN).tolist()


def compute_h_caps(
    numbers: list[int],
    positions_R: np.ndarray,
    frag1: list[int],
    frag2: list[int],
) -> list[dict]:
    """Find every R-graph bond that crosses the cut and place an H on each side."""
    bonds = detect_bonds(np.asarray(numbers, dtype=int), positions_R)
    f1, f2 = set(frag1), set(frag2)
    caps: list[dict] = []
    for i, j in bonds:
        if (i in f1 and j in f2) or (i in f2 and j in f1):
            anchor1, partner1 = (i, j) if i in f1 else (j, i)
            anchor2, partner2 = (i, j) if i in f2 else (j, i)
            caps.append({
                "attached_to_atom": int(anchor1),
                "h_position": _add_h_cap(positions_R, anchor1, partner1),
                "from_broken_bond": [int(i), int(j)],
            })
            caps.append({
                "attached_to_atom": int(anchor2),
                "h_position": _add_h_cap(positions_R, anchor2, partner2),
                "from_broken_bond": [int(i), int(j)],
            })
    return caps


def _smiles_for_fragment(
    numbers: list[int],
    positions: np.ndarray,
    atom_idx: list[int],
    cross_bonds_for_caps: list[tuple[int, int]],
) -> tuple[str | None, list[dict]]:
    """Generate a SMILES (XYZ-perceived if possible, else graph-based)."""
    nums = np.asarray(numbers, dtype=int)
    atom_idx_sorted = sorted(atom_idx)
    sub_z = list(nums[atom_idx_sorted])
    sub_pos = list(positions[atom_idx_sorted])
    cap_records: list[dict] = []
    for anchor, partner in cross_bonds_for_caps:
        if anchor not in atom_idx_sorted:
            anchor, partner = partner, anchor
        h_pos = _add_h_cap(positions, anchor, partner)
        sub_z.append(1)
        sub_pos.append(h_pos)
        cap_records.append({
            "attached_to_atom": int(anchor),
            "h_position": list(h_pos),
            "from_broken_bond": [int(anchor), int(partner)],
        })
    sub_z_np = np.asarray(sub_z, dtype=int)
    sub_pos_np = np.asarray(sub_pos)

    s = _smiles_from_xyz(sub_z_np, sub_pos_np, list(range(len(sub_z))))
    if s:
        return s, cap_records

    # Graph-based fallback
    bonds_full = detect_bonds(nums, positions)
    fallback_bonds = {(i, j) for (i, j) in bonds_full if i in set(atom_idx_sorted) and j in set(atom_idx_sorted)}
    s = _smiles_from_graph(nums, fallback_bonds, atom_idx_sorted)
    return s, cap_records


def validate_fragment(
    numbers: list[int],
    positions_R: list[list[float]],
    frag1: list[int],
    frag2: list[int],
) -> ValidationReport:
    """Top-level validator. Returns a ValidationReport with errors / warnings /
    SMILES / H-cap coordinates so the front-end can show everything."""
    rep = ValidationReport(ok=True)
    n = len(numbers)
    s1, s2 = set(frag1), set(frag2)
    if not frag1 or not frag2:
        rep.ok = False
        rep.errors.append("each fragment must contain at least 1 atom")
        return rep
    if s1 & s2:
        rep.ok = False
        rep.errors.append(f"fragments overlap on atoms {sorted(s1 & s2)}")
    union = s1 | s2
    if union != set(range(n)):
        missing = sorted(set(range(n)) - union)
        extra = sorted(union - set(range(n)))
        if missing:
            rep.errors.append(f"missing atoms: {missing}")
        if extra:
            rep.errors.append(f"out-of-range atom indices: {extra}")
        rep.ok = False

    pos = np.asarray(positions_R)
    bonds = detect_bonds(np.asarray(numbers, dtype=int), pos)
    cross = [(i, j) for (i, j) in bonds if (i in s1) != (j in s1)]
    if not cross and rep.ok:
        rep.warnings.append("no R-graph bonds cross the cut; the partition is already disconnected in R")

    # Build cap pairs per fragment (anchor on its own side, partner on the other).
    cross_for_1 = [(i if i in s1 else j, j if i in s1 else i) for (i, j) in cross]
    cross_for_2 = [(i if i in s2 else j, j if i in s2 else i) for (i, j) in cross]

    # SMILES for both fragments
    smi1, caps1 = _smiles_for_fragment(numbers, pos, sorted(frag1), cross_for_1)
    smi2, caps2 = _smiles_for_fragment(numbers, pos, sorted(frag2), cross_for_2)
    rep.frag1_smiles = smi1
    rep.frag2_smiles = smi2
    rep.frag1_formula = _formula([numbers[i] for i in frag1])
    rep.frag2_formula = _formula([numbers[i] for i in frag2])
    if smi1 is None:
        rep.warnings.append("SMILES for fragment 1 could not be generated (RDKit + graph fallback both failed)")
    if smi2 is None:
        rep.warnings.append("SMILES for fragment 2 could not be generated (RDKit + graph fallback both failed)")

    rep.h_caps = compute_h_caps(numbers, pos, frag1, frag2)
    return rep
