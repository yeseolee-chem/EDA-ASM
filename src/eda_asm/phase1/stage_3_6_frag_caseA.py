"""Stage 3.6 — Auto fragment definition for Case A reactions.

Case A reactions already have R as two (or more) disconnected components, so
fragmentation is just connected-component extraction. We write
fragments_auto.json (reading and rewriting if it already contains data
from Stage 3.7).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bonds import connected_components, detect_bonds
from .halo8_io import _formula_from_numbers
from .logging_setup import get_logger, log_header
from .paths import CASE_JSON, FRAGMENTS_AUTO_JSON, TMP_DIR, ensure_dirs

# Halogen anion detection table
ANIONIC_HALOGENS = {9, 17, 35, 53}  # F, Cl, Br, I


def _fragment_charge_multiplicity(numbers: np.ndarray, atom_idx: list[int], bonds_within: int) -> tuple[int, int]:
    """Heuristic: closed-shell neutral by default; lone halogen anion if isolated."""
    if len(atom_idx) == 1 and int(numbers[atom_idx[0]]) in ANIONIC_HALOGENS and bonds_within == 0:
        return -1, 1  # bare halogen anion
    return 0, 1


def _smiles_from_xyz(numbers: np.ndarray, positions: np.ndarray, atom_idx: list[int]) -> str | None:
    """Try RDKit's XYZ → bond perception → SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError:
        return None
    sub_z = numbers[atom_idx]
    sub_pos = positions[atom_idx]
    xyz_lines = [str(len(atom_idx)), ""]
    for z, p in zip(sub_z, sub_pos):
        sym = _z_to_symbol(int(z))
        xyz_lines.append(f"{sym} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    xyz = "\n".join(xyz_lines)
    try:
        mol = Chem.MolFromXYZBlock(xyz)
        if mol is None:
            return None
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _smiles_from_graph(
    numbers: np.ndarray,
    bonds: set[tuple[int, int]],
    atom_idx: list[int],
) -> str | None:
    """Fallback: build an RWMol with single bonds from the supplied graph and serialise.

    This produces a valid (single-bond) SMILES even when XYZ-based bond
    perception fails. Used when geometric cuts give a sane connectivity but
    one of the fragments has unusual valence.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return None
    sub_set = set(atom_idx)
    sub_idx_map = {a: i for i, a in enumerate(atom_idx)}
    rw = Chem.RWMol()
    for a in atom_idx:
        z = int(numbers[a])
        rw.AddAtom(Chem.Atom(z))
    for i, j in bonds:
        if i in sub_set and j in sub_set:
            rw.AddBond(sub_idx_map[i], sub_idx_map[j], Chem.BondType.SINGLE)
    try:
        # Don't sanitize: skeletons are partial / radical-like for some cuts.
        return Chem.MolToSmiles(rw, canonical=True)
    except Exception:
        return None


def _smiles_from_fragment(
    numbers: np.ndarray,
    positions: np.ndarray,
    atom_idx: list[int],
    fallback_bonds: set[tuple[int, int]] | None = None,
) -> str | None:
    s = _smiles_from_xyz(numbers, positions, atom_idx)
    if s:
        return s
    if fallback_bonds is not None:
        return _smiles_from_graph(numbers, fallback_bonds, atom_idx)
    return None


_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}


def _z_to_symbol(z: int) -> str:
    if z in _SYMBOLS:
        return _SYMBOLS[z]
    raise ValueError(f"unhandled atomic number {z}")


def _load_npz(reaction_id: str) -> dict:
    p = TMP_DIR / f"{reaction_id}.npz"
    with np.load(p, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _bond_count_within(bonds: set[tuple[int, int]], atoms: set[int]) -> int:
    return sum(1 for i, j in bonds if i in atoms and j in atoms)


def run(
    case_json: Path | None = None,
    output_json: Path | None = None,
) -> dict:
    ensure_dirs()
    log = get_logger("phase1.stage3_6")
    log_header(log, "3.6 Case A auto fragments")
    if case_json is None:
        case_json = CASE_JSON
    if output_json is None:
        output_json = FRAGMENTS_AUTO_JSON

    cases = json.loads(case_json.read_text())
    case_a_ids = [rid for rid, info in cases.items() if info["case"] == "A"]
    log.info("Case A reactions: %d", len(case_a_ids))

    out: dict[str, dict] = {}
    if output_json.exists():
        out = json.loads(output_json.read_text())

    skipped = []
    for rid in case_a_ids:
        try:
            bundle = _load_npz(rid)
        except FileNotFoundError:
            log.warning("missing npz for %s", rid)
            skipped.append((rid, "missing npz"))
            continue
        numbers = np.asarray(bundle["numbers"], dtype=int)
        coords = np.asarray(bundle["coords_5pts"])  # (5, N, 3)
        pos_R = coords[0]
        bonds_R = detect_bonds(numbers, pos_R)
        comps = connected_components(len(numbers), bonds_R)
        if len(comps) < 2:
            log.warning("%s: Case A but only %d component(s); reclassify", rid, len(comps))
            skipped.append((rid, f"only {len(comps)} component"))
            continue
        # Fragment 1 = largest component, fragment 2 = second largest, merge any extras into fragment 2.
        frag1 = comps[0]
        if len(comps) == 2:
            frag2 = comps[1]
            ternary_note = None
        else:
            # Combine all "minor" components into fragment 2 (the smaller/leaving group).
            frag2 = sorted([a for c in comps[1:] for a in c])
            ternary_note = f"merged {len(comps) - 1} components into fragment 2"
            log.info("%s: ternary case (%d comps); %s", rid, len(comps), ternary_note)

        b_within_1 = _bond_count_within(bonds_R, set(frag1))
        b_within_2 = _bond_count_within(bonds_R, set(frag2))
        c1, m1 = _fragment_charge_multiplicity(numbers, frag1, b_within_1)
        c2, m2 = _fragment_charge_multiplicity(numbers, frag2, b_within_2)

        smiles1 = _smiles_from_fragment(numbers, pos_R, frag1)
        smiles2 = _smiles_from_fragment(numbers, pos_R, frag2)

        record = {
            "case": "A",
            "method": "auto_connected_components",
            "frag1_atoms": list(frag1),
            "frag2_atoms": list(frag2),
            "h_caps": [],
            "frag1_charge": c1,
            "frag2_charge": c2,
            "frag1_multiplicity": m1,
            "frag2_multiplicity": m2,
            "frag1_smiles": smiles1,
            "frag2_smiles": smiles2,
            "frag1_formula": _formula_from_numbers(numbers[frag1]),
            "frag2_formula": _formula_from_numbers(numbers[frag2]),
            "auto_confidence": 1.0,
            "review_status": "auto_accepted",
            "rationale": "Case A: R has disconnected components",
        }
        if ternary_note:
            record["note"] = ternary_note
            record["auto_confidence"] = 0.7
            record["review_status"] = "needs_review"
        out[rid] = record

    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote %s with %d entries (Case A processed=%d, skipped=%d)", output_json, len(out), len(case_a_ids) - len(skipped), len(skipped))
    if skipped:
        for rid, why in skipped:
            log.warning("skipped %s: %s", rid, why)
    return {"processed": len(case_a_ids) - len(skipped), "skipped": skipped}
