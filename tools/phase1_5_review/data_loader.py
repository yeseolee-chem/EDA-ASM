"""Load Phase 1 outputs (HDF5 + JSON) and expose them to the Flask app.

Cached in-process for speed; the Flask process is single-threaded for our
use case so simple module-level globals are fine.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

# Make eda_asm importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PHASE1_DIR = ROOT / "outputs" / "phase1"
PHASE15_DIR = ROOT / "outputs" / "phase1.5"
SNAPSHOT_DIR = PHASE15_DIR / "snapshots"
PHASE1_H5 = PHASE1_DIR / "phase1_output.h5"
FRAGMENTS_AUTO = PHASE1_DIR / "fragments_auto.json"
FRAGMENTS_BE = PHASE1_DIR / "fragments_be.json"  # BE-matrix derived (preferred when present)
CASE_JSON = PHASE1_DIR / "case_classification.json"
SELECTED_CSV = PHASE1_DIR / "selected_reactions.csv"
BOND_CHANGES_JSON = PHASE1_DIR / "bond_changes.json"

REVIEW_LOG = PHASE15_DIR / "review_log_complete.json"
AUDIT_LOG = PHASE15_DIR / "review_audit.json"
PROGRESS_JSON = PHASE15_DIR / "review_progress.json"
BOOKMARKED_JSON = PHASE15_DIR / "bookmarked.json"
REJECTED_JSON = PHASE15_DIR / "rejected_for_phase2.json"


@dataclass
class ReactionStatic:
    """Static (read-only) info about a reaction loaded from Phase 1 outputs."""

    rxn_id: str
    source: str
    case: str  # original Stage 3.5 classification
    case_after_3_7: str  # may differ if demoted
    n_atoms: int
    n_heavy_atoms: int
    formula: str
    activation_energy: float
    energy_R: float
    energy_TS: float
    energy_P: float
    numbers: list[int]
    coords_5pts: list[list[list[float]]]
    energies_5pts: list[float]
    bonds_broken: list[list[int]]
    bonds_formed: list[list[int]]
    auto_suggestion: dict | None  # from fragments_auto.json (or None for hard-rejects)
    coords_P: list[list[float]] | None = None  # product (last trajectory frame), if extracted
    energy_P_actual: float | None = None  # tracking the P coord's energy (may differ from index summary)


def _load_p_bundle(rxn_id: str) -> tuple[list[list[float]] | None, float | None]:
    """Pull product-frame coords + energy from outputs/phase1/.tmp_p/<rxn>.npz."""
    p_path = PHASE1_DIR / ".tmp_p" / f"{rxn_id}.npz"
    if not p_path.exists():
        return None, None
    with np.load(p_path, allow_pickle=True) as data:
        coords = np.asarray(data["p_positions"]).tolist()
        energy = float(np.asarray(data["p_energy"]))
    return coords, energy


def _load_static() -> dict[str, ReactionStatic]:
    """Build the dict[rxn_id -> ReactionStatic] from Phase 1 artefacts.

    For reactions absent from `phase1_output.h5` (the 45 rejected/Case-C-no-cut
    set), we fall back to the in-progress npz bundles under outputs/phase1/.tmp.
    Product (P) coords come from outputs/phase1/.tmp_p (extracted post-hoc).
    """
    cases = json.loads(CASE_JSON.read_text())
    auto_phase1 = json.loads(FRAGMENTS_AUTO.read_text()) if FRAGMENTS_AUTO.exists() else {}
    auto_be = json.loads(FRAGMENTS_BE.read_text()) if FRAGMENTS_BE.exists() else {}
    # Strict-v1 BE-matrix output is the source of truth: even when the spec
    # ruled "strain_only" we surface that verdict so the reviewer can see
    # what the spec said and override manually rather than being shown the
    # legacy Phase 1 heuristic by default.
    auto: dict[str, dict] = {}
    for rid in set(auto_phase1) | set(auto_be):
        if rid in auto_be:
            auto[rid] = auto_be[rid]
        else:
            auto[rid] = auto_phase1[rid]
    bond_changes = json.loads(BOND_CHANGES_JSON.read_text())
    import pandas as pd
    sel = pd.read_csv(SELECTED_CSV).set_index("reaction_id")

    out: dict[str, ReactionStatic] = {}

    # 1) Pull rich data from HDF5 for the reactions that have it.
    in_h5: set[str] = set()
    if PHASE1_H5.exists():
        with h5py.File(PHASE1_H5, "r") as h5:
            for rxn_id in h5["reactions"]:
                grp = h5["reactions"][rxn_id]
                attrs = dict(grp.attrs)
                static_row = sel.loc[rxn_id] if rxn_id in sel.index else None
                p_coords, p_energy = _load_p_bundle(rxn_id)
                # Prefer the post-TS min-energy P (from .tmp_p) over the
                # halo8 last-frame value baked into the HDF5 attrs.
                resolved_p_energy = (
                    p_energy
                    if p_energy is not None
                    else float(attrs.get("halo8_energy_P", 0.0))
                )
                out[rxn_id] = ReactionStatic(
                    rxn_id=rxn_id,
                    source=str(attrs.get("source", static_row["source"] if static_row is not None else "")),
                    case=cases.get(rxn_id, {}).get("case", str(attrs.get("case", "?"))),
                    case_after_3_7=str(attrs.get("case", "?")),
                    n_atoms=int(attrs.get("n_atoms", 0)),
                    n_heavy_atoms=int(static_row["n_heavy_atoms"]) if static_row is not None else 0,
                    formula=str(attrs.get("halo8_formula", "")),
                    activation_energy=float(attrs.get("halo8_activation_energy", 0.0)),
                    energy_R=float(attrs.get("halo8_energy_R", 0.0)),
                    energy_TS=float(attrs.get("halo8_energy_TS", 0.0)),
                    energy_P=resolved_p_energy,
                    numbers=grp["numbers"][:].astype(int).tolist(),
                    coords_5pts=grp["coords_5pts"][:].tolist(),
                    energies_5pts=grp["energies_5pts"][:].tolist(),
                    bonds_broken=grp["bonds_broken"][:].astype(int).tolist(),
                    bonds_formed=grp["bonds_formed"][:].astype(int).tolist(),
                    auto_suggestion=auto.get(rxn_id),
                    coords_P=p_coords,
                    energy_P_actual=p_energy,
                )
                in_h5.add(rxn_id)

    # 2) Fill in the reactions that did not make it into the HDF5 from .npz bundles.
    tmp_dir = PHASE1_DIR / ".tmp"
    for rxn_id in cases:
        if rxn_id in out:
            continue
        npz_path = tmp_dir / f"{rxn_id}.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            numbers = np.asarray(data["numbers"], dtype=int).tolist()
            coords = np.asarray(data["coords_5pts"]).tolist()
            energies = np.asarray(data["energies_5pts"]).tolist()
        bd = bond_changes.get(rxn_id, {})
        static_row = sel.loc[rxn_id] if rxn_id in sel.index else None
        p_coords, p_energy = _load_p_bundle(rxn_id)
        out[rxn_id] = ReactionStatic(
            rxn_id=rxn_id,
            source=str(static_row["source"]) if static_row is not None else "",
            case=cases[rxn_id].get("case", "C"),
            case_after_3_7=cases[rxn_id].get("case", "C"),
            n_atoms=len(numbers),
            n_heavy_atoms=int(static_row["n_heavy_atoms"]) if static_row is not None else 0,
            formula=str(static_row["formula"]) if static_row is not None else "",
            activation_energy=float(static_row["activation_energy"]) if static_row is not None else 0.0,
            energy_R=float(energies[0]),
            energy_TS=float(energies[4]),
            energy_P=p_energy if p_energy is not None else 0.0,
            numbers=numbers,
            coords_5pts=coords,
            energies_5pts=energies,
            bonds_broken=[list(b) for b in bd.get("bonds_broken", [])],
            bonds_formed=[list(b) for b in bd.get("bonds_formed", [])],
            auto_suggestion=auto.get(rxn_id),  # often None for hard rejects
            coords_P=p_coords,
            energy_P_actual=p_energy,
        )

    return out


# Module-level cache populated by ensure_loaded().
_static: dict[str, ReactionStatic] = {}
_review_log: dict[str, dict] = {}
_audit: list[dict] = []


def ensure_loaded() -> None:
    """Idempotent: build caches and create review log on first run."""
    global _static, _review_log, _audit
    if _static:
        return
    PHASE15_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    _static = _load_static()

    if REVIEW_LOG.exists():
        _review_log = json.loads(REVIEW_LOG.read_text())
    else:
        _review_log = {}

    if AUDIT_LOG.exists():
        try:
            _audit = json.loads(AUDIT_LOG.read_text())
        except json.JSONDecodeError:
            _audit = []
    else:
        _audit = []

    # Initialise records for any reaction missing one. Auto suggestions become
    # the *current_definition* but with status "not_reviewed" until human looks.
    # For existing records, refresh the auto_suggestion (and current_definition
    # if still not_reviewed) so re-running the auto fragmentation flows through.
    for rxn_id, static in _static.items():
        if rxn_id in _review_log:
            existing = _review_log[rxn_id]
            existing["auto_suggestion"] = static.auto_suggestion
            if existing.get("review_status", "not_reviewed") == "not_reviewed" and static.auto_suggestion:
                a = static.auto_suggestion
                existing["current_definition"] = {
                    "frag1_atoms": list(a.get("frag1_atoms", [])),
                    "frag2_atoms": list(a.get("frag2_atoms", [])),
                    "h_caps": list(a.get("h_caps", [])),
                    "frag1_smiles": a.get("frag1_smiles"),
                    "frag2_smiles": a.get("frag2_smiles"),
                    "frag1_charge": a.get("frag1_charge", 0),
                    "frag2_charge": a.get("frag2_charge", 0),
                    "frag1_multiplicity": a.get("frag1_multiplicity", 1),
                    "frag2_multiplicity": a.get("frag2_multiplicity", 1),
                }
            continue
        rec: dict = {
            "rxn_id": rxn_id,
            "source": static.source,
            "case": static.case,
            "review_status": "not_reviewed",
            "current_definition": None,
            "auto_suggestion": static.auto_suggestion,
            "review_metadata": {
                "reviewer": None,
                "review_started_at": None,
                "review_completed_at": None,
                "review_duration_seconds": 0,
                "rationale": "",
                "confidence": None,
                "validated": False,
                "validation_warnings": [],
                "override_used": False,
            },
            "modification_history": [],
            "bookmarked": False,
        }
        if static.auto_suggestion is not None:
            a = static.auto_suggestion
            rec["current_definition"] = {
                "frag1_atoms": list(a.get("frag1_atoms", [])),
                "frag2_atoms": list(a.get("frag2_atoms", [])),
                "h_caps": list(a.get("h_caps", [])),
                "frag1_smiles": a.get("frag1_smiles"),
                "frag2_smiles": a.get("frag2_smiles"),
                "frag1_charge": a.get("frag1_charge", 0),
                "frag2_charge": a.get("frag2_charge", 0),
                "frag1_multiplicity": a.get("frag1_multiplicity", 1),
                "frag2_multiplicity": a.get("frag2_multiplicity", 1),
            }
        _review_log[rxn_id] = rec

    save_review_log()


def save_review_log() -> None:
    """Atomic write."""
    tmp = REVIEW_LOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_review_log, indent=2))
    os.replace(tmp, REVIEW_LOG)


def save_audit() -> None:
    tmp = AUDIT_LOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_audit, indent=2))
    os.replace(tmp, AUDIT_LOG)


def append_audit(entry: dict) -> None:
    _audit.append(entry)
    save_audit()


def get_static(rxn_id: str) -> ReactionStatic | None:
    return _static.get(rxn_id)


def get_review(rxn_id: str) -> dict | None:
    return _review_log.get(rxn_id)


def update_review(rxn_id: str, record: dict) -> None:
    _review_log[rxn_id] = record
    save_review_log()


def all_reactions() -> dict[str, ReactionStatic]:
    return _static


def all_reviews() -> dict[str, dict]:
    return _review_log


def progress_summary() -> dict:
    by_status: dict[str, int] = {}
    by_case: dict[str, dict[str, int]] = {"A": {"total": 0, "reviewed": 0},
                                          "B": {"total": 0, "reviewed": 0},
                                          "C": {"total": 0, "reviewed": 0}}
    bookmarks = 0
    for rxn_id, rec in _review_log.items():
        s = rec["review_status"]
        by_status[s] = by_status.get(s, 0) + 1
        case = rec.get("case", "C")
        if case in by_case:
            by_case[case]["total"] += 1
            if s != "not_reviewed":
                by_case[case]["reviewed"] += 1
        if rec.get("bookmarked"):
            bookmarks += 1
    total = len(_review_log)
    reviewed = total - by_status.get("not_reviewed", 0)
    return {
        "total": total,
        "reviewed": reviewed,
        "by_status": by_status,
        "by_case": by_case,
        "bookmarks": bookmarks,
    }
