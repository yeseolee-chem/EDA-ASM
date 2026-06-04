"""Loader + review-log glue for the Stage 5-A fragmentation pass.

Reads ``outputs/stage5a/`` (produced by
``scripts/run_stage5a_fragmentation.py``) and exposes it to the Flask app
in the same shape the existing Phase 1.5 dashboard uses, so the same UI
patterns work for confirming the 400 P0/P1/P2/P3 splits one by one.

Reviews are persisted to ``outputs/stage5a/review_log.json`` (separate
from the BE-matrix log under ``outputs/phase1.5/``).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


# Stretch factor for flagging spectator bonds that have unusually long
# TS-frame distance compared to the sum of covalent radii. 1.5x catches
# bonds that are clearly stretched beyond a normal TS distortion (e.g.
# T1x_C4H7NO2_rxn03938: N–H at 1.74 Å vs 1.02 Å covalent sum = ratio 1.71).
STRETCHED_BOND_RATIO_TS = 1.5

# Cordero 2008 covalent radii (Å), enough subset for Halo8 elements.
_COVALENT_RADIUS = {
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
    35: 1.20, 53: 1.39,
}


def _covalent_sum(zi: int, zj: int) -> float:
    return _COVALENT_RADIUS.get(int(zi), 0.80) + _COVALENT_RADIUS.get(int(zj), 0.80)


ROOT = Path(__file__).resolve().parents[2]
# The pickled frame cache contains ``eda_asm.stage5a.loader.ReactionFrames``
# instances, which transitively imports ``stage0_fragmentation``. Put both
# roots on sys.path so unpickle works regardless of cwd.
for _p in (ROOT, ROOT / "src"):
    p = str(_p)
    if p not in sys.path:
        sys.path.insert(0, p)

STAGE5A_DIR = ROOT / "outputs" / "stage5a"
SUMMARY_JSON = STAGE5A_DIR / "fragmentation_summary.json"
FRAMES_CACHE = STAGE5A_DIR / "frames_cache.pkl"
PER_REACTION_DIR = STAGE5A_DIR / "per_reaction"
REVIEW_LOG = STAGE5A_DIR / "review_log.json"
AUDIT_LOG = STAGE5A_DIR / "review_audit.json"


_summary: list[dict] | None = None
_summary_by_id: dict[str, dict] | None = None
_frames: dict | None = None
_review_log: dict[str, dict] | None = None
_audit: list[dict] | None = None
_loaded: bool = False


def ensure_loaded() -> None:
    global _summary, _summary_by_id, _frames, _review_log, _audit, _loaded
    if _loaded:
        return
    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(
            f"{SUMMARY_JSON} not found — run scripts/run_stage5a_fragmentation.py first"
        )
    summary = json.loads(SUMMARY_JSON.read_text())
    summary_by_id = {r["reaction_id"]: r for r in summary}
    if not FRAMES_CACHE.exists():
        raise FileNotFoundError(
            f"{FRAMES_CACHE} not found — re-run the driver with --xyz to populate the cache"
        )
    frames = pickle.loads(FRAMES_CACHE.read_bytes())
    review_log = (
        json.loads(REVIEW_LOG.read_text()) if REVIEW_LOG.exists() else {}
    )
    audit = (
        json.loads(AUDIT_LOG.read_text()) if AUDIT_LOG.exists() else []
    )
    # Publish only after every step succeeded; partial init would otherwise
    # poison subsequent calls.
    _summary = summary
    _summary_by_id = summary_by_id
    _frames = frames
    _review_log = review_log
    _audit = audit
    _loaded = True
    # Backfill review-log entries for every reaction in the summary.
    changed = False
    for rec in _summary:
        rid = rec["reaction_id"]
        if rid in _review_log:
            continue
        _review_log[rid] = {
            "rxn_id": rid,
            "review_status": "not_reviewed",
            "auto_pattern": rec["pattern"],
            "auto_confidence": rec["confidence"],
            "rationale": "",
            "reviewer": None,
            "review_completed_at": None,
            "bookmarked": False,
        }
        changed = True
    if changed:
        save_review_log()


def save_review_log() -> None:
    assert _review_log is not None
    STAGE5A_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REVIEW_LOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_review_log, indent=2))
    os.replace(tmp, REVIEW_LOG)


def append_audit(entry: dict) -> None:
    assert _audit is not None
    _audit.append(entry)
    tmp = AUDIT_LOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_audit, indent=2))
    os.replace(tmp, AUDIT_LOG)


def list_reactions() -> list[dict]:
    """Compact one-row-per-reaction list for the dashboard.

    Rows are returned with priority-review entries first (those tagged
    `priority_review=True` in review_log), so flagged cases land at the
    top of the dashboard table.
    """
    ensure_loaded()
    assert _summary is not None and _review_log is not None
    rows = []
    for rec in _summary:
        rid = rec["reaction_id"]
        rev = _review_log[rid]
        rows.append({
            "rxn_id": rid,
            "source": rec["source"],
            "n_atoms": rec["n_atoms"],
            "pattern": rec["pattern"],
            "n_fragments": rec["n_fragments"],
            "confidence": rec["confidence"],
            "n_bond_changes": rec["n_bond_changes"],
            "activation_energy": rec["activation_energy"],
            "review_status": rev["review_status"],
            "bookmarked": bool(rev.get("bookmarked")),
            "previously_modified": bool(rev.get("previously_modified")),
            "priority_review": bool(rev.get("priority_review")),
            "reviewer": rev.get("reviewer"),
            "review_completed_at": rev.get("review_completed_at"),
        })
    # Priority-review entries first, then the rest preserving original order.
    rows.sort(key=lambda r: (not r["priority_review"], r["rxn_id"]))
    return rows


def progress() -> dict:
    """Summary counters for the dashboard."""
    ensure_loaded()
    assert _summary is not None and _review_log is not None
    by_status: dict[str, int] = {}
    by_pattern: dict[str, dict[str, int]] = {}
    bookmarks = 0
    for rec in _summary:
        rid = rec["reaction_id"]
        pat = rec["pattern"]
        by_pattern.setdefault(pat, {"total": 0, "reviewed": 0})
        by_pattern[pat]["total"] += 1
        rev = _review_log[rid]
        s = rev["review_status"]
        by_status[s] = by_status.get(s, 0) + 1
        if s != "not_reviewed":
            by_pattern[pat]["reviewed"] += 1
        if rev.get("bookmarked"):
            bookmarks += 1
    total = len(_summary)
    return {
        "total": total,
        "reviewed": total - by_status.get("not_reviewed", 0),
        "by_status": by_status,
        "by_pattern": by_pattern,
        "bookmarks": bookmarks,
    }


def get_reaction_payload(rxn_id: str) -> dict | None:
    """Everything the per-reaction viewer needs in one JSON blob."""
    ensure_loaded()
    assert _summary_by_id is not None and _frames is not None and _review_log is not None
    if rxn_id not in _summary_by_id:
        return None
    summary = _summary_by_id[rxn_id]
    frames = _frames.get(rxn_id)
    if frames is None:
        return None
    # Per-reaction detailed json (has cap H positions for P3)
    detail_path = PER_REACTION_DIR / rxn_id / "result.json"
    detail = json.loads(detail_path.read_text()) if detail_path.exists() else {}
    result = detail.get("result", {})
    debug = detail.get("debug", {})

    fragments_rich = []
    for frag in result.get("fragments", []):
        # Spec uses A=blue, B=orange, tether=purple, whole=gray (set CSS-side too).
        fragments_rich.append({
            "role": frag["role"],
            "atom_indices": frag["atom_indices"],
            "multiplicity": frag["multiplicity"],
            "cap_attachment": frag.get("cap_attachment"),
        })

    rev = _review_log[rxn_id]

    # Skeleton bonds = bonds present in BOTH R and P (i.e., spectator
    # bonds that don't break or form). At the TS frame these can be
    # heavily stretched (e.g. a spectator N–H dragged through the saddle
    # point), and 3Dmol's auto bond detection may miss the stretched
    # ones — so we send them explicitly and the viewer draws them as
    # solid cylinders regardless of distance.
    bonds_R_list = debug.get("bonds_R", []) or []
    bonds_P_list = debug.get("bonds_P", []) or []
    bonds_R_set = {tuple(sorted((int(i), int(j)))) for i, j in bonds_R_list}
    bonds_P_set = {tuple(sorted((int(i), int(j)))) for i, j in bonds_P_list}
    skeleton_bonds = sorted(bonds_R_set & bonds_P_set)

    # Flag spectator bonds whose TS distance is significantly longer
    # than the typical covalent bond length for their element pair.
    numbers_arr = frames.numbers
    ts_pos = frames.positions_TS
    stretched_at_TS: list[dict[str, Any]] = []
    for i, j in skeleton_bonds:
        d_ts = float(np.linalg.norm(ts_pos[i] - ts_pos[j]))
        cs = _covalent_sum(int(numbers_arr[i]), int(numbers_arr[j]))
        if cs <= 0:
            continue
        ratio = d_ts / cs
        if ratio >= STRETCHED_BOND_RATIO_TS:
            stretched_at_TS.append({
                "i": int(i),
                "j": int(j),
                "distance_TS": round(d_ts, 3),
                "covalent_sum": round(cs, 3),
                "ratio": round(ratio, 2),
            })

    # TS bond-graph connected components — when > 1, the molecule is
    # geometrically in pieces at the saddle point (the reviewer raised
    # this for cases like T1x_C3H3NO_rxn00389 where the TS is two
    # equal halves). Useful diagnostic for dissociative / fly-by-style
    # mechanisms even when R and P are single molecules.
    from stage0_fragmentation.bond_detection import detect_bonds_strict
    try:
        ts_bonds = detect_bonds_strict(numbers_arr, ts_pos)
        ts_adj: dict[int, set[int]] = {a: set() for a in range(int(frames.n_atoms))}
        for i, j in ts_bonds:
            ts_adj[int(i)].add(int(j))
            ts_adj[int(j)].add(int(i))
        ts_components_list: list[list[int]] = []
        seen: set[int] = set()
        for a in range(int(frames.n_atoms)):
            if a in seen:
                continue
            stack = [a]
            comp = []
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                comp.append(x)
                stack.extend(ts_adj[x] - seen)
            ts_components_list.append(sorted(comp))
        ts_components_list.sort(key=len, reverse=True)
    except Exception:
        ts_components_list = []
    ts_n_components = len(ts_components_list)

    # Tag each atom with its fragment role for the viewer.
    role_of_atom: dict[int, str] = {}
    for frag in fragments_rich:
        for a in frag["atom_indices"]:
            role_of_atom[int(a)] = frag["role"]
    # Atoms not assigned (shouldn't happen after our fixes) default to "unassigned".
    for i in range(int(frames.n_atoms)):
        role_of_atom.setdefault(i, "unassigned")

    return {
        "rxn_id": rxn_id,
        "source": summary["source"],
        "n_atoms": int(frames.n_atoms),
        "numbers": [int(z) for z in frames.numbers.tolist()],
        "positions_R": frames.positions_R.tolist(),
        "positions_TS": frames.positions_TS.tolist(),
        "positions_P": frames.positions_P.tolist(),
        "energy_R": float(frames.energy_R),
        "energy_TS": float(frames.energy_TS),
        "energy_P": float(frames.energy_P),
        "activation_energy": float(frames.energy_TS - frames.energy_R),
        "pattern": result.get("pattern", summary["pattern"]),
        "confidence": result.get("confidence", summary["confidence"]),
        "notes": result.get("notes", summary["notes"]),
        "n_bond_changes": debug.get("n_bond_changes", summary["n_bond_changes"]),
        "fragments": fragments_rich,
        "role_of_atom": role_of_atom,
        "cap_h_positions": result.get("cap_h_positions"),
        "bonds_R": debug.get("bonds_R", []),
        "bonds_P": debug.get("bonds_P", []),
        "bonds_broken": debug.get("bonds_broken", []),
        "bonds_formed": debug.get("bonds_formed", []),
        "core_atoms": debug.get("core_atoms", []),
        "bonds_skeleton": [list(b) for b in skeleton_bonds],
        "stretched_at_TS": stretched_at_TS,
        "ts_n_components": ts_n_components,
        "ts_components": ts_components_list,
        "debug": debug,
        "review": rev,
    }


def update_review(rxn_id: str, patch: dict[str, Any]) -> dict:
    ensure_loaded()
    assert _review_log is not None
    rec = _review_log.get(rxn_id)
    if rec is None:
        raise KeyError(rxn_id)
    rec.update(patch)
    save_review_log()
    return rec


def neighbour_ids(rxn_id: str) -> tuple[str | None, str | None]:
    ensure_loaded()
    assert _summary is not None
    ids = [r["reaction_id"] for r in _summary]
    if rxn_id not in ids:
        return None, None
    i = ids.index(rxn_id)
    prev = ids[i - 1] if i > 0 else None
    nxt = ids[i + 1] if i < len(ids) - 1 else None
    return prev, nxt
