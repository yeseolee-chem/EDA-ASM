"""Stage 3.9 — Integrate auto fragments with manual review log.

Reads:
- fragments_auto.json (auto-decided fragments for Case A and most Case B)
- manual_review_log.json (reviewer decisions for queued reactions)
- case_classification.json
Writes:
- fragments_final.json — one entry per reaction that survived review
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bonds import detect_bonds
from .halo8_io import _formula_from_numbers
from .logging_setup import get_logger, log_header
from .paths import (
    CASE_JSON,
    FRAGMENTS_AUTO_JSON,
    FRAGMENTS_FINAL_JSON,
    MANUAL_REVIEW_LOG,
    TMP_DIR,
    ensure_dirs,
)


def _load_review_log(p: Path) -> dict[str, dict]:
    """Accept three formats:
    1. List of {reaction_id, decision, ...} dicts (legacy phase1 spec).
    2. Dict {rxn_id -> {decision, frag1_atoms, frag2_atoms, ...}} (legacy).
    3. Dict {rxn_id -> phase1.5 review record with review_status,
       current_definition, review_metadata, ...} (Phase 1.5 tool output).
    """
    if not p.exists():
        return {}
    text = p.read_text().strip()
    if not text:
        return {}
    data = json.loads(text)
    if isinstance(data, list):
        return {entry["reaction_id"]: entry for entry in data if "reaction_id" in entry}
    if not isinstance(data, dict):
        raise ValueError(f"unrecognized review log format: {type(data).__name__}")

    out: dict[str, dict] = {}
    for rid, rec in data.items():
        if isinstance(rec, dict) and "review_status" in rec and "current_definition" in rec:
            # Phase 1.5 schema -> normalise to legacy {decision, frag1_atoms, ...}.
            status = rec.get("review_status", "not_reviewed")
            decision_map = {
                "accepted": "accept",
                "modified": "modify",
                "rejected": "reject",
                "not_reviewed": None,
                "bookmarked": None,
            }
            decision = decision_map.get(status)
            if decision is None:
                # No actionable review yet -> let caller fall back to auto.
                continue
            curr = rec.get("current_definition") or {}
            out[rid] = {
                "reaction_id": rid,
                "decision": decision,
                "frag1_atoms": curr.get("frag1_atoms"),
                "frag2_atoms": curr.get("frag2_atoms"),
                "rationale": (rec.get("review_metadata") or {}).get("rationale"),
                "reviewer": (rec.get("review_metadata") or {}).get("reviewer"),
                "h_caps": curr.get("h_caps"),
                "frag1_smiles": curr.get("frag1_smiles"),
                "frag2_smiles": curr.get("frag2_smiles"),
                "frag1_charge": curr.get("frag1_charge", 0),
                "frag2_charge": curr.get("frag2_charge", 0),
                "frag1_multiplicity": curr.get("frag1_multiplicity", 1),
                "frag2_multiplicity": curr.get("frag2_multiplicity", 1),
            }
        else:
            out[rid] = rec
    return out


def _validate(numbers: np.ndarray, frag1: list[int], frag2: list[int]) -> str | None:
    set1, set2 = set(frag1), set(frag2)
    if set1 & set2:
        return "fragments overlap"
    if set1 | set2 != set(range(len(numbers))):
        return "fragments do not cover all atoms"
    if not frag1 or not frag2:
        return "empty fragment"
    return None


def run(
    case_json: Path | None = None,
    auto_json: Path | None = None,
    review_log: Path | None = None,
    output_json: Path | None = None,
) -> dict:
    ensure_dirs()
    log = get_logger("phase1.stage3_9")
    log_header(log, "3.9 final fragment integration")
    if case_json is None:
        case_json = CASE_JSON
    if auto_json is None:
        auto_json = FRAGMENTS_AUTO_JSON
    if review_log is None:
        review_log = MANUAL_REVIEW_LOG
    if output_json is None:
        output_json = FRAGMENTS_FINAL_JSON

    cases = json.loads(case_json.read_text())
    autos = json.loads(auto_json.read_text()) if auto_json.exists() else {}
    reviews = _load_review_log(review_log)

    out: dict[str, dict] = {}
    rejected: list[str] = []
    for rid, info in cases.items():
        review = reviews.get(rid)
        decision = (review or {}).get("decision")
        auto = autos.get(rid)

        if decision == "reject":
            rejected.append(rid)
            continue

        if decision == "modify" and review:
            frag1 = list(map(int, review.get("frag1_atoms") or []))
            frag2 = list(map(int, review.get("frag2_atoms") or []))
        elif auto is not None:
            frag1 = list(auto.get("frag1_atoms", []))
            frag2 = list(auto.get("frag2_atoms", []))
        else:
            log.info("skip %s: case=%s, no auto fragment and no review", rid, info.get("case"))
            rejected.append(rid)
            continue

        # Validate against actual atom count
        npz_path = TMP_DIR / f"{rid}.npz"
        if not npz_path.exists():
            log.warning("skip %s: missing npz", rid)
            rejected.append(rid)
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            numbers = np.asarray(data["numbers"], dtype=int)
            coords = np.asarray(data["coords_5pts"])
        err = _validate(numbers, frag1, frag2)
        if err:
            log.warning("skip %s: %s", rid, err)
            rejected.append(rid)
            continue

        record: dict = dict(auto) if auto else {}
        # Preserve auto review_status (auto_accepted / needs_review / ...) when
        # there is no manual decision; otherwise the manual decision wins.
        if review:
            record["review_status"] = review.get("decision", record.get("review_status", "auto_accepted"))
            record["reviewer"] = review.get("reviewer")
            record["review_rationale"] = review.get("rationale")
            # When the reviewer modified the partition, the SMILES / H caps
            # computed by the review tool override whatever fragments_auto had.
            if review.get("decision") == "modify":
                if review.get("frag1_smiles") is not None:
                    record["frag1_smiles"] = review["frag1_smiles"]
                if review.get("frag2_smiles") is not None:
                    record["frag2_smiles"] = review["frag2_smiles"]
                if review.get("h_caps") is not None:
                    record["h_caps"] = review["h_caps"]
                for k in ("frag1_charge", "frag2_charge", "frag1_multiplicity", "frag2_multiplicity"):
                    if review.get(k) is not None:
                        record[k] = review[k]
        record.update(
            {
                "case": info["case"],
                "frag1_atoms": frag1,
                "frag2_atoms": frag2,
                "frag1_formula": _formula_from_numbers(numbers[frag1]),
                "frag2_formula": _formula_from_numbers(numbers[frag2]),
            }
        )
        # Ensure required fields exist with defaults.
        record.setdefault("h_caps", [])
        record.setdefault("frag1_charge", 0)
        record.setdefault("frag2_charge", 0)
        record.setdefault("frag1_multiplicity", 1)
        record.setdefault("frag2_multiplicity", 1)
        record.setdefault("frag1_smiles", None)
        record.setdefault("frag2_smiles", None)
        record.setdefault("auto_confidence", 1.0 if not review else None)
        out[rid] = record

    output_json.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s with %d entries (rejected=%d)", output_json, len(out), len(rejected))
    return {"final": out, "rejected": rejected}
