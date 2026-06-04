"""Re-fragment the 7 reactions the user re-flagged as `modified`.

The user's instruction (2026-05-13):

    수정이 필요한 7개를 다시 설정해놨어. fragment안에서도 또 다른 fragmet로
    나뉠 수 있어. 이점을 염두해두고 다시 수정해줘. 기존 accept은 건들지 말아줘

i.e. allow each fragment to be sub-divided into further fragments where
the TS bond-change set warrants it; do not touch the 493 currently
``accepted`` reactions.

Strategy: drive ``fragment_hierarchical`` (added in fragmenters.py) on
each of the 7 reactions using the bond-change debug already cached in
``outputs/stage5a/per_reaction/<rxn>/result.json``. Overwrite the per-
reaction JSON and the matching entry of ``fragmentation_summary.json``;
flip the review-log entries back to ``not_reviewed`` so the user re-
reviews them. The 493 accepted entries remain byte-identical.
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eda_asm.stage5a.fragmenters import fragment_hierarchical  # noqa: E402

STAGE5A = ROOT / "outputs" / "stage5a"
PER_REACTION = STAGE5A / "per_reaction"

# These are the 7 reactions the user re-flagged after rejecting the
# previous P2B / P3-tether / P4-demoted treatments. Pattern label per
# reaction is preserved from the classifier (we only swap in a
# hierarchical fragmentation; the underlying classification doesn't
# change).
MODIFIED_RXN_IDS = [
    "Halogen_BrC4H4NO_rxn10056",
    "Halogen_BrC4H4NS_rxn10113",
    "Halogen_C4ClH4NS_rxn12917",
    "Halogen_C4ClH4NS_rxn12932",
    "Halogen_C4ClH5N2O_rxn13289",
    "T1x_C3H3NO_rxn00389",
    "T1x_C3H5NO2_rxn01106",
]


def _as_bond_set(seq) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for pair in seq or []:
        i, j = int(pair[0]), int(pair[1])
        out.add((i, j) if i < j else (j, i))
    return out


def _migrants_from_debug(dbg: dict) -> list[int]:
    if dbg.get("migrating_atoms"):
        return [int(a) for a in dbg["migrating_atoms"]]
    mh = dbg.get("migrating_H_atoms") or []
    mx = dbg.get("migrating_halogen_atoms") or []
    return sorted({int(a) for a in mh} | {int(a) for a in mx})


def rerun_one(rxn_id: str, frames) -> dict:
    detail_path = PER_REACTION / rxn_id / "result.json"
    detail = json.loads(detail_path.read_text())
    debug = detail["debug"]
    pattern = debug["pattern_from_classifier"]

    bonds_R = _as_bond_set(debug.get("bonds_R"))
    bonds_broken = _as_bond_set(debug.get("bonds_broken"))
    bonds_formed = _as_bond_set(debug.get("bonds_formed"))
    migrants = _migrants_from_debug(debug)

    r = frames[rxn_id]
    n_atoms = int(r.n_atoms)
    numbers = np.asarray(r.numbers, dtype=int)
    positions_R = np.asarray(r.positions_R, dtype=float)

    result = fragment_hierarchical(
        n_atoms=n_atoms,
        numbers=numbers,
        bonds_R=bonds_R,
        bonds_broken=bonds_broken,
        bonds_formed=bonds_formed,
        migrating_atoms=migrants,
        positions_R=positions_R,
        pattern_label=pattern,
    )

    # Write per_reaction/<rxn>/result.json with the new fragmentation,
    # preserving the existing debug block and energy summary.
    new_detail = {
        **{k: v for k, v in detail.items() if k not in ("result",)},
        "result": result.to_dict(),
    }
    detail_path.write_text(json.dumps(new_detail, indent=2))

    fragment_atoms = [list(map(int, f.atom_indices.tolist())) for f in result.fragments]
    fragment_roles = [f.role for f in result.fragments]
    fragment_mults = [int(f.multiplicity) for f in result.fragments]
    return {
        "reaction_id": rxn_id,
        "pattern": result.pattern,
        "n_fragments": len(result.fragments),
        "confidence": float(result.confidence),
        "notes": result.notes,
        "fragment_atoms": fragment_atoms,
        "fragment_roles": fragment_roles,
        "fragment_multiplicities": fragment_mults,
        "n_caps": 0,
    }


def main() -> None:
    print("loading frames cache…")
    with open(STAGE5A / "frames_cache.pkl", "rb") as f:
        frames = pickle.load(f)

    print("loading summary + review log…")
    summary = json.loads((STAGE5A / "fragmentation_summary.json").read_text())
    by_id = {r["reaction_id"]: i for i, r in enumerate(summary)}

    review_log = json.loads((STAGE5A / "review_log.json").read_text())
    audit_path = STAGE5A / "review_audit.json"
    audit = (
        json.loads(audit_path.read_text()) if audit_path.exists() else []
    )

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for rxn in MODIFIED_RXN_IDS:
        new = rerun_one(rxn, frames)
        idx = by_id[rxn]
        old = summary[idx]
        # Preserve fields the rerun doesn't recompute.
        merged = {
            **old,
            "pattern": new["pattern"],
            "n_fragments": new["n_fragments"],
            "confidence": new["confidence"],
            "notes": new["notes"],
            "fragment_atoms": new["fragment_atoms"],
            "fragment_roles": new["fragment_roles"],
            "fragment_multiplicities": new["fragment_multiplicities"],
            "n_caps": new["n_caps"],
        }
        # p2_subtype no longer applies after sub-fragmentation
        merged.pop("p2_subtype", None)
        summary[idx] = merged

        # Reset review log so the user re-reviews; keep bookmark + flag.
        prev = review_log.get(rxn, {})
        review_log[rxn] = {
            "rxn_id": rxn,
            "review_status": "not_reviewed",
            "auto_pattern": new["pattern"],
            "auto_confidence": new["confidence"],
            "rationale": prev.get("rationale", ""),
            "reviewer": None,
            "review_completed_at": None,
            "bookmarked": bool(prev.get("bookmarked", True)),
            "previously_modified": True,
            "last_rerun_at": now_iso,
        }

        audit.append({
            "rxn_id": rxn,
            "event": "hierarchical_rerun_2026-05-13",
            "old_n_fragments": old["n_fragments"],
            "old_roles": old.get("fragment_roles"),
            "old_atoms": old.get("fragment_atoms"),
            "new_n_fragments": new["n_fragments"],
            "new_roles": new["fragment_roles"],
            "new_atoms": new["fragment_atoms"],
            "new_multiplicities": new["fragment_multiplicities"],
            "notes": new["notes"],
            "at": now_iso,
        })

        print(
            f"  {rxn}: {new['pattern']} → {new['n_fragments']} fragments "
            f"(mult={new['fragment_multiplicities']})"
        )

    (STAGE5A / "fragmentation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    (STAGE5A / "review_log.json").write_text(
        json.dumps(review_log, indent=2)
    )
    audit_path.write_text(json.dumps(audit, indent=2))

    # Final state report.
    statuses: dict[str, int] = {}
    for v in review_log.values():
        s = v.get("review_status", "not_reviewed")
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\n=== Final review state ===")
    print(f"total={len(review_log)}, status={statuses}")


if __name__ == "__main__":
    main()
