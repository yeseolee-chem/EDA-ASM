"""Fix the last 3 modified reactions where the halogen is actually a
spectator but the distance-based bond detector mis-attributes it to a
different atom in the product frame.

User instruction (2026-05-13):

    마지막 3개의 수정이 필요한 반응은 할로젠은 어디에도 움직이지 않아.
    이것만 수정해줘

Background: ``detect_bonds_strict`` enforces a valence cap of 1 on
halogens. For these reactions, the halogen sits at ~1.78 Å from its
original carbon in P (a normal Cl–C bond) but is even closer (~1.6 Å)
to another atom (N or C) — so the greedy distance-first selector drops
the original Cl–C edge in favour of the spurious one, making the
classifier think Cl migrated.

Fix: for the listed (rxn_id, halogen_atom) pairs, force the halogen to
stay bonded to its R-frame partner in both R and P, drop the spurious
formed bond, and re-derive ``bonds_broken`` / ``bonds_formed`` /
migrant lists. Then hand the corrected bond-change set to
``fragment_hierarchical``.
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

# Reactions where the named atom is actually a spectator halogen
# despite the bond detector flagging it as migrating. The integer is the
# atom index of the halogen.
SPECTATOR_HALOGEN = {
    "Halogen_C4ClH4NS_rxn12917": 0,   # Cl bonded to C(1) throughout
    "Halogen_C4ClH4NS_rxn12932": 0,   # Cl bonded to C(1) throughout
    "Halogen_C4ClH5N2O_rxn13289": 7,  # already correctly handled, no-op
}


def _as_bond_set(seq) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for pair in seq or []:
        i, j = int(pair[0]), int(pair[1])
        out.add((i, j) if i < j else (j, i))
    return out


def _apply_spectator(
    spec_atom: int,
    bonds_R: set[tuple[int, int]],
    bonds_P: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Force ``spec_atom`` to keep its R-frame bonds in P.

    Any bond in P that involves ``spec_atom`` but isn't in R is dropped.
    Any bond in R that involves ``spec_atom`` but isn't in P is added
    back. Returns ``(new_bonds_R, new_bonds_P, broken, formed)``.
    """
    r_incident = {b for b in bonds_R if spec_atom in b}
    p_incident = {b for b in bonds_P if spec_atom in b}

    spurious_in_P = p_incident - r_incident      # to drop from P
    missing_in_P = r_incident - p_incident       # to add back to P

    new_P = (bonds_P - spurious_in_P) | missing_in_P
    new_R = set(bonds_R)
    broken = new_R - new_P
    formed = new_P - new_R
    return new_R, new_P, broken, formed


def main() -> None:
    with open(STAGE5A / "frames_cache.pkl", "rb") as f:
        frames = pickle.load(f)

    summary = json.loads((STAGE5A / "fragmentation_summary.json").read_text())
    by_id = {r["reaction_id"]: i for i, r in enumerate(summary)}
    review_log = json.loads((STAGE5A / "review_log.json").read_text())
    audit_path = STAGE5A / "review_audit.json"
    audit = json.loads(audit_path.read_text())

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for rxn, spec_atom in SPECTATOR_HALOGEN.items():
        detail_path = PER_REACTION / rxn / "result.json"
        detail = json.loads(detail_path.read_text())
        dbg = detail["debug"]
        bonds_R = _as_bond_set(dbg["bonds_R"])
        bonds_P = _as_bond_set(dbg["bonds_P"])

        new_R, new_P, broken, formed = _apply_spectator(
            spec_atom, bonds_R, bonds_P
        )

        # Drop the spectator from migrant lists.
        m_h = [int(a) for a in (dbg.get("migrating_H_atoms") or []) if int(a) != spec_atom]
        m_x = [int(a) for a in (dbg.get("migrating_halogen_atoms") or []) if int(a) != spec_atom]
        migrants = sorted(set(m_h) | set(m_x))

        r = frames[rxn]
        n_atoms = int(r.n_atoms)
        numbers = np.asarray(r.numbers, dtype=int)
        positions_R = np.asarray(r.positions_R, dtype=float)

        result = fragment_hierarchical(
            n_atoms=n_atoms,
            numbers=numbers,
            bonds_R=new_R,
            bonds_broken=broken,
            bonds_formed=formed,
            migrating_atoms=migrants,
            positions_R=positions_R,
            pattern_label=detail["result"]["pattern"],
        )

        # Patch the debug block so the dashboard shows the corrected
        # bond changes and notes the override.
        new_debug = {
            **dbg,
            "bonds_R": sorted(tuple(sorted(b)) for b in new_R),
            "bonds_P": sorted(tuple(sorted(b)) for b in new_P),
            "bonds_broken": sorted(tuple(sorted(b)) for b in broken),
            "bonds_formed": sorted(tuple(sorted(b)) for b in formed),
            "n_bond_changes": len(broken) + len(formed),
            "migrating_H_atoms": m_h,
            "migrating_halogen_atoms": m_x,
            "migrating_atoms": migrants or None,
            "spectator_atoms": [int(spec_atom)],
            "spectator_atom_reason": (
                f"halogen atom {spec_atom} mis-classified as migrating by the "
                "valence-capped bond detector; geometry shows the original "
                "halogen–scaffold bond stays intact in P"
            ),
        }
        # Keep the old fragmentation 'pattern_from_classifier' label so
        # downstream filters keep working.

        new_detail = {
            **{k: v for k, v in detail.items() if k != "result"},
            "debug": new_debug,
            "result": result.to_dict(),
        }
        detail_path.write_text(json.dumps(new_detail, indent=2))

        # Update fragmentation_summary.json entry.
        idx = by_id[rxn]
        old = summary[idx]
        fragment_atoms = [list(map(int, f.atom_indices.tolist())) for f in result.fragments]
        fragment_roles = [f.role for f in result.fragments]
        fragment_mults = [int(f.multiplicity) for f in result.fragments]
        merged = {
            **old,
            "pattern": result.pattern,
            "n_fragments": len(result.fragments),
            "confidence": float(result.confidence),
            "notes": result.notes + f"; halogen {spec_atom} forced as spectator",
            "fragment_atoms": fragment_atoms,
            "fragment_roles": fragment_roles,
            "fragment_multiplicities": fragment_mults,
            "n_bond_changes": len(broken) + len(formed),
            "n_caps": 0,
        }
        merged.pop("p2_subtype", None)
        summary[idx] = merged

        # Reset review log entry.
        prev = review_log.get(rxn, {})
        review_log[rxn] = {
            "rxn_id": rxn,
            "review_status": "not_reviewed",
            "auto_pattern": result.pattern,
            "auto_confidence": float(result.confidence),
            "rationale": prev.get("rationale", ""),
            "reviewer": None,
            "review_completed_at": None,
            "bookmarked": bool(prev.get("bookmarked", True)),
            "previously_modified": True,
            "last_rerun_at": now_iso,
        }

        audit.append({
            "rxn_id": rxn,
            "event": "halogen_spectator_override_2026-05-13",
            "spectator_atom": int(spec_atom),
            "new_n_fragments": len(result.fragments),
            "new_roles": fragment_roles,
            "new_atoms": fragment_atoms,
            "new_multiplicities": fragment_mults,
            "notes": result.notes,
            "n_bond_changes_corrected": len(broken) + len(formed),
            "at": now_iso,
        })

        print(
            f"  {rxn} (spec atom {spec_atom}): {result.pattern} → "
            f"{len(result.fragments)} fragments (mult={fragment_mults})"
        )
        for role, atoms, mult in zip(fragment_roles, fragment_atoms, fragment_mults):
            print(f"    {role:20s} atoms={atoms}  mult={mult}")

    (STAGE5A / "fragmentation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    (STAGE5A / "review_log.json").write_text(
        json.dumps(review_log, indent=2)
    )
    audit_path.write_text(json.dumps(audit, indent=2))

    statuses: dict[str, int] = {}
    for v in review_log.values():
        s = v.get("review_status", "not_reviewed")
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\n=== Final review state ===")
    print(f"total={len(review_log)}, status={statuses}")


if __name__ == "__main__":
    main()
