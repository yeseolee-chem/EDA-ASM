"""Run Stage-0 BE-matrix fragmentation on every selected reaction.

Reads:
- outputs/phase1/.tmp/<rxn>.npz       (R coords from frame 0)
- outputs/phase1/.tmp_p/<rxn>.npz     (post-TS-min P coords)
- outputs/phase1/selected_reactions.csv

Writes:
- outputs/phase1/fragments_be.json    (BE-matrix-derived fragments)

The output schema mirrors fragments_auto.json so the Phase 1.5 review tool
can swap it in transparently.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from stage0_fragmentation.strict_v1 import fragmentation_strict_v1  # noqa: E402

PHASE1 = ROOT / "outputs" / "phase1"
TMP = PHASE1 / ".tmp"
TMP_P = PHASE1 / ".tmp_p"
OUT = PHASE1 / "fragments_be.json"

ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br"}


def _formula(numbers: list[int], atom_idx: list[int]) -> str:
    counts: dict[str, int] = {}
    for i in atom_idx:
        sym = ELEM.get(int(numbers[i]), f"Z{numbers[i]}")
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


def _h_caps_from_sites(
    sites: list[tuple[int, int]],
    coords_R: np.ndarray,
) -> list[dict]:
    """Place an H atom along the (anchor → partner) direction at 1.09 Å."""
    out = []
    H_LEN = 1.09
    for anchor, partner in sites:
        d = coords_R[partner] - coords_R[anchor]
        norm = float(np.linalg.norm(d))
        if norm < 1e-6:
            unit = np.array([1.0, 0.0, 0.0])
        else:
            unit = d / norm
        h_pos = coords_R[anchor] + unit * H_LEN
        out.append({
            "attached_to_atom": int(anchor),
            "h_position": h_pos.tolist(),
            "from_broken_bond": [int(anchor), int(partner)],
        })
    return out


def main() -> int:
    selected = pd.read_csv(PHASE1 / "selected_reactions.csv")
    out: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    counter: Counter[str] = Counter()

    for rxn_id in selected["reaction_id"]:
        npz_r = TMP / f"{rxn_id}.npz"
        npz_p = TMP_P / f"{rxn_id}.npz"
        if not npz_r.exists():
            failures.append((rxn_id, "missing R npz"))
            continue
        try:
            with np.load(npz_r, allow_pickle=True) as d:
                numbers = np.asarray(d["numbers"], dtype=int)
                coords_5pts = np.asarray(d["coords_5pts"])
                coords_R = coords_5pts[0]
                coords_TS = coords_5pts[4]
            coords_P = None
            if npz_p.exists():
                with np.load(npz_p, allow_pickle=True) as d:
                    coords_P = np.asarray(d["p_positions"])
        except Exception as e:
            failures.append((rxn_id, f"npz load: {e}"))
            continue

        # Strict v1 uses the trajectory-end P (Halo8 last frame). No TS
        # fallback — that would conflate transition-state geometry with
        # product geometry and break case classification.
        if coords_P is None:
            failures.append((rxn_id, "no P coords"))
            continue
        chosen_label = "last_frame"
        try:
            res = fragmentation_strict_v1(numbers, coords_R, coords_P)
        except Exception as e:
            failures.append((rxn_id, f"{type(e).__name__}: {e}"))
            continue

        if len(res.fragments) < 2:
            counter["only_one_fragment"] += 1
        if res.is_pure_rearrangement:
            counter[f"pure_rearrangement:{res.fallback_strategy}"] += 1
        if not res.reactive_bonds:
            counter["no_reactive_bonds_degenerate"] += 1

        # Pick fragment 1 = larger, fragment 2 = smaller (or merge tail).
        frags = sorted(res.fragments, key=lambda f: (-len(f), min(f) if f else 0))
        if len(frags) == 1:
            frag1 = sorted(frags[0])
            frag2: list[int] = []
        else:
            frag1 = sorted(frags[0])
            frag2 = sorted(frags[1])

        cap_sites_combined: list[tuple[int, int]] = []
        for sites in res.cap_sites.values():
            cap_sites_combined.extend(sites)
        h_caps = _h_caps_from_sites(cap_sites_combined, coords_R)

        out[rxn_id] = {
            "case": "B",  # placeholder; case classification preserved separately
            "method": f"stage0_be_matrix:{chosen_label}",
            "frag1_atoms": list(frag1),
            "frag2_atoms": list(frag2),
            "frag1_charge": 0,
            "frag2_charge": 0,
            "frag1_multiplicity": 1,
            "frag2_multiplicity": 1,
            "frag1_formula": _formula(numbers.tolist(), frag1),
            "frag2_formula": _formula(numbers.tolist(), frag2),
            "frag1_smiles": None,
            "frag2_smiles": None,
            "h_caps": h_caps,
            "reactive_bonds": [list(b) for b in res.reactive_bonds],
            "migrating_atoms": [
                {
                    "atom": int(m["atom"]),
                    "from": [int(x) for x in m["from"]],
                    "to": [int(x) for x in m["to"]],
                    "loss": int(m["loss"]),
                    "gain": int(m["gain"]),
                }
                for m in res.migrating_atoms
            ],
            "is_pure_rearrangement": bool(res.is_pure_rearrangement),
            "fallback_strategy": res.fallback_strategy,
            "auto_confidence": 0.5 if res.is_pure_rearrangement else 0.85,
            "review_status": (
                "needs_review" if res.is_pure_rearrangement else "auto_accepted"
            ),
            "rationale": "; ".join(res.notes) if res.notes else "BE-matrix split",
        }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT} with {len(out)} entries")
    print(f"Failures: {len(failures)}")
    print(f"Counter: {dict(counter)}")
    if failures[:5]:
        for rid, why in failures[:5]:
            print(f"  fail: {rid}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
