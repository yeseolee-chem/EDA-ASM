#!/usr/bin/env python3
"""Precompute the 62 reactions into one data.json the viz server serves."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Validate"))

from validate_asr import (  # type: ignore
    Config, derive, check1_schema, check3_topology, check4_conservation,
    check5_signs, aggregate,
)

SYM = {1:"H", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 15:"P",
       16:"S", 17:"Cl", 35:"Br", 53:"I"}


def _numbers_for(cache, rid):
    """Return atomic numbers list for one reaction; None if not in cache."""
    e = cache.get(rid)
    if e is None:
        return None
    nums = e.numbers
    try:
        return [int(n) for n in nums]
    except TypeError:
        return list(nums)


def _positions_for(cache, rid, label):
    """Return positions (Å) array (n×3 list) for R/TS/P or None."""
    e = cache.get(rid)
    if e is None:
        return None
    arr = {"R": e.positions_R, "TS": e.positions_TS, "P": e.positions_P}.get(label)
    if arr is None:
        return None
    try:
        return [[float(x), float(y), float(z)] for x, y, z in arr.reshape(-1, 3)]
    except Exception:
        try:
            return [[float(a) for a in row] for row in arr]
        except Exception:
            return None


def _failing_rids() -> list[str]:
    """All rids whose verdict==FAIL and Check 3 or 4 failed."""
    rids: list[str] = []
    for row in csv.DictReader(open(ROOT / "Validate/manifest.csv")):
        if row["verdict"] != "FAIL":
            continue
        fc = set(row["failed_checks"].split(";"))
        if "3" in fc or "4" in fc:
            rids.append(row["reaction_id"])
    return rids


def _derive_for(raw: dict) -> dict:
    """Re-derive validation metrics from a result JSON (or return Nones)."""
    cfg = Config()
    d = derive("memory.json", raw)
    issues = (check1_schema(d) + check3_topology(d, cfg)
              + check4_conservation(d, cfg) + check5_signs(d))
    verdict = aggregate(issues)
    return {
        "dE_act": d.dE_act,
        "dE_rxn": d.dE_rxn,
        "sigma_R": d.sigma.get("R"),
        "sigma_TS": d.sigma.get("TS"),
        "sigma_P": d.sigma.get("P"),
        "E_frag": d.E_frag,
        "max_abs_res_cons": d.max_abs_res_cons,
        "max_abs_res_ref": d.max_abs_res_ref,
        "offset_spread": d.offset_spread,
        "verdict": verdict,
        "failed_check_nos": sorted({i.check_no for i in issues if i.level == "FAIL"}),
        "warn_codes": sorted({i.code for i in issues if i.level == "WARN"}),
    }


def main() -> None:
    """Build Validate/viz/data.json with everything the HTML page needs."""
    rids = _failing_rids()
    print(f"Targeting {len(rids)} reactions")

    with open(ROOT / "ADF_500/stage5a/frames_cache.pkl", "rb") as fh:
        cache = pickle.load(fh)
    print(f"Loaded frames_cache: {len(cache)} entries")

    # Still-FAIL diagnostic categories (if present).
    diag_path = ROOT / "Validate" / "refrag" / "still_fail_diagnosis.json"
    diag_by_rid: dict[str, dict] = {}
    if diag_path.exists():
        try:
            diag = json.loads(diag_path.read_text())
            for r in diag.get("rows", []):
                diag_by_rid[r["rid"]] = r
        except Exception:
            pass

    out: list[dict] = []
    for rid in rids:
        family = "T1x" if rid.startswith("T1x") else "Halogen"
        stage5a = json.loads((ROOT / f"ADF_500/stage5a/per_reaction/{rid}/result.json").read_text())
        old_raw = json.loads((ROOT / f"ADF_500/results/{rid}.json").read_text())
        old_derived = _derive_for(old_raw)

        alt_path = ROOT / f"Validate/refrag/stage5a/per_reaction/{rid}/result.json"
        alt_stage5a = json.loads(alt_path.read_text()) if alt_path.exists() else None

        new_path = ROOT / f"Validate/refrag/results/{rid}.json"
        new_derived = None
        new_status = None
        if new_path.exists():
            try:
                new_raw = json.loads(new_path.read_text())
                new_derived = _derive_for(new_raw)
                new_status = new_raw.get("status_at_queue")
            except Exception as exc:
                new_derived = {"error": str(exc)}

        numbers = _numbers_for(cache, rid)
        symbols = [SYM.get(z, f"Z{z}") for z in numbers] if numbers else None

        out.append({
            "rid": rid,
            "family": family,
            "pattern": stage5a["result"]["pattern"],
            "n_atoms": stage5a["n_atoms"],
            "ts_frame_idx": stage5a.get("ts_frame_idx"),
            "frame_first": stage5a.get("frame_index_first"),
            "frame_last": stage5a.get("frame_index_last"),
            "halo8_Ea_eV": stage5a.get("activation_energy"),
            "orig": {
                "fragments": [{
                    "role": f["role"],
                    "atom_indices": f["atom_indices"],
                    "multiplicity": f["multiplicity"],
                } for f in stage5a["result"]["fragments"]],
                "n_fragments": stage5a["result"]["n_fragments"],
                "asr_vector": old_raw.get("asr_vector_kcal"),
                "fragment_opt_energy_kcal": old_raw.get("fragment_opt_energy_kcal"),
                "status": old_raw.get("status_at_queue"),
                "schema_version": old_raw.get("schema_version"),
                "derived": old_derived,
            },
            "alt_available": alt_stage5a is not None,
            "alt": (None if alt_stage5a is None else {
                "fragments": [{
                    "role": f["role"],
                    "atom_indices": f["atom_indices"],
                    "multiplicity": f["multiplicity"],
                } for f in alt_stage5a["result"]["fragments"]],
                "coupling": alt_stage5a["result"]["coupling"],
                "n_fragments": alt_stage5a["result"]["n_fragments"],
                "pattern_tag": alt_stage5a["result"]["pattern"],
            }),
            "new_status": new_status,
            "new": new_derived,
            "bonds": {
                "R": stage5a["debug"]["bonds_R"],
                "P": stage5a["debug"]["bonds_P"],
                "broken": stage5a["debug"]["bonds_broken"],
                "formed": stage5a["debug"]["bonds_formed"],
            },
            "core_atoms": stage5a["debug"].get("core_atoms", []),
            "migrating_atoms": stage5a["debug"].get("migrating_atoms", []),
            "symbols": symbols,
            "positions_R": _positions_for(cache, rid, "R"),
            "positions_TS": _positions_for(cache, rid, "TS"),
            "positions_P": _positions_for(cache, rid, "P"),
            "still_fail_category": diag_by_rid.get(rid, {}).get("cat"),
            "still_fail_detail": diag_by_rid.get(rid, {}).get("detail"),
        })

    target = ROOT / "Validate/viz/data.json"
    target.write_text(json.dumps({
        "n_total": len(out),
        "reactions": out,
    }, indent=2))
    print(f"Wrote {target} ({len(out)} reactions)")

    # Summary
    n_alt = sum(1 for r in out if r["alt_available"])
    n_new = sum(1 for r in out if r["new"] is not None and r["new"].get("verdict") is not None)
    n_fixed = sum(1 for r in out if r["new"] and r["new"].get("verdict") in ("PASS", "WARN")
                                     and r["orig"]["derived"].get("verdict") == "FAIL")
    print(f"  alt fragmentation available : {n_alt}/{len(out)}")
    print(f"  re-run completed            : {n_new}/{n_alt}")
    print(f"  newly PASS/WARN (was FAIL)  : {n_fixed}")


if __name__ == "__main__":
    main()
