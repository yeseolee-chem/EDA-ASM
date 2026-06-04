#!/usr/bin/env python3
"""Classify the still-FAIL reactions by root cause.

Categories:
  schema       : Check-1 FAIL (missing/None values in asr_vector or fragment_opt)
  ts_not_max   : Check-3 ts_not_max (E_TS not maximum on R/TS/P triple)
  barrier      : Check-3 barrier_nonpositive (dE_act ≤ 0)
  cons_huge    : Check-4 res_cons > 50 kcal/mol (fundamental issue)
  cons_mid     : Check-4 res_cons 5–50 kcal/mol (deep but maybe salvageable)
  cons_small   : Check-4 res_cons 0.5–5 kcal/mol (close to passing)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(ROOT / "Validate"))
from validate_asr import (  # type: ignore
    Config, derive, check1_schema, check3_topology,
    check4_conservation, check5_signs, aggregate,
)
cfg = Config()


def main() -> int:
    """Print a categorized table of still-FAIL reactions."""
    # All originally failing rids
    orig_fails: list[str] = []
    for row in csv.DictReader(open(ROOT / "Validate/manifest.csv")):
        if row["verdict"] != "FAIL":
            continue
        if "3" in row["failed_checks"].split(";") or "4" in row["failed_checks"].split(";"):
            orig_fails.append(row["reaction_id"])

    # Classify each by current canonical winner
    rows: list[dict] = []
    for rid in orig_fails:
        rp = ROOT / "Validate" / "refrag" / "results" / f"{rid}.json"
        if not rp.exists():
            continue
        try:
            raw = json.loads(rp.read_text())
        except Exception:
            rows.append({"rid": rid, "cat": "schema", "detail": "result.json unreadable"})
            continue
        d = derive("w", raw)
        i1 = check1_schema(d)
        i3 = check3_topology(d, cfg)
        i4 = check4_conservation(d, cfg)
        verdict = aggregate(i1 + i3 + check4_conservation(d, cfg) + check5_signs(d))
        if verdict != "FAIL":
            continue

        # Pick the worst failure code
        fail_codes = [i.code for i in (i1 + i3 + i4) if i.level == "FAIL"]
        cat = "unknown"
        detail = ""
        if any(c in ("missing_id", "missing_point", "missing_vector",
                      "missing_component", "missing_fragment", "missing_setting",
                      "non_finite") for c in fail_codes):
            cat = "schema"
            detail = ";".join(fail_codes)
        elif "ts_not_max" in fail_codes:
            cat = "ts_not_max"
            detail = (f"E_R={d.E.get('R', float('nan')):.3f}, "
                      f"E_TS={d.E.get('TS', float('nan')):.3f}, "
                      f"E_P={d.E.get('P', float('nan')):.3f}")
        elif "barrier_nonpositive" in fail_codes:
            cat = "barrier"
            detail = f"dE_act={d.dE_act:.3f}"
        elif "conservation_fail" in fail_codes:
            r = d.max_abs_res_cons or 0.0
            if r > 50:
                cat = "cons_huge"
            elif r > 5:
                cat = "cons_mid"
            else:
                cat = "cons_small"
            detail = (f"res_cons={d.max_abs_res_cons:.3f}, "
                      f"res_ref={d.max_abs_res_ref:.3f}, "
                      f"spread={d.offset_spread:.3f}")
        rows.append({
            "rid": rid, "cat": cat, "detail": detail,
            "selected_candidate": raw.get("selected_candidate", "?"),
        })

    # Sort by category for readability
    cat_order = ["schema", "ts_not_max", "barrier",
                 "cons_huge", "cons_mid", "cons_small", "unknown"]
    rows.sort(key=lambda r: (cat_order.index(r["cat"]) if r["cat"] in cat_order else 99, r["rid"]))

    # Print
    print(f"\n=== {len(rows)} still-FAIL reactions ===\n")
    print(f"{'rid':42s}  {'cat':12s}  {'winner':22s}  detail")
    print("-" * 130)
    from collections import Counter
    cats = Counter()
    for r in rows:
        cats[r["cat"]] += 1
        print(f"{r['rid']:42s}  {r['cat']:12s}  {r['selected_candidate']:22s}  {r['detail']}")
    print()
    print("Category counts:")
    for c in cat_order:
        if cats[c]:
            print(f"  {c:12s} {cats[c]}")

    # Write JSON summary
    out = ROOT / "Validate" / "refrag" / "still_fail_diagnosis.json"
    out.write_text(json.dumps({
        "rows": rows,
        "category_counts": dict(cats),
    }, indent=2))
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
