#!/usr/bin/env python3
"""Post-process work/labels_all.json → project-root labels_all.json.

Applies the two policy decisions the pipeline needs (kept from the pre-fix
workflow, see TRANSFER.md § "미결 결정 두 가지"):
  1. Relax the |sum(6ch) - Bond Energy| threshold: promote status=eda_sum_mismatch
     to status=ok. Residual is kept in the record for downstream re-filtering.
  2. Exclude 5 rxns for physical/quality reasons and mark status=excluded:
     3090, 3766, 4252 (flag_foreign_bond — TS mismatched with fragments)
     3400, 5783        (oi_dft > 0 — nonsensical orbital interaction)

Writes labels_all.json to the repo root next to CLAUDE.md.
"""
import json
from pathlib import Path

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SRC  = REPO / "label_true" / "work" / "labels_all.json"
DST  = REPO / "labels_all.json"

EXCLUDED = {
    3090: "flag_foreign_bond",
    3766: "flag_foreign_bond",
    4252: "flag_foreign_bond",
    3400: "oi_dft>0",
    5783: "oi_dft>0",
}


def main():
    data = json.load(open(SRC))
    from collections import Counter
    before = Counter(r["status"] for r in data)
    promoted = excluded = 0
    for r in data:
        rid = int(r["rxn_id"])
        if rid in EXCLUDED:
            r["status"] = "excluded"
            r["exclusion_reason"] = EXCLUDED[rid]
            excluded += 1
        elif r["status"] == "eda_sum_mismatch":
            r["status"] = "ok"
            r["sum_mismatch_promoted"] = True     # provenance: was above 0.02 tolerance
            promoted += 1
    after = Counter(r["status"] for r in data)
    print(f"pre  status counts: {dict(before)}")
    print(f"post status counts: {dict(after)}")
    print(f"promoted eda_sum_mismatch -> ok : {promoted}")
    print(f"excluded (foreign_bond / oi>0)  : {excluded}")
    json.dump(data, open(DST, "w"), indent=2)
    print(f"wrote {DST} ({len(data)} records)")


if __name__ == "__main__":
    main()
