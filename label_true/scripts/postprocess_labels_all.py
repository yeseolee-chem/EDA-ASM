#!/usr/bin/env python3
"""Post-process work/labels_all.json → project-root labels_all.json.

Applies the two policy decisions the pipeline needs (kept from the pre-fix
workflow, see TRANSFER.md § "미결 결정 두 가지"):
  1. Relax the |sum(6ch) - Bond Energy| threshold: promote status=eda_sum_mismatch
     to status=ok. Residual is kept in the record for downstream re-filtering.
  2. Exclude 5 rxns for physical/quality reasons and mark status=excluded:
     3090, 3766, 4252 (flag_foreign_bond — TS mismatched with fragments)
     3400, 5783        (no_forming_bond_ts — both forming bonds >= 3.3 A at the
                        TS, so it is not a bond-forming cycloaddition TS; every
                        accepted rxn has its shorter forming bond <= 3.18 A.
                        The earlier CPCM-era reason, oi_dft > 0, does not hold
                        with the SMD labels.)
     Excluded records keep their stage3 status as status_stage3.

Writes labels_all.json to the repo root next to CLAUDE.md.

Usage: postprocess_labels_all.py [SRC [DST]]   (defaults below). SRC defaults to
the SMD build label_true/work_smd/labels_all.json; label_true/work/labels_all.json
is the CPCM-era build and must not reach the repo root. DST is written atomically.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SRC  = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "label_true" / "work_smd" / "labels_all.json"
DST  = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "labels_all.json"

EXCLUDED = {
    3090: "flag_foreign_bond",
    3766: "flag_foreign_bond",
    4252: "flag_foreign_bond",
    3400: "no_forming_bond_ts",
    5783: "no_forming_bond_ts",
}
NO_BOND_A = 3.3        # shorter forming bond at or above this: no bond is forming at the TS


def main():
    data = json.load(open(SRC))
    from collections import Counter
    before = Counter(r["status"] for r in data)
    promoted = excluded = 0
    for r in data:
        rid = int(r["rxn_id"])
        if rid in EXCLUDED:
            r["status_stage3"] = r["status"]
            r["status"] = "excluded"
            r["exclusion_reason"] = EXCLUDED[rid]
            if EXCLUDED[rid] == "no_forming_bond_ts":
                d = sorted((float(r["formed_d1"]), float(r["formed_d2"])))
                if d[0] < NO_BOND_A:
                    print(f"WARNING: rxn {rid} shorter forming bond {d[0]:.2f} A < {NO_BOND_A}")
                r["exclusion_detail"] = (f"both forming bonds >= {NO_BOND_A} A at the TS "
                                         f"({d[0]:.2f} / {d[1]:.2f} A): not a bond-forming TS")
            excluded += 1
        elif r["status"] == "eda_sum_mismatch":
            r["status"] = "ok"
            r["sum_mismatch_promoted"] = True     # provenance: was above 0.02 tolerance
            promoted += 1
    # the no-forming-bond rule must not also describe an accepted rxn
    same = [int(r["rxn_id"]) for r in data if r["status"] != "excluded"
            and min(float(r["formed_d1"]), float(r["formed_d2"])) >= NO_BOND_A]
    if same:
        print(f"WARNING: accepted rxns with both forming bonds >= {NO_BOND_A} A: {same}")
    after = Counter(r["status"] for r in data)
    print(f"pre  status counts: {dict(before)}")
    print(f"post status counts: {dict(after)}")
    print(f"promoted eda_sum_mismatch -> ok : {promoted}")
    print(f"excluded (foreign_bond / no_forming_bond_ts): {excluded}")
    tmp = DST.with_name(DST.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, DST)
    print(f"wrote {DST} ({len(data)} records)")


if __name__ == "__main__":
    main()
