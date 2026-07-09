"""Translate manual_partitions.json (R-based) to orca_partitions.json (TS-based)
using the same family-aware R→TS mapping as make_orca_eda_inputs.py.

This lets the review app visualise the ACTUAL fragment assignment that goes
into the ORCA input file. Load with VIEW_GEOM=TS.
"""
from __future__ import annotations
import json, sys, importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_orca_eda_inputs as mkinp

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
MANUAL_PART = REPO / "outputs/frag_review/manual_partitions.json"
AUTO_PART = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/partitions.json")
COHORT_V7 = REPO / "outputs/frag_review/cohort_v7.parquet"
OUT = REPO / "outputs/frag_review/orca_partitions.json"


def main():
    import pandas as pd
    labels = pd.read_parquet(COHORT_V7)
    with open(MANUAL_PART) as f: manual = json.load(f)
    with open(AUTO_PART) as f: auto = json.load(f)

    out = {}
    n_ok = n_err = 0
    for row in labels.itertuples(index=False):
        rid, fam = row.reaction_id, row.family
        m = manual.get(rid)
        if not m or "frag_A_indices" not in m: continue
        try:
            A, B = mkinp.resolve_ts_fragments(rid, fam, manual, auto)
            out[rid] = {
                "frag_A_indices": list(A),
                "frag_B_indices": list(B),
                "reviewed": m.get("reviewed", False),
                "note": (m.get("note", "") + " [TS-native for ORCA]").strip(),
            }
            n_ok += 1
        except Exception as exc:
            out[rid] = {
                "frag_A_indices": [],
                "frag_B_indices": [],
                "reviewed": False,
                "note": f"[FAILED: {exc}]",
            }
            n_err += 1
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"ok={n_ok} err={n_err} → {OUT}")


if __name__ == "__main__":
    main()
