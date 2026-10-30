#!/usr/bin/env python3
"""Interim (report only): G2-raw primary metric per D1 fold, to read the tune inner-validation scores against.
GPU env (CPU partition is fine):  python val_baseline.py  -> results/R_val_baseline.json"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
from metrics import evaluate_preds  # noqa: E402

recs = GC.load_records()
out = {}
for k in range(5):
    test = [r for r in recs if r["src"] == "D1" and r["fold"] == k and r["xyz_g2"] is not None]
    m = evaluate_preds({r["uid"]: r["xyz_g2"] for r in test}, test, with_rmsd=False)
    out[k] = dict(n_primary=m["n_primary"], g2raw_primary_share=m["primary_share"],
                  g2raw_rxn_form_abs_median=m["rxn_form_abs_median"])
    print(k, out[k], flush=True)
GC.write_json(GC.RES / "R_val_baseline.json", out)
