#!/usr/bin/env python3
"""Interim Phase E look (user question 2026-10-29): the evaluate.py metrics on the test folds whose 3 final seeds are
already done (all 5 selects are finished, so nothing here can feed a choice). Report only; it never writes
E_report.json (the R launcher's 'evaluation done' marker).  GPU env:  python diag_interim.py
Writes results/E_interim.json."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as E  # noqa: E402
import geo_common as GC  # noqa: E402
from metrics import evaluate_preds  # noqa: E402

KEYS = ("primary_share", "n_primary", "pair_pe_median_same_para", "rxn_form_abs_median", "rmsd_heavy_median",
        "all_pairs_share", "n_pairs")


def main():
    recs = GC.load_records()
    folds = [k for k in range(5) if E.preds_for("model", [k]) is not None]
    out = dict(note="interim: test folds with all 3 final seeds only (report only)", folds=folds, per_fold={})
    rng = np.random.default_rng(E.SEED)
    for k in folds + ["all"]:
        fl = folds if k == "all" else [k]
        test = [r for r in recs if r["src"] == "D1" and r["fold"] in fl and r["xyz_g2"] is not None]
        T = {"g2_raw": {r["uid"]: r["xyz_g2"] for r in test}}
        for name in ("model", "seed0", "seed1", "seed2", "no_d0", "q1_only", "r0"):
            P = E.preds_for(name, fl)
            if P is not None:
                T[name] = P
        M = {n: evaluate_preds(P, [r for r in test if r["uid"] in P]) for n, P in T.items()}
        row = {n: {key: M[n][key] for key in KEYS} for n in M}
        row["ci"] = {f"model_vs_{n}": E.boot_diff(M["model"]["table_pairs"], M[n]["table_pairs"], rng)
                     for n in M if n not in ("model", "seed0", "seed1", "seed2")}
        out["per_fold"][str(k)] = row
        d_lo = row["ci"]["model_vs_g2_raw"]["ci_diff"][0]
        s = row["model"]["primary_share"]
        print(f"== fold {k}: n_test {len(test)}  spec rule -> "
              f"{'pass' if s >= 0.5 and d_lo > 0 else ('partial' if d_lo > 0 else 'fail')}", flush=True)
        for n in M:
            m = row[n]
            ci = row["ci"].get(f"model_vs_{n}")
            print(f"  {n:8s} primary {m['primary_share']:.3f} (n {m['n_primary']}) pair_med {m['pair_pe_median_same_para']:.4f}"
                  f" rxn_med {m['rxn_form_abs_median']:.4f} rmsd {m['rmsd_heavy_median']:.3f} all_pairs "
                  f"{m['all_pairs_share']:.3f}" + (f"  | model-{n} diff CI [{ci['ci_diff'][0]:+.3f}, {ci['ci_diff'][1]:+.3f}]"
                                                    if ci else ""), flush=True)
    GC.write_json(GC.RES / "E_interim.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
