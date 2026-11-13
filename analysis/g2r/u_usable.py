#!/usr/bin/env python3
"""D1-G2R Phase U (docs/specs/D1G2R.md 10): the usable list.

    python analysis/g2r/u_usable.py          env_d1fu2 (pandas + pyarrow); py_g2r.slurm

usable = target (results/X_targets.csv, 9,252) and L .done ($G2R/lf/g2lf_final/<rid>/.done) and a row of
$G2R/lf/X182_lf.parquet (feat_lf merge keeps only F_ok rows, all 182 finite) and a token (row of
results/lf/fu2_lf/T_manifest.csv and $G2R/lf/fu2_lf/tokens/<rid>.npz).
Outputs: results/U_usable.csv (rxn_id, job_id, usable, missing_stage, reason), results/U_funnel.csv (the spec 0-4 rows),
results/U_report.json (usable count, sha256 of labels_lf.json, input_meta_lf.csv, X182_lf.parquet, the token manifest).
U-1: usable != 8,614 + k (k = repaired ok) -> the per-stage cause table in U_report (report, not STOP). Exit 0.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2r_common as C  # noqa: E402


def main() -> int:
    import pandas as pd
    X = pd.read_csv(C.RES / "X_targets.csv", dtype={"job_id": str})
    xr = C.read_json(C.RES / "X_report.json")
    rr = C.read_json(C.RES / "R_report.json")
    R = pd.read_csv(C.RES / "R_status.csv", dtype={"job_id": str}).set_index("job_id")
    fr = C.read_json(C.RES / "lf" / "F_report.json")
    x182 = set(pd.read_parquet(C.G2R_LF / "X182_lf.parquet", columns=["rxn_id"]).rxn_id.astype(int))
    tm_p = C.RES / "lf" / "fu2_lf" / "T_manifest.csv"
    tm = set(pd.read_csv(tm_p).rxn_id.astype(int))
    qm = pd.read_csv(C.RES / "lf" / "fu2_lf" / "Q_manifest.csv").set_index("rxn_id")
    rows = []
    for t in X.itertuples():
        rid, j = int(t.rxn_id), t.job_id
        d = C.G2R_LF / "g2lf_final" / str(rid)
        lf_m = ".done" if (d / ".done").is_file() else (sorted(p.name for p in d.glob(".fail_*"))[:1] or [None])[0]
        tok = rid in tm and (C.G2R_LF / "fu2_lf" / "tokens" / f"{rid}.npz").is_file()
        if t.group in ("excluded_ts", "excluded_check"):
            stage, why = "X", f"{t.group}: {t.g2_marker}"
        elif t.group == "repair" and (j not in R.index or R.at[j, "new_status"] != "ok"):
            stage, why = "R", f"repair failed: {R.at[j, 'new_reason'] if j in R.index else 'no record'}"
        elif lf_m != ".done":
            stage, why = "L", f"L marker {lf_m}"
        elif rid not in x182:
            stage, why = "F", (fr.get("not_ok") or {}).get(str(rid), "not in X182_lf")
        elif not tok:
            if rid in qm.index:
                q = qm.loc[rid]
                why = (f"use={q.get('use')} ext_ok={q.get('ext_ok')} qa_ok={q.get('qa_ok')} qb_ok={q.get('qb_ok')} "
                       f"{str(q.get('ext_reason') or '')[:120]}")
            else:
                why = "no Q_manifest row"
            stage = "T"
        else:
            stage, why = "", ""
        rows.append(dict(rxn_id=rid, job_id=j, usable=stage == "", missing_stage=stage, reason=why))
    U = pd.DataFrame(rows)
    buf = io.StringIO()
    U.to_csv(buf, index=False)
    C.write_atomic(C.RES / "U_usable.csv", buf.getvalue())
    n = lambda s: int((U.missing_stage == s).sum())                           # noqa: E731
    g = X.group.value_counts().to_dict()
    k = int(rr.get("k_ok") or 0)
    x2 = xr["gates"]["X-2"]
    steps = [("D1 non-D'' labels", 0, xr["gates"]["X-1"]["n_non_dpp"]),
             ("short_forming, regio_ambiguous", x2["short_forming"] + x2["regio_ambiguous"] - x2["overlap"], x2["n_targets"]),
             ("reference excess > 20, hygiene (cr2, tolerant)", xr["gates"]["X-6"]["n_ex_new_gt20"]
              + xr["gates"]["X-6"]["hygiene_tolerant_fail"], None),
             ("g2 D-6: no TS, autodE error, stereo mismatch, no imaginary frequency", g.get("excluded_ts", 0), None),
             ("g2 D-4: TS check, partition", g.get("excluded_check", 0), None),
             ("g2 reference code errors (D-3), 20 - k", n("R"), None),
             ("L frequency finalisation", n("L"), None), ("F 182 columns", n("F"), None), ("T atom tokens", n("T"), None)]
    left, fun = None, []
    for name, out, rem in steps:
        left = rem if rem is not None else left - out
        fun.append(dict(stage=name, removed=out, remaining=left))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["stage", "removed", "remaining"])
    w.writeheader()
    w.writerows(fun)
    C.write_atomic(C.RES / "U_funnel.csv", buf.getvalue())
    n_use = int(U.usable.sum())
    want = 8614 + k
    files = {"labels_lf.json": C.G2R_LF / "labels_lf.json", "input_meta_lf.csv": C.G2R_LF / "input_meta_lf.csv",
             "X182_lf.parquet": C.G2R_LF / "X182_lf.parquet", "T_manifest.csv": tm_p}
    rep = dict(spec="docs/specs/D1G2R.md 10", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), n_targets=int(len(U)), n_usable=n_use,
               usable_fraction=n_use / len(U), k_repaired_ok=k, expected=want, U1_equal_expected=n_use == want,
               missing_by_stage={s: n(s) for s in ("X", "R", "L", "F", "T")},
               cause_table=(U[~U.usable].groupby(["missing_stage", "reason"]).size().reset_index(name="n")
                            .sort_values(["missing_stage", "n"], ascending=[True, False]).to_dict("records")
                            if n_use != want else None),
               sha256={k_: C.sha256(p) for k_, p in files.items()}, paths={k_: str(p) for k_, p in files.items()},
               token_dir=str(C.G2R_LF / "fu2_lf" / "tokens"), funnel="results/U_funnel.csv", all_pass=True)
    C.write_json(C.RES / "U_report.json", rep)
    print(json.dumps({k_: v for k_, v in rep.items() if k_ != "cause_table"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
