#!/usr/bin/env python3
"""D1-G2R Phase F checks (docs/specs/D1G2R.md 8-3) after feat_lf.py slice @I 2 + merge (g_g2r.slurm) -> results/F_check.json,
results/F_newly_passed.csv. Before Phase D ($CH3/lf is read).

    python analysis/g2r/f_check.py          env_d1fu2 (pandas + pyarrow); py_g2r.slurm

F-1  X182_lf.parquet column names and order = $CH3/lf/X182_lf.parquet (and feat_lf's own F_report without STOP). STOP
F-2  the tolerance records results/F_tol/<tag>.json of every F job (2 slice tasks + merge; f_preflight.py, job process +
     spawn worker + feat_lf-style child) show GATE_E_EH = 1e-4, TOL_DIP_AU = 1e-2, Si patch on, other tolerances
     unchanged; and no ext failure in the merged ext table sits below the new tolerances (engine_gate1 value < 1e-4 or a
     dipole-sum difference < 1e-2 would mean a worker ran without the patch).                              STOP
F-3  feature failures (F_report F_fail_rate = all non-F_ok .done reactions / .done reactions) <= 1 %.     STOP
F-R  the reactions that passed before (CH3 X182_lf rows that are targets: 8,502): every row is present and
     max |new - old| over the 182 columns <= 1e-6.                                                          STOP
F-N  newly passing reactions (new X182 rows not in the CH3 X182): expected = the 112 CH3 failures + repaired k. Per
     reaction the engine gate value max |qc_b1_dE_{fA,fB,ts}_eh| and the dipole check qc_b3_dip_check_au (fA / fB;
     the rel1 / rel2 checks are not stored by ext_features) -> results/F_newly_passed.csv; the CH3 failures that
     still fail are listed with their new reason.                                                           report
F-D  over every reaction of the merged ext table: max engine gate value, dipole check median / 99 % / max.    report
Exit 0 / 3.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2r_common as C  # noqa: E402

TOL_R = 1e-6
F3_MAX = 0.01
NEW_GATE_E, NEW_DIP = 1e-4, 1e-2
EXT = C.G2R_LF / "F" / "xtb" / "xtb_features_ext_g1_lf.parquet"
RE_GATE = re.compile(r"engine_gate1:([0-9.eE+-]+)")
RE_DIP = re.compile(r"differs from .* by ([0-9.eE+-]+) a\.u\.")
QC = ["qc_b1_dE_fA_eh", "qc_b1_dE_fB_eh", "qc_b1_dE_ts_eh"]


def ids(name):
    with open(C.G2R_X / name, newline="") as fh:
        return [r["job_id"] for r in csv.DictReader(fh)]


def main() -> int:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    X = pd.read_csv(C.RES / "X_targets.csv", dtype={"job_id": str})
    tgt = set(X.rxn_id.astype(int))
    job_of = dict(zip(X.rxn_id.astype(int), X.job_id))
    rep_ok = set(ids("repaired_ok_ids.csv"))
    fr = C.read_json(C.RES / "lf" / "F_report.json")
    old_fr = C.read_json(C.LF_REPORTS / "F_report.json")
    new_p, old_p = C.G2R_LF / "X182_lf.parquet", C.CH3_LF_OLD / "X182_lf.parquet"
    # F-1
    cn, co = pq.read_schema(new_p).names, pq.read_schema(old_p).names
    f1 = dict(n_columns=len(cn), columns_equal=list(cn) == list(co), feat_lf_stop=fr.get("stop"),
              feat_lf_stop_reasons=fr.get("stop_reasons"), ok=list(cn) == list(co) and fr.get("stop") is False)
    # F-2
    tol = sorted((C.RES / "F_tol").glob("*.json")) if (C.RES / "F_tol").is_dir() else []
    recs = [C.read_json(p) for p in tol]
    f_recs = [r for r in recs if str(r.get("tag", "")).startswith(("g2r_F_", "g2r_Fm_"))]
    ok_recs = [r for r in f_recs if r.get("ok")]          # a failed preflight exits before any compute (g_g2r.slurm)
    failed_pre = sorted(str(r.get("tag")) for r in f_recs if not r.get("ok"))
    slice_tasks = sorted({str(r.get("slurm_array_task")) for r in ok_recs if str(r.get("tag", "")).startswith("g2r_F_")})
    merge_n = sum(str(r.get("tag", "")).startswith("g2r_Fm_") for r in ok_recs)
    e = pd.read_parquet(EXT)
    st = e.ext_status.astype(str) if "ext_status" in e else pd.Series([], dtype=str)
    below = []
    for rid, s in zip(e.rxn_id.astype(int), st):
        m1, m2 = RE_GATE.search(s), RE_DIP.search(s)
        if m1 and float(m1.group(1)) < NEW_GATE_E:
            below.append(dict(rxn_id=rid, status=s[:160]))
        if m2 and float(m2.group(1)) < NEW_DIP:
            below.append(dict(rxn_id=rid, status=s[:160]))
    f2 = dict(records=[p.name for p in tol], n_f_records=len(f_recs), n_ok_records=len(ok_recs),
              slice_tasks_ok=slice_tasks, n_merge_records_ok=merge_n,
              failed_preflight_records=failed_pre, tolerances=(ok_recs[0]["tolerances"] if ok_recs else None),
              n_ext_fail_below_new_tolerance=len(below), ext_fail_below_new_tolerance=below[:20],
              rule="an ok preflight record for slice tasks 0 and 1 and the merge (failed preflights computed nothing)",
              ok=slice_tasks == ["0", "1"] and merge_n >= 1 and not below)
    # F-3
    f3 = dict(n_done=fr.get("n_done"), n_feat=fr.get("n_feat"), F_fail_rate=fr.get("F_fail_rate"), max=F3_MAX,
              feat_lf_F3=fr.get("F3"), reasons=fr.get("reasons"),
              ok=fr.get("F_fail_rate") is not None and fr["F_fail_rate"] <= F3_MAX)
    # F-R
    new, old = pd.read_parquet(new_p).set_index("rxn_id"), pd.read_parquet(old_p).set_index("rxn_id")
    old_t = sorted(r for r in old.index.astype(int) if r in tgt)
    miss = [r for r in old_t if r not in new.index]
    both = [r for r in old_t if r in new.index]
    cols = [c for c in co if c != "rxn_id"]
    d = np.abs(new.loc[both, cols].to_numpy(float) - old.loc[both, cols].to_numpy(float))
    per_row = d.max(axis=1) if len(both) else np.zeros(0)
    worst = sorted(zip(per_row.tolist(), both), reverse=True)[:10]
    col_worst = d.max(axis=0) if len(both) else np.zeros(0)
    fr_ = dict(n_old_pass_targets=len(old_t), n_missing_in_new=len(miss), missing=miss[:20], n_compared=len(both),
               max_abs_diff=float(d.max()) if d.size else 0.0, n_rows_gt_tol=int((per_row > TOL_R).sum()),
               worst_rows=[dict(rxn_id=r, max_abs_diff=v) for v, r in worst],
               worst_columns=sorted(([float(v), c] for v, c in zip(col_worst, cols)), reverse=True)[:10], tol=TOL_R,
               ok=not miss and len(both) == len(old_t) and (float(d.max()) if d.size else 0.0) <= TOL_R)
    # F-N
    old_not_ok = {int(k): v for k, v in (old_fr.get("not_ok") or {}).items()}
    old_fail_t = sorted(r for r in old_not_ok if r in tgt)
    new_ids = set(new.index.astype(int))
    newly = sorted(r for r in new_ids if r not in set(old.index.astype(int)))
    ee = e.set_index(e.rxn_id.astype(int))
    rows = []
    for r in sorted(set(newly) | set(old_fail_t)):
        g = ee.loc[r] if r in ee.index else None
        gate = float(np.nanmax(np.abs(g[QC].to_numpy(float)))) if g is not None and all(c in ee for c in QC) else None
        dip = float(g["qc_b3_dip_check_au"]) if g is not None and "qc_b3_dip_check_au" in ee else None
        old_r = old_not_ok.get(r, "")
        rows.append(dict(rxn_id=r, job_id=job_of.get(r), group="repaired" if job_of.get(r) in rep_ok else "reuse",
                         old_reason=old_r, new_pass=r in new_ids,
                         new_reason=(fr.get("not_ok") or {}).get(str(r), "ok" if r in new_ids else ""),
                         engine_gate_max_abs_eh=gate, dip_check_fA_fB_au=dip,
                         note=("old dipole failure on rel1: the rel1 dipole check value is not stored by ext_features "
                               "(it passed: < 1e-2)" if "rel1" in old_r and r in new_ids else "")))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ["rxn_id"])
    w.writeheader()
    w.writerows(rows)
    C.write_atomic(C.RES / "F_newly_passed.csv", buf.getvalue())
    still = [x for x in rows if x["old_reason"] and not x["new_pass"]]
    unexpected = [x["rxn_id"] for x in rows if x["new_pass"] and not x["old_reason"] and x["group"] != "repaired"]
    fn = dict(n_old_fail_targets=len(old_fail_t), n_newly_passed=len(newly),
              n_newly_from_old_fail=sum(1 for x in rows if x["old_reason"] and x["new_pass"]),
              n_newly_repaired=sum(1 for x in rows if x["group"] == "repaired" and x["new_pass"]),
              n_newly_other=len(unexpected), newly_other=unexpected[:20], n_old_fail_still_fail=len(still),
              old_fail_still_fail=[{k: x[k] for k in ("rxn_id", "old_reason", "new_reason", "engine_gate_max_abs_eh",
                                                       "dip_check_fA_fB_au")} for x in still],
              table="results/F_newly_passed.csv", ok=True, note="report only (spec F-N)")
    # F-D
    gate_all = np.nanmax(np.abs(e[QC].to_numpy(float)), axis=1) if all(c in e for c in QC) else np.zeros(0)
    gate_all = gate_all[np.isfinite(gate_all)]
    dip_all = e["qc_b3_dip_check_au"].to_numpy(float) if "qc_b3_dip_check_au" in e else np.zeros(0)
    dip_all = dip_all[np.isfinite(dip_all)]
    q = (lambda v, p: float(np.quantile(v, p)) if len(v) else None)
    fd = dict(n_engine_gate=int(len(gate_all)), engine_gate_max_eh=float(gate_all.max()) if len(gate_all) else None,
              engine_gate_median_eh=q(gate_all, 0.5), engine_gate_p99_eh=q(gate_all, 0.99),
              engine_gate_max_kcal=float(gate_all.max()) * 627.5094740631 if len(gate_all) else None,
              n_dip_check=int(len(dip_all)), dip_check_median_au=q(dip_all, 0.5), dip_check_p99_au=q(dip_all, 0.99),
              dip_check_max_au=float(dip_all.max()) if len(dip_all) else None,
              source=f"{EXT} (qc_b1_dE_* = tblite - xtb per structure fA/fB/TS; qc_b3_dip_check_au = max over fA/fB, "
                     "neutral only)", ok=True, note="report only (spec F-D)")
    gates = {"F-1": f1, "F-2": f2, "F-3": f3, "F-R": fr_, "F-N": fn, "F-D": fd}
    rep = dict(spec="docs/specs/D1G2R.md 8-3", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"),
               inputs={str(new_p): C.sha256(new_p), str(old_p): C.sha256(old_p), str(EXT): C.sha256(EXT),
                       "results/lf/F_report.json": C.sha256(C.RES / "lf" / "F_report.json")},
               gates=gates, all_pass=all(g["ok"] for g in gates.values()))
    C.write_json(C.RES / "F_check.json", rep)
    print(json.dumps(dict(all_pass=rep["all_pass"], **{k: g["ok"] for k, g in gates.items()},
                          F_fail_rate=f3["F_fail_rate"], max_abs_diff=fr_["max_abs_diff"], newly=fn["n_newly_passed"],
                          still=fn["n_old_fail_still_fail"]), indent=1))
    return 0 if rep["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
