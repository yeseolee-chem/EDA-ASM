#!/usr/bin/env python3
"""D1-G2R Phase L checks (docs/specs/D1G2R.md 6) after deploy finalize_lf.py run (lf_ids) and report (reuse_ids).

    python analysis/g2r/l_check.py          stdlib only (py_g2r.slurm); before Phase D ($CH3/lf is read)

L-1  results/lf/L_report.json (deploy L-1, L-2, L-4 on the reuse reactions): complete, all_pass, subset = sorted reuse
     job_ids.                                                                                              STOP
L-R  every reuse reaction: $G2R/lf/g2lf_final/<rid>/{ts,rel1,rel2}.xyz byte-equal to $CH3/lf/g2lf_final/<rid>/, and
     |lowest imaginary frequency (new) - (old)| <= 1e-3 cm-1 from the two result.json files.              STOP
L-K  repaired reactions (repaired_ok_ids): marker; if .done, n_imag >= 1 and imag_mode_forming_share
     -> results/L_repaired.csv (deploy L-2 / L-4 do not apply: no $G2/g2_final structure to compare).  report
Also: every lf_ids reaction has a marker in $G2R/lf/g2lf_final. Output results/L_check.json. Exit 0 / 2 (not complete,
or results/lf/L_report.json missing or incomplete) / 3 (STOP).
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2r_common as C  # noqa: E402

TOL_IMAG = 1e-3
FILES = ("ts.xyz", "rel1.xyz", "rel2.xyz")
NEW = C.G2R_LF / "g2lf_final"
OLD = C.CH3_LF_OLD / "g2lf_final"
LRES = C.RES / "lf"


def ids(path):
    with open(path, newline="") as fh:
        return [r["job_id"] for r in csv.DictReader(fh)]


def marker(d: Path):
    if (d / ".done").is_file():
        return ".done"
    f = sorted(p.name for p in d.glob(".fail_*")) if d.is_dir() else []
    return f[0] if f else None


def lowest(res):
    v = [float(x) for x in (res.get("imag_freqs_cm") or [])]
    return min(v) if v else None


def main() -> int:
    with open(C.RES / "X_targets.csv", newline="") as fh:
        X = {r["job_id"]: r for r in csv.DictReader(fh)}
    reuse = ids(C.G2R_X / "reuse_ids.csv")
    rep_ok = ids(C.G2R_X / "repaired_ok_ids.csv")
    lf_ids = ids(C.G2R_X / "lf_ids.csv")
    rid = {j: X[j]["rxn_id"] for j in X}
    missing_marker = [j for j in lf_ids if marker(NEW / rid[j]) is None]
    # L-1
    lrep = C.read_json(LRES / "L_report.json") if (LRES / "L_report.json").is_file() else {}
    l1 = dict(path=str(LRES / "L_report.json"), complete=lrep.get("complete"), all_pass=lrep.get("all_pass"),
              subset_is_reuse=lrep.get("subset") == sorted(reuse), n_targets=lrep.get("n_targets"),
              lf_markers=lrep.get("lf_markers"), L1_pass=(lrep.get("L1") or {}).get("pass"),
              L2_pass=(lrep.get("L2") or {}).get("pass"), L4_pass=(lrep.get("L4") or {}).get("pass"))
    l1["ok"] = bool(lrep.get("complete") and lrep.get("all_pass") and l1["subset_is_reuse"])
    # L-R
    n_cmp, bad_bytes, bad_imag, not_done, dmax = 0, [], [], [], 0.0
    for j in reuse:
        r = rid[j]
        a, b = NEW / r, OLD / r
        if marker(a) != ".done" or marker(b) != ".done":
            not_done.append(dict(job_id=j, rxn_id=r, new=marker(a), old=marker(b)))
            continue
        n_cmp += 1
        diff = [f for f in FILES if (a / f).read_bytes() != (b / f).read_bytes()]
        if diff:
            bad_bytes.append(dict(rxn_id=r, files=diff))
        fa, fb = lowest(C.read_json(a / "result.json")), lowest(C.read_json(b / "result.json"))
        d = abs(fa - fb) if fa is not None and fb is not None else math.inf
        dmax = max(dmax, d)
        if not d <= TOL_IMAG:
            bad_imag.append(dict(rxn_id=r, new=fa, old=fb, diff=d))
    lr = dict(n_reuse=len(reuse), n_compared=n_cmp, n_not_done=len(not_done), not_done=not_done[:20],
              n_bytes_differ=len(bad_bytes), bytes_differ=bad_bytes[:20], n_imag_differ=len(bad_imag),
              imag_differ=bad_imag[:20], max_abs_d_lowest_imag_cm=dmax, tol_cm=TOL_IMAG,
              ok=n_cmp == len(reuse) and not bad_bytes and not bad_imag)
    # L-K
    rows = []
    for j in rep_ok:
        r = rid[j]
        d = NEW / r
        m = marker(d)
        res = C.read_json(d / "result.json") if (d / "result.json").is_file() else {}
        n_imag = res.get("n_imag")
        rows.append(dict(job_id=j, rxn_id=r, lf_marker=m, lf_status=res.get("status"), n_imag=n_imag,
                         imag1_cm=lowest(res), imag_mode_forming_share=res.get("imag_mode_forming_share"),
                         n_imag_ge_1=(m == ".done" and n_imag is not None and n_imag >= 1) if m == ".done" else ""))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["job_id", "rxn_id", "lf_marker", "lf_status", "n_imag", "imag1_cm",
                                        "imag_mode_forming_share", "n_imag_ge_1"])
    w.writeheader()
    w.writerows(rows)
    C.write_atomic(C.RES / "L_repaired.csv", buf.getvalue())
    lk = dict(n_repaired_ok=len(rep_ok), markers={m: sum(str(r["lf_marker"]) == m for r in rows)
                                                   for m in sorted({str(r["lf_marker"]) for r in rows})},
              n_done=sum(r["lf_marker"] == ".done" for r in rows),
              n_done_n_imag_ge_1=sum(r["n_imag_ge_1"] is True for r in rows), table="results/L_repaired.csv",
              ok=True, note="report only (spec L-K)")
    complete = not missing_marker
    gates = {"L-1": l1, "L-R": lr, "L-K": lk}
    rep = dict(spec="docs/specs/D1G2R.md 6", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), lf_root=str(NEW), old_root=str(OLD), n_lf_ids=len(lf_ids),
               complete=complete, missing_marker=missing_marker[:20], n_done_total=sum(
                   marker(NEW / rid[j]) == ".done" for j in lf_ids),
               gates=gates, all_pass=complete and l1["ok"] and lr["ok"])
    C.write_json(C.RES / "L_check.json", rep)
    print(json.dumps(dict(all_pass=rep["all_pass"], complete=complete, L1=l1["ok"], LR=lr["ok"],
                          n_compared=n_cmp, max_d=dmax, LK=lk["markers"]), indent=1))
    if not complete or lrep.get("complete") is not True:              # deploy L report missing / not run yet
        return 2
    return 0 if rep["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
