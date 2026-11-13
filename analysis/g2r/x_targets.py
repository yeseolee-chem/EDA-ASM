#!/usr/bin/env python3
"""D1-G2R Phase X (docs/specs/D1G2R.md 4): the 9,252 targets and their groups.

    python analysis/g2r/x_targets.py          env_d1fu2 (pandas + pyarrow); job: jobs/py_g2r.slurm, after Phase C

1. d1pair_common.load_d1_labels() -> skeleton not in DPP -> 9,266 rows; flags from the record's `flags`.
2. targets = flag_short_forming and flag_regio_ambiguous both False, job_id order.
3. label_flag_* of analysis/d1_g2/results/C_compare.csv must equal the record flags row by row.
4. group from the G_status.csv `marker` (reuse '.done'; repair '.fail_ref_AtomValenceException' / '.fail_ref_KeyError';
   excluded_check '.fail_ts_TS_check_failed' / '.fail_ts_partition'; excluded_ts every other '.fail_*'). The job
   folders hold only stage markers (.done_<stage>, .fail_<stage>); the copy in $G2R/jobs must give the same state by the
   finalize rule (s0_start.derive; recorded decision, S0 1-3).
5. cr2 record (labels_d1_cr2.parquet): reference excess = e_frag{k}_rel_new_eh - min over the same key (key1/key2
   stacked, all 9,266 rows) x 627.5094740631; hygiene strict (d1_new, d2_new >= 0, d2_new <= 50) and tolerant (>= -0.5 =
   d_min_kcal of the D1 run config, d2_new <= 50), counted over the targets.
   ex_orig_gt20 = ex1 > 20 or ex2 > 20 of $D1PAIR_SCRATCH/P/rowattr.pkl.all.pkl (build_rowattr.py, original references).
Outputs: results/X_targets.csv, $G2R/x/{reuse_ids,repair_ids}.csv (job_id column), results/X_report.json.
Exit 0 / 3 (STOP: X-1..X-5 differ, X-6 excess > 20 or tolerant hygiene failures > 0).
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2r_common as C  # noqa: E402
from s0_start import STAGE_FAIL, derive  # noqa: E402

sys.path.insert(0, str(C.REPO / "analysis" / "d1_pair"))
import d1pair_common as DP  # noqa: E402

KCAL = 627.5094740631
EX_MAX = 20.0
D_MIN_TOL, D2_MAX = -0.5, 50.0
ROWATTR_ALL = Path(str(DP.ROWATTR) + ".all.pkl")
FLAGS = ("flag_async", "flag_short_forming", "flag_regio_ambiguous", "flag_small_extra_imag")
REPAIR = (".fail_ref_AtomValenceException", ".fail_ref_KeyError")
CHECK = (".fail_ts_TS_check_failed", ".fail_ts_partition")
WANT_GROUPS = dict(reuse=8614, repair=20, excluded_check=19, excluded_ts=599)
COLS = ["rxn_id", "job_id", "dipole", "skeleton", "series", "attach", "flag_async", "ex_orig_gt20", "g2_marker", "group"]


def group_of(marker: str) -> str:
    if marker == ".done":
        return "reuse"
    if marker in REPAIR:
        return "repair"
    if marker in CHECK:
        return "excluded_check"
    if marker.startswith(".fail_"):
        return "excluded_ts"
    raise ValueError(f"unknown G_status marker {marker!r}")


def ids_csv(path: Path, ids) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["job_id"])
    w.writerows([j] for j in ids)
    C.write_atomic(path, buf.getvalue())
    return C.sha256(path)


def main() -> int:
    import pandas as pd
    lab = [d for d in DP.load_d1_labels() if d["skeleton"] not in DP.DPP]
    lab.sort(key=lambda d: d["job_id"])
    fl = {d["job_id"]: {k: bool((d.get("flags") or {}).get(k)) for k in FLAGS} for d in lab}
    top_async_diff = [d["job_id"] for d in lab if bool(d.get("flag_async")) != fl[d["job_id"]]["flag_async"]]
    sf = [j for j, f in fl.items() if f["flag_short_forming"]]
    ra = [j for j, f in fl.items() if f["flag_regio_ambiguous"]]
    both = sorted(set(sf) & set(ra))
    tgt = [d for d in lab if not fl[d["job_id"]]["flag_short_forming"] and not fl[d["job_id"]]["flag_regio_ambiguous"]]
    tj = {d["job_id"] for d in tgt}
    # X-4 C_compare flags
    cc = pd.read_csv(C.C_COMPARE, dtype=str, keep_default_na=False).set_index("job_id")
    cc_missing = sorted(set(fl) - set(cc.index))
    flag_diff = []
    for j, f in fl.items():
        if j in cc.index:
            for k in FLAGS:
                v = cc.at[j, f"label_{k}"]
                if v not in ("True", "False") or (v == "True") != f[k]:
                    flag_diff.append(dict(job_id=j, flag=k, labels=f[k], c_compare=v))
    # groups
    st = pd.read_csv(C.G_STATUS, dtype=str, keep_default_na=False).set_index("job_id")
    st_missing = sorted(set(fl) - set(st.index))
    jobs_state_diff = []
    for j in sorted(fl):
        m = st.at[j, "marker"] if j in st.index else ""
        d = derive(C.G2R_JOBS / j) if (C.G2R_JOBS / j).is_dir() else "no_folder"
        if d != (m if STAGE_FAIL.match(m) else "stages_done"):
            jobs_state_diff.append(dict(job_id=j, g_status_marker=m, g2r_jobs_state=d))
    # ex_orig_gt20 (original references, build_rowattr.py)
    try:
        ra_all = pd.read_pickle(ROWATTR_ALL)
        ra_all = ra_all.set_index("job_id")
        exo = {j: bool(ra_all.at[j, "ex1"] > EX_MAX or ra_all.at[j, "ex2"] > EX_MAX) for j in fl if j in ra_all.index}
        rowattr_error = None
    except Exception as e:                                                    # noqa: BLE001
        exo, rowattr_error = {}, f"{type(e).__name__}: {e}"
    # cr2 record
    cr = pd.read_parquet(C.CR2_LABELS)
    stack = pd.concat([pd.DataFrame(dict(key=cr[f"key{k}"], e=cr[f"e_frag{k}_rel_new_eh"])) for k in (1, 2)])
    kmin = stack.groupby("key").e.min()
    for k in (1, 2):
        cr[f"ex{k}_new_kcal"] = (cr[f"e_frag{k}_rel_new_eh"] - cr[f"key{k}"].map(kmin)) * KCAL
    cr["ex_new_max"] = cr[["ex1_new_kcal", "ex2_new_kcal"]].max(axis=1)
    crt = cr[cr.job_id.isin(tj)]
    strict = crt[(crt.d1_new < 0) | (crt.d2_new < 0) | (crt.d2_new > D2_MAX)]
    toler = crt[(crt.d1_new < D_MIN_TOL) | (crt.d2_new < D_MIN_TOL) | (crt.d2_new > D2_MAX)]
    cr_missing = sorted(tj - set(cr.job_id))
    # table
    rows, groups = [], {}
    for d in tgt:
        j = d["job_id"]
        m = st.at[j, "marker"]
        g = group_of(m)
        groups.setdefault(g, []).append(j)
        rows.append([int(d["rxn_id"]), j, d.get("dipole"), d.get("skeleton"), d.get("series"), d.get("attach"),
                     fl[j]["flag_async"], exo.get(j), m, g])
    X = pd.DataFrame(rows, columns=COLS)
    buf = io.StringIO()
    X.to_csv(buf, index=False)
    C.write_atomic(C.RES / "X_targets.csv", buf.getvalue())
    sha_reuse = ids_csv(C.G2R_X / "reuse_ids.csv", groups.get("reuse", []))
    sha_repair = ids_csv(C.G2R_X / "repair_ids.csv", groups.get("repair", []))
    n_async = int(X.flag_async.sum())
    n_ex = int(sum(bool(v) for v in X.ex_orig_gt20 if v is not None))
    n_ex_missing = int(sum(v is None for v in X.ex_orig_gt20))
    gates = {
        "X-1": dict(n_non_dpp=len(lab), want=9266, ok=len(lab) == 9266),
        "X-2": dict(short_forming=len(sf), regio_ambiguous=len(ra), overlap=len(both), n_targets=len(tgt),
                    want=[9, 5, 0, 9252], ok=[len(sf), len(ra), len(both), len(tgt)] == [9, 5, 0, 9252]),
        "X-3": dict(counts={g: len(v) for g, v in sorted(groups.items())}, want=WANT_GROUPS,
                    g_status_missing=st_missing[:20], g2r_jobs_state_differs=len(jobs_state_diff),
                    g2r_jobs_state_examples=jobs_state_diff[:20],
                    ok={g: len(v) for g, v in groups.items()} == WANT_GROUPS and not st_missing
                    and not jobs_state_diff),
        "X-4": dict(n_flag_differs=len(flag_diff), examples=flag_diff[:20], c_compare_missing=cc_missing[:20],
                    flags=list(FLAGS), ok=not flag_diff and not cc_missing),
        "X-5": dict(flag_async=n_async, ex_orig_gt20=n_ex, want=[209, 8], rowattr=str(ROWATTR_ALL),
                    rowattr_error=rowattr_error, ex_orig_gt20_missing=n_ex_missing,
                    ex_orig_gt20_ids=X.loc[X.ex_orig_gt20 == True, "job_id"].tolist(),       # noqa: E712
                    ok=[n_async, n_ex] == [209, 8] and rowattr_error is None and n_ex_missing == 0),
        "X-6": dict(ex_new_max_kcal=float(crt.ex_new_max.max()), ex_new_max_kcal_all_9266=float(cr.ex_new_max.max()),
                    n_ex_new_gt20=int((crt.ex_new_max > EX_MAX).sum()),
                    hygiene_tolerant_fail=int(len(toler)), hygiene_strict_fail=int(len(strict)),
                    strict_fail_rows=strict[["rxn_id", "job_id", "d1_new", "d2_new"]].to_dict("records"),
                    tolerant_rule=f"d1_new, d2_new >= {D_MIN_TOL}, d2_new <= {D2_MAX}",
                    strict_rule=f"d1_new, d2_new >= 0, d2_new <= {D2_MAX}", cr2_rows=int(len(cr)),
                    cr2_missing_targets=cr_missing[:20], want="excess <= 20 (about 0.05), tolerant 0 / strict 4",
                    ok=float(crt.ex_new_max.max()) <= EX_MAX and len(toler) == 0 and not cr_missing),
    }
    bad = [k for k, v in gates.items() if not v["ok"]]
    rep = dict(spec="docs/specs/D1G2R.md 4 (Phase X)", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), n_targets=len(tgt),
               non_targets=[dict(job_id=d["job_id"], rxn_id=int(d["rxn_id"]),
                                 **{k: fl[d["job_id"]][k] for k in ("flag_short_forming", "flag_regio_ambiguous")},
                                 g2_marker=st.at[d["job_id"], "marker"] if d["job_id"] in st.index else None)
                            for d in lab if d["job_id"] not in tj],
               flag_async_top_level_vs_flags_differs=top_async_diff[:20],
               group_rule="G_status.csv marker (module doc); $G2R/jobs stage state checked with s0_start.derive",
               repair_ids=groups.get("repair", []), excluded_check_ids=groups.get("excluded_check", []),
               outputs={"results/X_targets.csv": C.sha256(C.RES / "X_targets.csv"),
                        str(C.G2R_X / "reuse_ids.csv"): sha_reuse, str(C.G2R_X / "repair_ids.csv"): sha_repair},
               inputs={str(DP.D1_LABELS): DP.D1_LABELS_SHA, str(C.G_STATUS): C.sha256(C.G_STATUS),
                       str(C.C_COMPARE): C.sha256(C.C_COMPARE), str(C.CR2_LABELS): C.sha256(C.CR2_LABELS),
                       str(ROWATTR_ALL): C.sha256(ROWATTR_ALL) if ROWATTR_ALL.is_file() else None},
               gates=gates, failed=bad, all_pass=not bad)
    C.write_json(C.RES / "X_report.json", rep)
    print(json.dumps(dict(all_pass=not bad, failed=bad, counts=gates["X-3"]["counts"], x5=[n_async, n_ex],
                          x6=[gates["X-6"]["ex_new_max_kcal"], len(toler), len(strict)]), indent=1))
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())
