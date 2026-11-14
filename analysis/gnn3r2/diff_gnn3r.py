#!/usr/bin/env python3
"""GNN3R2: comparison with GNN3R (docs/specs/GNN3R2.md 4-2 G-1 .. G-4, 5 S-5).

    python analysis/gnn3r2/diff_gnn3r.py pairs     # after pairs_r.py rerun (PYENV=reactot)
    python analysis/gnn3r2/diff_gnn3r.py splits    # after splits_r.py (PYENV=espley)

pairs -> results/G_diff_gnn3r.csv (every pair / row that differs from GNN3R) and results/G_report.json. When a G check
fails, results/B_label_report.json is removed (the GNN3 loader then refuses F_pairs_r.csv) and the exit code is 3.
  T (4-2) = aryl pairs (pos p / m / o) whose a has F = (azomethine ylide, Bp, R1, cfg t, orient of T10104, endo,
  c3 in {cis, trans}). G-1: pairs outside T equal GNN3R field by field (CSV text). G-2 (DECISIONS.md DEC-1): inside T,
  new - GNN3R is a subset of the 6 expected new pairs (any missing one was dropped by P-4 v2 mismatch), GNN3R - new is
  exactly T10104 -> T10116, pairs in both are equal field by field. G-3: B_row_status rows that differ from GNN3R
  (status or fam) are among the 7 affected reactions. G-4: reaction / pair counts = expected minus the P-4 drops,
  reference pairs unchanged.
splits -> results/S5_report.json: the test lists of folds 0, 2, 3, 4 as (a_job, b_job) = GNN3R (ref_gnn3r/S_folds_r.csv).
Exit 0 pass, 3 = STOP.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402

DD = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "c_ghost", "barrier"]
COMPARE = ["ref_kind", "fam", "core", "pos", "Xb", "sigma", "p4_v2"] + [f"dd_{t}" for t in DD]
EXPECTED_ADDED = {("T10104", "T07195"), ("T10116", "T07195"), ("T10047", "T10056"), ("T10067", "T10056"),
                  ("T10092", "T10056"), ("T10111", "T10056")}                       # 0-4
EXPECTED_REMOVED = {("T10104", "T10116")}
ALLOWED_ROWS = {"T10047", "T10056", "T10067", "T10092", "T10111", "T10104", "T10116"}   # G-3
GNN3R_REFERENCE_PAIRS = 6978
KEY = ["a_job", "b_job"]


def read(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def in_t(P: pd.DataFrame, orient: str) -> pd.Series:
    return ((P.dipole == "azomethine ylide") & (P.skeleton == "Bp") & (P.series == "R1") & (P.cfg == "t")
            & (P.orient == orient) & (P.face == "endo") & P.c3.isin(["cis", "trans"]) & P.pos.isin(["p", "m", "o"]))


def field_diffs(old: pd.DataFrame, new: pd.DataFrame, in_t_flag: bool) -> tuple[list, list, list]:
    """(only in GNN3R, only in GNN3R2, differing fields of common pairs) as diff records"""
    m = old[KEY + COMPARE].merge(new[KEY + COMPARE], on=KEY, how="outer", suffixes=("_gnn3r", "_gnn3r2"),
                                 indicator=True)
    only_old = m[m._merge == "left_only"]
    only_new = m[m._merge == "right_only"]
    both = m[m._merge == "both"]
    removed = [dict(kind="pair_removed", a_job=a, b_job=b, in_T=in_t_flag, column="fam", gnn3r=f, gnn3r2="")
               for a, b, f in zip(only_old.a_job, only_old.b_job, only_old.fam_gnn3r)]
    added = [dict(kind="pair_added", a_job=a, b_job=b, in_T=in_t_flag, column="fam", gnn3r="", gnn3r2=f)
             for a, b, f in zip(only_new.a_job, only_new.b_job, only_new.fam_gnn3r2)]
    values = []
    for col in COMPARE:
        d = both[both[f"{col}_gnn3r"] != both[f"{col}_gnn3r2"]]
        values += [dict(kind="pair_value", a_job=a, b_job=b, in_T=in_t_flag, column=col, gnn3r=x, gnn3r2=y)
                   for a, b, x, y in zip(d.a_job, d.b_job, d[f"{col}_gnn3r"], d[f"{col}_gnn3r2"])]
    return removed, added, values


def cmd_pairs() -> int:
    t0 = time.time()
    rows = pd.read_pickle(C.ROWATTR_R)
    o = rows.loc[rows.job_id == "T10104", "orient"]
    orient = "" if o.empty or pd.isna(o.iloc[0]) else str(o.iloc[0])
    old, new = read(C.REF_GNN3R / "F_pairs_r.csv"), read(C.PAIRS_R)
    ot, nt = in_t(old, orient), in_t(new, orient)
    rem1, add1, val1 = field_diffs(old[~ot], new[~nt], False)
    rem2, add2, val2 = field_diffs(old[ot], new[nt], True)
    gates = {"G-1": dict(n_gnn3r=int((~ot).sum()), n_gnn3r2=int((~nt).sum()), only_gnn3r=len(rem1), only_gnn3r2=len(add1),
                         value_diffs=len(val1), ok=not rem1 and not add1 and not val1)}
    b = C.read_json(C.RES / "B_report.json")
    dropped = {(d["a_job"], d["b_job"]) for d in b["p4"]["dropped"]}
    added = {(d["a_job"], d["b_job"]) for d in add2}
    removed = {(d["a_job"], d["b_job"]) for d in rem2}
    missing = EXPECTED_ADDED - added
    n_t_new = {tuple(k) for k in new.loc[nt, KEY].to_numpy().tolist()}
    gates["G-2"] = dict(rule="DEC-1: change inside T", orient_T=orient, n_T_gnn3r=int(ot.sum()), n_T_gnn3r2=int(nt.sum()),
                        added=sorted(map(list, added)), removed=sorted(map(list, removed)),
                        expected_added=sorted(map(list, EXPECTED_ADDED)), missing_expected=sorted(map(list, missing)),
                        missing_dropped_by_p4=sorted(map(list, missing & dropped)), value_diffs_in_T=len(val2),
                        literal_subset_of_expected=n_t_new <= EXPECTED_ADDED,
                        ok=added <= EXPECTED_ADDED and missing <= dropped and removed == EXPECTED_REMOVED
                        and not val2 and not (EXPECTED_REMOVED & n_t_new))
    so, sn = read(C.REF_GNN3R / "B_row_status.csv"), read(C.RES / "B_row_status.csv")
    ms = so.merge(sn, on="job_id", how="outer", suffixes=("_gnn3r", "_gnn3r2"), indicator=True)
    diff_rows = ms[(ms._merge != "both") | (ms.status_gnn3r != ms.status_gnn3r2) | (ms.fam_gnn3r != ms.fam_gnn3r2)]
    row_recs = [dict(kind="row_status", a_job=j, b_job="", in_T="", column="status | fam",
                     gnn3r=f"{s1} | {f1}", gnn3r2=f"{s2} | {f2}")
                for j, s1, f1, s2, f2 in zip(diff_rows.job_id, diff_rows.status_gnn3r, diff_rows.fam_gnn3r,
                                             diff_rows.status_gnn3r2, diff_rows.fam_gnn3r2)]
    gates["G-3"] = dict(n_rows_differ=len(diff_rows), jobs=sorted(diff_rows.job_id), allowed=sorted(ALLOWED_ROWS),
                        ok=set(diff_rows.job_id) <= ALLOWED_ROWS and len(so) == len(sn) == C.N_TARGETS)
    survive = EXPECTED_ADDED - missing
    old_keys = {tuple(k) for k in old[KEY].to_numpy().tolist()}
    want_pairs = (old_keys - EXPECTED_REMOVED) | survive
    want_rxn = {j for k in want_pairs for j in k}
    rxn_new = set(read(C.RES / "B_reactions.csv").job_id)
    new_keys = {tuple(k) for k in new[KEY].to_numpy().tolist()}
    n_ref = int((new.ref_kind == "reference").sum())
    gates["G-4"] = dict(n_pairs=len(new), want_pairs=len(want_pairs), want_pairs_spec=C.EXPECT_PAIRS - len(missing),
                        n_reactions=len(rxn_new), want_reactions=len(want_rxn),
                        n_reference=n_ref, want_reference=GNN3R_REFERENCE_PAIRS, p4_dropped_new_pairs=len(missing),
                        ok=new_keys == want_pairs and rxn_new == want_rxn and len(new) == C.EXPECT_PAIRS - len(missing)
                        and n_ref == GNN3R_REFERENCE_PAIRS)
    recs = rem1 + add1 + val1 + rem2 + add2 + val2 + row_recs
    cols = ["kind", "a_job", "b_job", "in_T", "column", "gnn3r", "gnn3r2"]
    C.write_atomic(C.RES / "G_diff_gnn3r.csv", pd.DataFrame(recs, columns=cols).to_csv(index=False))
    bad = [g for g, v in gates.items() if not v["ok"]]
    rep = dict(spec="docs/specs/GNN3R2.md 4-2", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), wall_s=round(time.time() - t0, 1),
               inputs={str(p): C.sha256(p) for p in (C.REF_GNN3R / "F_pairs_r.csv", C.PAIRS_R,
                                                    C.REF_GNN3R / "B_row_status.csv", C.RES / "B_row_status.csv")},
               gates=gates, outputs={str(C.RES / "G_diff_gnn3r.csv"): C.sha256(C.RES / "G_diff_gnn3r.csv")},
               all_pass=not bad, failed=bad)
    if bad and C.LABEL_REPORT.is_file():
        C.LABEL_REPORT.unlink()
        rep["label_report_removed"] = str(C.LABEL_REPORT)
    C.write_json(C.RES / "G_report.json", rep)
    print(json.dumps(dict(all_pass=not bad, failed=bad, added=gates["G-2"]["added"], removed=gates["G-2"]["removed"],
                          missing=gates["G-2"]["missing_expected"], rows=gates["G-3"]["jobs"],
                          counts={k: gates["G-4"][k] for k in ("n_pairs", "want_pairs", "n_reactions", "want_reactions",
                                                               "n_reference")}), indent=1))
    return 0 if not bad else 3


def cmd_splits() -> int:
    old, new = read(C.REF_GNN3R / "S_folds_r.csv"), read(C.S_FOLDS_R)
    per = {}
    for k in ("-1", "0", "1", "2", "3", "4"):
        a = {tuple(x) for x in old.loc[old.fold_T1 == k, KEY].to_numpy().tolist()}
        b = {tuple(x) for x in new.loc[new.fold_T1 == k, KEY].to_numpy().tolist()}
        per[k] = dict(n_gnn3r=len(a), n_gnn3r2=len(b), only_gnn3r=sorted(map(list, a - b))[:20],
                      only_gnn3r2=sorted(map(list, b - a))[:20], equal=a == b)
    ok = all(per[k]["equal"] for k in ("0", "2", "3", "4"))
    rep = dict(spec="docs/specs/GNN3R2.md 5 S-5", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), rule="test lists of folds 0, 2, 3, 4 = GNN3R as (a_job, b_job)",
               folds=per, all_pass=ok, failed=[] if ok else ["S-5"])
    C.write_json(C.RES / "S5_report.json", rep)
    print(json.dumps(dict(all_pass=ok, **{k: (v["n_gnn3r"], v["n_gnn3r2"], v["equal"]) for k, v in per.items()}), indent=1))
    return 0 if ok else 3


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("pairs", "splits"):
        raise SystemExit(__doc__)
    sys.exit(cmd_pairs() if sys.argv[1] == "pairs" else cmd_splits())
