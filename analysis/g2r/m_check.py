#!/usr/bin/env python3
"""D1-G2R Phase M checks (docs/specs/D1G2R.md 7) -> results/M_check.json. Before Phase D ($CH3/lf is read).

    python analysis/g2r/m_check.py          env_d1fu2 (pandas); py_g2r.slurm

M-1  records of $G2R/lf/labels_lf.json = L .done count over lf_ids, results/lf/M_report.json all_pass with charge != 0
     count 0.                                                                                              STOP
M-2  every reuse reaction: its record equals the $CH3/lf/labels_lf.json record of the same rxn_id (json.dumps of the
     parsed records, so key order is compared and NaN == NaN), and its input_meta_lf.csv A_idx string equals
     $CH3/lf/input_meta_lf.csv.                                                                             STOP
Exit 0 / 3.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2r_common as C  # noqa: E402


def ids(name):
    with open(C.G2R_X / name, newline="") as fh:
        return [r["job_id"] for r in csv.DictReader(fh)]


def meta(path):
    import pandas as pd
    m = pd.read_csv(path, dtype={"A_idx": str})
    return {int(r): a for r, a in zip(m.rxn_id, m.A_idx)}, len(m)


def main() -> int:
    with open(C.RES / "X_targets.csv", newline="") as fh:
        rid = {r["job_id"]: int(r["rxn_id"]) for r in csv.DictReader(fh)}
    reuse, lf = ids("reuse_ids.csv"), ids("lf_ids.csv")
    done = sorted(rid[j] for j in lf if (C.G2R_LF / "g2lf_final" / str(rid[j]) / ".done").is_file())
    mrep = C.read_json(C.RES / "lf" / "M_report.json")
    new = {int(r["rxn_id"]): r for r in json.loads((C.G2R_LF / "labels_lf.json").read_text())}
    old = {int(r["rxn_id"]): r for r in json.loads((C.CH3_LF_OLD / "labels_lf.json").read_text())}
    m_new, n_meta_new = meta(C.G2R_LF / "input_meta_lf.csv")
    m_old, _ = meta(C.CH3_LF_OLD / "input_meta_lf.csv")
    m1 = dict(n_records=len(new), n_l_done=len(done), records_equal_l_done=sorted(new) == done,
              n_input_meta_rows=n_meta_new, input_meta_equal_l_done=sorted(m_new) == done,
              n_charge_nonzero=mrep.get("n_charge_nonzero"), m_report_all_pass=mrep.get("all_pass"),
              m_report_spec=mrep.get("spec"))
    m1["ok"] = bool(m1["records_equal_l_done"] and m1["input_meta_equal_l_done"] and m1["n_charge_nonzero"] == 0
                    and m1["m_report_all_pass"])
    rec_diff, meta_diff, missing = [], [], []
    for j in reuse:
        r = rid[j]
        if r not in new or r not in old or r not in m_new or r not in m_old:
            missing.append(dict(rxn_id=r, new=r in new, old=r in old, meta_new=r in m_new, meta_old=r in m_old))
            continue
        if json.dumps(new[r]) != json.dumps(old[r]):
            keys = [k for k in old[r] if json.dumps(old[r].get(k)) != json.dumps(new[r].get(k))]
            rec_diff.append(dict(rxn_id=r, keys=keys or ["(key order)"]))
        if m_new[r] != m_old[r]:
            meta_diff.append(dict(rxn_id=r, new=m_new[r], old=m_old[r]))
    m2 = dict(n_reuse=len(reuse), n_missing=len(missing), missing=missing[:20], n_record_differs=len(rec_diff),
              record_differs=rec_diff[:20], n_input_meta_differs=len(meta_diff), input_meta_differs=meta_diff[:20],
              n_old_records=len(old), ok=not missing and not rec_diff and not meta_diff)
    gates = {"M-1": m1, "M-2": m2}
    rep = dict(spec="docs/specs/D1G2R.md 7", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"),
               inputs={str(C.G2R_LF / "labels_lf.json"): C.sha256(C.G2R_LF / "labels_lf.json"),
                       str(C.G2R_LF / "input_meta_lf.csv"): C.sha256(C.G2R_LF / "input_meta_lf.csv"),
                       str(C.CH3_LF_OLD / "labels_lf.json"): C.sha256(C.CH3_LF_OLD / "labels_lf.json"),
                       str(C.CH3_LF_OLD / "input_meta_lf.csv"): C.sha256(C.CH3_LF_OLD / "input_meta_lf.csv")},
               gates=gates, all_pass=m1["ok"] and m2["ok"])
    C.write_json(C.RES / "M_check.json", rep)
    print(json.dumps(dict(all_pass=rep["all_pass"], M1=m1["ok"], M2=m2["ok"], n_records=len(new),
                          n_l_done=len(done), rec_diff=len(rec_diff), meta_diff=len(meta_diff)), indent=1))
    return 0 if rep["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
