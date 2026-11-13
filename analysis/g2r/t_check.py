#!/usr/bin/env python3
"""D1-G2R Phase T checks (docs/specs/D1G2R.md 9) after tokens_lf.py atoms / gate / tokens (g_g2r.slurm)
-> results/T_check.json. Before Phase D ($CH3/lf is read).

    python analysis/g2r/t_check.py          env_d1fu2 (numpy, pandas); py_g2r.slurm

T-1  results/lf/T_lf_check.json (deploy spec 6 = DEPLOY 6: elements [1, 6, 7, 8, 9, 14, 16, 17, 35], d_in 29, columns,
     Q-a / Q-b / T-a): gate and tokens without STOP, every tokens check true; and the tolerance records of the T jobs
     (results/F_tol/g2r_T*.json) ok.                                                                         STOP
T-R  the reactions usable before (CH3 T_manifest rows that are targets: 8,500): the new token
     $G2R/lf/fu2_lf/tokens/<rid>.npz has the same arrays (names, shapes, dtypes) as $CH3/lf/fu2_lf/tokens/<rid>.npz;
     Z, bin, edges, etype, n_bonds_rel, rxn_id, token_version equal exactly; num, pos, glob max |diff| <= 1e-6 (NaN at
     the same places).                                                                                         STOP
T-N  rxn 1007205 / 1007289 (CH3 extension failures, dipole sum 3.7e-3 / 2.0e-3 a.u.): atoms npz ext_ok, dip_check,
     ext_reason, Q_manifest use, token present.                                                               report
Exit 0 / 3.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2r_common as C  # noqa: E402

TOL = 1e-6
EXACT = ("Z", "bin", "edges", "etype", "n_bonds_rel", "rxn_id", "token_version")
FLOAT = ("num", "pos", "glob")
TN = (1007205, 1007289)
NEW_TOK, OLD_TOK = C.G2R_LF / "fu2_lf" / "tokens", C.CH3_LF_OLD / "fu2_lf" / "tokens"
NEW_RES = C.RES / "lf" / "fu2_lf"


def cmp_npz(a, b, np):
    za, zb = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    if sorted(za.files) != sorted(zb.files):
        return dict(keys=[sorted(za.files), sorted(zb.files)]), None
    bad, dmax = {}, 0.0
    for k in za.files:
        x, y = za[k], zb[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            bad[k] = f"shape/dtype {x.shape} {x.dtype} vs {y.shape} {y.dtype}"
            continue
        if k in FLOAT:
            nx, ny = np.isnan(x), np.isnan(y)
            if (nx != ny).any():
                bad[k] = "NaN positions differ"
                continue
            d = float(np.abs(x[~nx] - y[~ny]).max()) if (~nx).any() else 0.0
            dmax = max(dmax, d)
            if d > TOL:
                bad[k] = f"max |diff| {d:.3e}"
        elif not np.array_equal(x, y):
            bad[k] = "values differ"
    return bad, dmax


def main() -> int:
    import numpy as np
    import pandas as pd
    X = pd.read_csv(C.RES / "X_targets.csv", dtype={"job_id": str})
    tgt = set(X.rxn_id.astype(int))
    # T-1
    tc = C.read_json(C.RES / "lf" / "T_lf_check.json")
    g, t = tc.get("gate") or {}, tc.get("tokens") or {}
    trecs = [C.read_json(p) for p in sorted((C.RES / "F_tol").glob("g2r_T*.json"))]
    t1 = dict(gate_stop=g.get("stop"), gate_rc=g.get("gate_rc"), n_target=g.get("n_target"), n_use=g.get("n_use"),
              n_qa_fail=g.get("n_qa_fail"), n_qb_fail=g.get("n_qb_fail"), ext_used=g.get("ext_used"),
              ext_pass_rate=g.get("ext_pass_rate"), tokens_stop=t.get("stop"), tokens_checks=t.get("checks"),
              elements=t.get("elements"), d_in=t.get("d_in"), n_tol_records=len(trecs),
              tol_records_ok=bool(trecs) and all(r.get("ok") for r in trecs),
              ok=(g.get("stop") is False and t.get("stop") is False and bool(t.get("checks"))
                  and all((t.get("checks") or {}).values()) and bool(trecs) and all(r.get("ok") for r in trecs)))
    # T-R
    old_m = pd.read_csv(C.LF_REPORTS / "fu2_lf" / "T_manifest.csv")
    new_m = pd.read_csv(NEW_RES / "T_manifest.csv")
    old_t = sorted(int(r) for r in old_m.rxn_id if int(r) in tgt)
    miss, bad, dmax = [], [], 0.0
    for r in old_t:
        a, b = NEW_TOK / f"{r}.npz", OLD_TOK / f"{r}.npz"
        if not a.is_file():
            miss.append(r)
            continue
        bd, d = cmp_npz(a, b, np)
        if bd:
            bad.append(dict(rxn_id=r, problems=bd))
        if d is not None:
            dmax = max(dmax, d)
    tr = dict(n_old_use_targets=len(old_t), n_missing_in_new=len(miss), missing=miss[:20], n_differ=len(bad),
              differ=bad[:20], max_abs_diff_float=dmax, tol=TOL, n_new_manifest=int(len(new_m)),
              ok=not miss and not bad)
    # T-N
    qm = pd.read_csv(NEW_RES / "Q_manifest.csv").set_index("rxn_id")
    tn = {}
    for r in TN:
        p = C.G2R_LF / "fu2_lf" / "atoms" / f"{r}.npz"
        rec = dict(atoms_npz=p.is_file(), token=(NEW_TOK / f"{r}.npz").is_file(),
                   in_new_T_manifest=bool((new_m.rxn_id == r).any()))
        if p.is_file():
            z = np.load(p, allow_pickle=False)
            for k in ("ext_ok", "dip_check", "ext_reason", "q_json_maxdiff"):
                if k in z.files:
                    v = z[k]
                    rec[k] = v.item() if v.size == 1 else v.tolist()
        if r in qm.index:
            rec.update({f"Q_{k}": (qm.at[r, k].item() if hasattr(qm.at[r, k], "item") else qm.at[r, k])
                        for k in ("ext_ok", "ext_reason", "qa_ok", "qb_ok", "use") if k in qm.columns})
        tn[str(r)] = rec
    tn_rep = dict(reactions=tn, n_pass=sum(bool(v.get("token")) for v in tn.values()), ok=True,
                  note="report only (spec T-N)")
    gates = {"T-1": t1, "T-R": tr, "T-N": tn_rep}
    rep = dict(spec="docs/specs/D1G2R.md 9", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"),
               inputs={"results/lf/T_lf_check.json": C.sha256(C.RES / "lf" / "T_lf_check.json"),
                       "results/lf/fu2_lf/T_manifest.csv": C.sha256(NEW_RES / "T_manifest.csv")},
               gates=gates, all_pass=t1["ok"] and tr["ok"])
    C.write_json(C.RES / "T_check.json", rep)
    print(json.dumps(dict(all_pass=rep["all_pass"], T1=t1["ok"], TR=tr["ok"], n_compared=len(old_t) - len(miss),
                          max_abs=dmax, TN={k: v.get("token") for k, v in tn.items()}), indent=1, default=str))
    return 0 if rep["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
