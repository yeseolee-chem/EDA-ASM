#!/usr/bin/env python3
"""devtest_m.py — developer dry run of the CH3M code paths (not a spec phase; nothing here is a result). Needs
CH3M_SCRATCH=$CH3M/dev and CH3M_RESULTS=$CH3M/dev/res (every write stays there); CPU, tiny limits.

  python devtest_m.py nn     (env_d1fu2) train_m.run_unit TF 10 on g2 and GNN 3 on lf (limits, 1 epoch), the GNN C-T
                             checks of cmd_smoke, train_final TF10 on lf (1 epoch) + predict_m check-i2
  python devtest_m.py xgb    (env_espley) xgb_m PackData split checks (lf, g2), run_unit XGB 3 mode on lf with 1 grid cell
  python devtest_m.py eval   (env_espley) eval_m selftest + run on a synthetic KM (deploy / ch3 predictions + noise)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3m_common as C  # noqa: E402

if not str(C.CH3M).endswith("/dev") or not str(C.RES).endswith("/dev/res"):
    C.stop(f"devtest needs CH3M_SCRATCH=.../ch3m/dev and CH3M_RESULTS=.../ch3m/dev/res (got {C.CH3M}, {C.RES})")


def cmd_nn():
    import train_m as TM
    import predict_m as PM
    import torch
    out = {}
    D = TM.data_for(10, "g2")
    u = dict(rung="g2", arm="S", model="TF", fold=0, mode=10)
    r = TM.run_unit(D, u, TM.T10, "TF", root=C.CH3M / "t_tf", n_epochs=1, seeds=TM.F.FINAL_SEEDS[:1], cells=[0], log=1,
                    smoke=True, limit=300)
    out["tf10_g2"] = dict(chosen=r["chosen"], out_dim=r["tune"][0]["out_dim"], gates=TM.TT.gates_ok(r), n_test=r["n_test"])
    D3 = TM.data_for(3)
    u3 = dict(rung="lf3", arm="S", model="GNN", fold=0, mode=3)
    r3 = TM.run_unit(D3, u3, TM.T3, "GNN", root=C.CH3M / "t_gnn", n_epochs=1, seeds=TM.F.FINAL_SEEDS[:1], cells=[0, 3],
                     log=1, smoke=True, limit=150)
    pt = C.CH3M / "t_gnn" / f"{TM.uname(u3)}__s0.pt"
    ct = TM.TC.ct_checks(D3, pt)
    obj = torch.load(pt, map_location="cpu")
    ct["checks"]["pt_model_GNN"] = obj.get("model") == "GNN"
    mm = TM.Mo.build("GNN", obj["cell"], int(obj["d_in"]), 4, int(obj["d_glob"]), TM.VARIANT, TM.F.DROPOUT)
    mm.load_state_dict(obj["state"], strict=True)
    ct["checks"]["grid_is_Mo_GRID_GNN"] = r3["grid"] == TM.norm(TM.Mo.GRID["GNN"])
    out["gnn3_lf"] = dict(chosen=r3["chosen"], out_dims=[t["out_dim"] for t in r3["tune"]], gates=TM.TT.gates_ok(r3),
                          ct=ct["checks"], val_metric=TM.TC.val_metric_check(r3, 4))
    D10 = TM.data_for(10)
    TM._GATE["prereg"] = "devtest"
    TM.train_final(D10, dict(cell=TM.grid("TF")[0], n_epochs=1), {}, "TF", 10, root=C.MODEL_DIR, check_n=50)
    rc = PM.cmd_check_i2("TF10")
    out["final_tf10_i2_rc"] = rc
    out["IM"] = C.read_json(C.RES / "IM_selftest.json").get("I-2")
    out["ok"] = bool(out["tf10_g2"]["gates"] and out["tf10_g2"]["out_dim"] == 10 and all(ct["checks"].values())
                     and out["gnn3_lf"]["val_metric"]["ok"] and rc == 0)
    C.write_json(C.CH3M / "devtest_nn.json", out)
    print(json.dumps(out, indent=1, default=str))
    return 0 if out["ok"] else 1


def cmd_xgb():
    import xgb_m as XM
    out = {}
    for src, mode in (("lf", 3), ("g2", 10)):
        D = XM.PackData(src, mode)
        sp = D.split(0)
        out[f"split_{src}"] = dict(n_train=int(len(sp["train"])), n_test=int(len(sp["test"])),
                                   n_inner=[(int(len(f["train"])), int(len(f["val"]))) for f in sp["inner"]])
    D = XM.PackData("lf", 3)
    u = dict(rung=D.rung, arm="S", model="XGB", fold=0)
    r = XM.run_unit(D, u, XM.T3, C.CH3M / "t_xgb", cells=[0], smoke=True)
    P = pd.read_parquet(C.CH3M / "t_xgb" / f"{XM.uname(u)}.parquet")
    out["xgb3"] = dict(chosen=r["chosen"], val=r["val_kcal_mae"], targets=sorted(P.target.unique()), gates=r["gates_ok"],
                       cpu_seconds=r["cpu_seconds"], tx_mode=XM.mode_now())
    out["ok"] = bool(r["gates_ok"] and sorted(P.target.unique()) == sorted(XM.T3))
    C.write_json(C.CH3M / "devtest_xgb.json", out)
    print(json.dumps(out, indent=1, default=str))
    return 0 if out["ok"] else 1


def cmd_eval():
    import eval_m as EM
    EV = EM.EV
    EM.cmd_selftest()
    L, Y, attr = EV.labels()
    rng = np.random.default_rng(3)
    dk = pd.read_parquet(EM.DEP_K)
    k3 = pd.read_parquet(EM.K3)
    parts = []
    for model in ("TF", "GNN", "XGB"):
        q = dk.copy()
        q["yhat"] = q.yhat + rng.normal(0, 0.03, len(q))
        q["rung"], q["model"] = "lf10", model
        q["yhat_seed_sd"] = 0.0 if model == "XGB" else np.abs(rng.normal(0, 0.1, len(q)))
        parts.append(q)
    for model in ("GNN", "XGB"):
        q = k3.copy()
        q["yhat"] = q.yhat + rng.normal(0, 0.03, len(q))
        q["rung"], q["model"] = "lf3", model
        parts.append(q)
    KM = pd.concat(parts, ignore_index=True)[list(dk.columns)]
    C.write_parquet(EM.KM, KM)
    C.write_bytes(C.RES / "S0_start.json", (C.REPO / "analysis" / "ch3m" / "results" / "S0_start.json").read_bytes())
    rc = EM.cmd_run()
    rep = C.read_json(C.RES / "EM_report.json")
    print(json.dumps(EV._clean(dict(rc=rc, choice=rep["choice"]["chosen"], final=rep["choice"]["final_model"],
                                    q1=rep["summary"]["q1_marks"])), indent=1, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    a = sys.argv[1:]
    fn = dict(nn=cmd_nn, xgb=cmd_xgb, eval=cmd_eval).get(a[0] if a else "")
    if fn is None:
        raise SystemExit(__doc__)
    sys.exit(fn())
