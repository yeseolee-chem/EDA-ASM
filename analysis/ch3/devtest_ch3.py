#!/usr/bin/env python3
"""devtest_ch3.py — developer dry run of the CH3 code paths (not a spec phase; nothing here is a result).
Runs on CPU with CH3_RESULTS=$CH3/dev/res (every write stays under $CH3/dev) on the LADDER g2 inputs
(DeployData('g2'), which exist before the lf regeneration), tiny limits, 1-2 epochs.

  python devtest_ch3.py train    (env_d1fu2) C-Y / C-D / C-5 functions, ch3-mode run_unit + C-T + val metric, train_final,
                                 predict_ch3 final / cv against the recorded predictions, 10-target-mode run_unit
  python devtest_ch3.py eval     (env_espley) eval_ch3.py run on a fake K3 (lf_sum + noise) -> $CH3/dev/res
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
import ch3_common as C3  # noqa: E402

DEV = C3.CH3 / "dev"
if C3.RES.resolve() != (DEV / "res").resolve():
    C3.stop(f"devtest needs CH3_RESULTS={DEV / 'res'} (got {C3.RES})")


def cmd_train():
    import data_ch3 as DC3
    import train_ch3 as TC
    import predict_ch3 as PC
    import tests_ch3 as TS
    import train_tf as TT
    import commit_phase  # noqa: F401
    DL, FT, F = DC3.DL, DC3.FT, DC3.F
    out = {}
    E = DL.DeployData("g2")
    D = DL.DeployData("g2")
    D.y10 = D.y
    D.y = D.y10 @ C3.M()
    D.rung = C3.RUNG3
    out["C_Y"] = {k: v for k, v in TS.c_y(D).items() if k != "identity_residual_all_labels"}
    out["C_D_ok"] = TS.c_d(D, E)["ok"]
    out["C_5_ok"] = TS.c_5(D, FT, TT)["ok"]
    DC3.use_targets(C3.TARGETS3)
    u = dict(rung=C3.RUNG3, arm="S", model="TF", fold=0)
    rec = TC.run_unit(D, u, C3.TARGETS3, root=DEV / "units", n_epochs=2, seeds=F.FINAL_SEEDS[:1], cells=[0, 3], log=1,
                      smoke=True, limit=300)
    out["unit"] = dict(chosen=rec["chosen"], n_test=rec["n_test"], targets=rec["targets"],
                       gates=TT.gates_ok(rec), out_dims=[t["out_dim"] for t in rec["tune"]])
    ct = TC.ct_checks(D, DEV / "units" / f"{TC.uname(u)}__s0.pt")
    out["C_T"] = ct["checks"]
    out["val_metric"] = TC.val_metric_check(rec, 4)
    recs = {TC.uname(u): rec}
    TC.train_final(D, dict(cell=TC.GRID[0], n_epochs=1), recs, {}, {}, root=DEV / "model", limit=300, n_epochs=1,
                   check_n=50, smoke=True)
    tok = C3.G2 / "fu2" / "tokens"
    x182 = C3.G2 / "X182_d1_g2.parquet"
    trep = C3.REPO / "analysis" / "d1_g2" / "results" / "fu2" / "T_report.json"
    o, ex, rep = PC.run_predict(tok, x182, DEV / "model" / "W_check_pairs.csv", "final", t_report=trep,
                                model_dir=DEV / "model")
    W = pd.read_parquet(DEV / "model" / "W_check_pred.parquet")
    m = W.merge(o, on=["pair_id", "target"], suffixes=("_w", "_p"))
    out["predict_final"] = dict(n=int(len(m)), n_excluded=int(len(ex)), max_abs=float(np.abs(m.yhat_w - m.yhat_p).max()),
                                targets=sorted(set(o.target)))
    P = pd.read_parquet(DEV / "units" / f"{TC.uname(u)}.parquet")
    pc = DEV / "cv_pairs.csv"
    C3.write_csv(pc, D.P.loc[sorted(P.pair_id.unique()), ["a_rxn", "b_rxn"]].reset_index())
    o2, ex2, _ = PC.run_predict(tok, x182, pc, "cv", folds=[0], seeds=[0], t_report=trep, model_dir=DEV / "units")
    m2 = P.merge(o2, on=["pair_id", "target"], suffixes=("_k", "_p"))
    out["predict_cv"] = dict(n=int(len(m2)), n_excluded=int(len(ex2)), max_abs=float(np.abs(m2.yhat_k - m2.yhat_p).max()))
    DC3.use_targets(C3.TARGETS10)
    u10 = dict(rung="g2", arm="S", model="TF", fold=0)
    r10 = TC.run_unit(E, u10, C3.TARGETS10, root=DEV / "c3r", n_epochs=1, seeds=F.FINAL_SEEDS[:1], cells=[0], log=1,
                      smoke=True, limit=200)
    out["mode10"] = dict(n_targets=len(r10["targets"]), out_dim=r10["tune"][0]["out_dim"], gates=TT.gates_ok(r10),
                         mode=DC3.current_mode())
    out["ok"] = bool(out["C_D_ok"] and out["C_5_ok"] and all(ct["checks"].values()) and out["val_metric"]["ok"]
                     and out["predict_final"]["max_abs"] <= 1e-5 and out["predict_cv"]["max_abs"] <= 1e-5
                     and out["unit"]["gates"] and out["mode10"]["out_dim"] == 10)
    C3.write_json(DEV / "devtest_train.json", out)
    print(json.dumps(out, indent=1, default=str))
    return 0 if out["ok"] else 1


def cmd_eval():
    import eval_ch3 as EV
    L, Y, attr = EV.labels()
    lfs, _ = EV.lf_sum(L)
    rng = np.random.default_rng(1)
    K = lfs.copy()
    K["y"] = Y.to_numpy()[Y.index.get_indexer(K.pair_id), [C3.TARGETS3.index(t) for t in K.target]]
    K["yhat"] = K.yhat + rng.normal(0, 0.05, len(K))
    K["yhat_seed_sd"] = np.abs(rng.normal(0, 0.1, len(K)))
    S = EV.s_folds()
    K = K.join(S[["pos", "ref_kind"]], on="pair_id")
    K = K.assign(rung=C3.RUNG3, arm="S", model="TF", base=0.0, beta=0.0)
    cols = ["pair_id", "ref_kind", "pos", "fold", "rung", "arm", "model", "target", "y", "yhat", "base", "beta",
            "yhat_seed_sd"]
    C3.write_parquet(EV.K3_PRED, K[cols])
    rc = EV.cmd_run()
    rep = C3.read_json(C3.RES / "E3_report.json")
    print(json.dumps(dict(rc=rc, adoption=rep["adoption"]["adopted"], per=rep["adoption"]["per_target"]), indent=1))
    return rc


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["train"]:
        sys.exit(cmd_train())
    if a == ["eval"]:
        sys.exit(cmd_eval())
    print(__doc__)
    sys.exit(2)
