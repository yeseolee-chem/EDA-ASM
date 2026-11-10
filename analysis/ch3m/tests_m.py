#!/usr/bin/env python3
r"""tests_m.py — CH3M Phase C (docs/specs/CH3M_MODELS.md 5). Exit 0 all pass, 2 not ready, 3 STOP.

  python tests_m.py cpu      C-D on CPU (env_d1fu2): the train_m lf data / splits = ch3 Ch3Data / DeployData('lf'),
                             the XP lf pack pairs / splits / labels = the same lists -> results/C_part_cpu.json
  python tests_m.py merge    C_part_cpu + CR_tf (C-RT) + CR_gnn (C-RG, [+ CR_gnn2 = C-RG2]) + CR_xgb (C-RX) +
                             C8_smoke_nn (C-8 GNN, C-T GNN) + C8_smoke_xgb (C-8 XGB, C-T XGB) + C-4 / C-5 of every
                             fit -> results/C_tests.json (all_pass, trainer_code_sha256 of train_m and xgb_m);
                             the C-RG exception rule of spec 5 -> results/C_deviation.json when it applies

C-RG exception (spec 5, only when C-RT passes and C-RG fails): C-RG2 (train_m.py cr GNN rg2, run by cmd_cr itself) =
C-RG (test yhat max |diff| <= 1e-4) but not LADDER -> code difference -> STOP; C-RG2 != C-RG -> GNN training is not
deterministic -> recorded in C_deviation.json, C-RG counted as passed (the summary notes the larger GNN wobble).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3m_common as C  # noqa: E402

RG2_TOL = 1e-4


def cmd_cpu():
    import data_ch3 as DC3
    D10, D3 = DC3.get_data("lf"), DC3.get_data("ch3")
    arrays = ("onehot", "bin", "num", "pos", "mask", "etype", "frag", "n", "glob", "ra", "rb", "usable_pair", "rids")
    a_eq = {k: bool(np.array_equal(getattr(D10, k), getattr(D3, k))) for k in arrays}
    a_eq["y3_eq_y10_at_M"] = bool(np.array_equal(D3.y, D10.y @ C.M(), equal_nan=True))
    lists = {}
    for k in list(range(C.NF)) + ["all"]:
        a, b = D10.split(k), D3.split(k)
        for key in ("train", "test", "inner_train", "inner_val"):
            lists[f"{k}/{key}"] = bool(np.array_equal(a[key], b[key]))
    z = np.load(C.XPACK / "lf.npz", allow_pickle=False)
    info = C.read_json(C.XPACK / "lf.json")
    pk = dict(pair_order=bool((z["pair_id"] == np.array([str(q) for q in D10.P.index])).all()),
              ra=bool(np.array_equal(z["ra"], D10.ra)), rb=bool(np.array_equal(z["rb"], D10.rb)),
              usable_pair=bool(np.array_equal(z["usable_pair"], D10.usable_pair)),
              y10=bool(np.array_equal(z["y10"], D10.y, equal_nan=True)),
              y4=bool(np.array_equal(z["y4"], D3.y, equal_nan=True)),
              pair_list_sha=info["pair_list_sha"] == D10.pair_list_sha == D3.pair_list_sha,
              npz_sha=C.sha256(C.XPACK / "lf.npz") == info["npz_sha256"])
    for k in range(C.NF):
        sp = D10.split(k)
        for key in ("train", "test", "inner_train", "inner_val"):
            rows = np.array([D10.pid_row[q] for q in z[f"f{k}_{key}"]], dtype=int)
            pk[f"{k}/{key}"] = bool(np.array_equal(rows[D10.usable_pair[rows]], sp[key]))
    rec = dict(written=C.now(), C_D=dict(ok=bool(all(a_eq.values()) and all(lists.values()) and all(pk.values())),
                                         arrays=a_eq, splits=lists, xgb_pack=pk, pair_list_sha=D10.pair_list_sha))
    rec["ok"] = rec["C_D"]["ok"]
    C.write_json(C.RES / "C_part_cpu.json", rec)
    print(json.dumps(dict(ok=rec["ok"]), indent=1))
    return 0 if rec["ok"] else 3


def _need(name):
    p = C.RES / name
    if not p.is_file():
        C.not_ready(f"{p} missing")
    return C.read_json(p)


def cmd_merge():
    import train_m as TM
    cpu, rt, rg, rx = (_need(n) for n in ("C_part_cpu.json", "CR_tf.json", "CR_gnn.json", "CR_xgb.json"))
    snn, sx = _need("C8_smoke_nn.json"), _need("C8_smoke_xgb.json")
    xsha = {}
    for f in ("xgb_m.py", "ch3m_common.py"):
        xsha[f] = C.sha256(HERE / f)
    dev = None
    rg_pass = bool(rg.get("ok"))
    rg2 = C.read_json(C.RES / "CR_gnn2.json") if (C.RES / "CR_gnn2.json").is_file() else None
    if not rg_pass and rt.get("ok"):
        if rg2 is None:
            C.not_ready("C-RG failed and C-RG2 (train_m.py cr GNN rg2) has not run")
        d = rg2.get("rg2_vs_rg_max_abs_dyhat")
        if d is None:
            C.stop("CR_gnn2.json has no rg2_vs_rg_max_abs_dyhat")
        if d <= RG2_TOL:
            rg_pass = False                                              # deterministic but not LADDER: code difference
        else:
            rg_pass = True                                               # GNN training not deterministic: recorded
            dev = dict(spec="docs/specs/CH3M_MODELS.md 5 (C-RG exception)", written=C.now(),
                       rule="C-RT passed, C-RG failed, C-RG2 != C-RG (test yhat max |diff| > 1e-4): GNN training is not "
                            "deterministic on RTX 3090 -> recorded, not a STOP",
                       rg2_vs_rg_max_abs_dyhat=d, C_RG=dict(checks=rg.get("checks"), max_abs_dyhat=rg.get("max_abs_dyhat"),
                                                            chosen=rg.get("chosen"), chosen_ladder=rg.get("chosen_ladder"),
                                                            max_abs_dval_kcal=rg.get("max_abs_dval_kcal")),
                       C_RG2=dict(checks=rg2.get("checks"), max_abs_dyhat=rg2.get("max_abs_dyhat"), chosen=rg2.get("chosen")))
            C.write_json(C.RES / "C_deviation.json", dev)
    ct_gnn = ((snn.get("runs") or {}).get(C.uname("lf3", "GNN", 0)) or {}).get("C_T") or {}
    checks = {"C-RT TF regression (g2, fold 0)": bool(rt.get("ok")),
              "C-RG GNN regression (g2, fold 0; spec 5 exception)": rg_pass,
              "C-RX XGB regression (g2, fold 0)": bool(rx.get("ok")),
              "C-T GNN 3 mode": bool(ct_gnn.get("ok")),
              "C-T XGB 3 mode": all((sx.get("C_T") or {}).values()) and bool(sx.get("C_T")),
              "C-D data identity": bool(cpu.get("ok")),
              "C-4 / C-5 every fit (smoke + regressions)": bool(snn.get("ok")) and bool(sx.get("leak_C5"))
              and bool(rt.get("checks", {}).get("gates_C4")) and bool(rg.get("checks", {}).get("gates_C4"))
              and bool(rx.get("checks", {}).get("gates")),
              "C-8 smoke GNN + XGB": bool(snn.get("ok")) and bool(sx.get("ok"))}
    tm_code = {f: TM.CODE_SHA[f] for f in TM.TRAINER_FILES}
    same = all((r.get("code_sha256") or {}).get(f) == tm_code[f] for r in (rt, rg, snn) for f in TM.TRAINER_FILES) and \
        all((r.get("code_sha256") or {}).get(f) == xsha[f] for r in (rx, sx) for f in xsha)
    checks["GPU / XGB parts ran the current trainer code"] = bool(same)
    rec = dict(spec="docs/specs/CH3M_MODELS.md 5", written=C.now(), all_pass=bool(all(checks.values())), checks=checks,
               C_RT=dict(chosen=rt.get("chosen"), chosen_ladder=rt.get("chosen_ladder"),
                         max_abs_dval_kcal=rt.get("max_abs_dval_kcal"), max_abs_dyhat=rt.get("max_abs_dyhat")),
               C_RG=dict(chosen=rg.get("chosen"), chosen_ladder=rg.get("chosen_ladder"),
                         max_abs_dval_kcal=rg.get("max_abs_dval_kcal"), max_abs_dyhat=rg.get("max_abs_dyhat"),
                         rg2=None if rg2 is None else dict(max_abs_dyhat=rg2.get("max_abs_dyhat"),
                                                           rg2_vs_rg=rg2.get("rg2_vs_rg_max_abs_dyhat"))),
               C_RX=dict(cells=rx.get("cells"), max_abs_dyhat=rx.get("max_abs_dyhat")),
               C_deviation=dev, trainer_code_sha256=dict(train_m=tm_code, xgb_m=xsha),
               parts_sha256={n: C.sha256(C.RES / n) for n in ("C_part_cpu.json", "CR_tf.json", "CR_gnn.json", "CR_xgb.json",
                                                               "C8_smoke_nn.json", "C8_smoke_xgb.json")})
    C.write_json(C.RES / "C_tests.json", rec)
    print(json.dumps(dict(all_pass=rec["all_pass"], checks=checks), indent=1, ensure_ascii=False))
    return 0 if rec["all_pass"] else 3


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["cpu"]:
        sys.exit(cmd_cpu())
    if a == ["merge"]:
        sys.exit(cmd_merge())
    print(__doc__)
    sys.exit(2)
