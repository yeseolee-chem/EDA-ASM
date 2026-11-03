#!/usr/bin/env python3
r"""predict_xgb.py — GEO2b DEPLOY, XGB (analysis/geo2/results/DECISIONS_deploy.md DP-1 .. DP-3). LADDER saved no XGB model
files, so each of the 10 dft XGB units L__dft__<arm>__XGB__f<k> is refitted exactly as LADDER fitted its final models
(train_xgb.py code path, same pairs / order / per-target cell recorded in the unit json / random_state), and the same
fitted model predicts (1) the unit's DFT test pairs = regression gate against analysis/d1_ladder/results/
L_predictions.parquet and (2) the geo-rung inputs of the same pairs (deployment, rung label 'dft_on_geo').
CPU, env_espley (xgboost 3.2.0, scikit-learn 1.3.1; analysis/geo2/jobs/m3_cpu.slurm M3ENV=espley). Not run when written.

  python predict_xgb.py run
        -> analysis/geo2/results/DEPLOY_regress_xgb.json and, if the gate passes, $GEO2B/deploy/parts/xgb.parquet.
           Exit 0 done, 2 not ready, 3 STOP (gate failed: no part written). Rerun = the same command (deterministic).

How a prediction is produced (LADDER code imported, not copied; analysis/d1_ladder/train_xgb.py = TX, d1_pair/d1_train.py)
  data      TX.get_data(rung, arm) (train_xgb.py 933-940) -> TX.XData (837-927): X = the rung's X182 file rows
            ($LAD/X182_d1_dft.parquet | $GEO2B/X182_d1_geo.parquet, T_report glob_cols order) for S, [X182 | H 9 of the
            rung input table] for SH; dH = H(a) - H(b) of the rung; the usable-pair filter = train_atoms'.
  refit     = train_xgb.run_unit final part (1373-1387): fin = TX.prep_fit(Dd, train, {test, train}, gate pairs)
            (957-987: SH beta = TX.beta_fit on the training pairs, r = y - beta dH, d1_train.Scale on the training pairs
            (d1_train.py 159-172), z = [dx / s_delta, (xbar - mu) / s], ys = r / s_y), then per target TX._task
            (1008-1022) = d1_train._xgb_fit (227-232: clone of GRIDS["XGB"][0] + the unit's chosen cell, both pair
            orders) and kcal = d1_train._xgb_g (235-236, 1/2 [h(z) - h(-dx, xbar)]) x s_y, through TX.run_parallel
            (1025-1030; every fit n_jobs 1, so the worker count does not change results). yhat = base + kcal with
            base = beta dH (SH targets but c_ghost) or 0 (= train_xgb.py 1384). Gate pairs = run_unit's rng rule
            (1324-1326, seeded by the unit name) so gate M-1' can be compared with the unit json too.
  deploy    the same fitted model and the SAME d1_train.Scale (reconstructed from the training pairs; required to give
            exactly fin's test z) map the geo X of the deployed pairs to z; base = the dft beta x geo dH. No refit on geo.
Gate (DP-2): max |yhat - L_predictions yhat| <= 1e-6 kcal/mol over the 10 units ('exact'; CPU, deterministic; R-3 of
  GEO2b reproduced a full retrain with difference 0), y / base / beta <= 1e-12, final train MAE and gate M-1' values =
  the unit json (1e-9); unit fingerprints may differ only in code_sha256 (train_xgb.py).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy_common as DC  # noqa: E402

sys.path.insert(0, str(DC.LAD_DIR))
import train_xgb as TX  # noqa: E402     (LADDER XGB trainer: XData, prep_fit, run_parallel, gate_eval, fingerprint, ...)

D1T, C = TX.D1T, TX.C
if list(TX.TARGETS) != DC.TARGETS or tuple(TX.ARMS) != DC.ARMS:
    DC.stop("train_xgb TARGETS / ARMS differ from deploy_common")
NT = TX.NT
JSON_TOL = 1e-9                                   # train MAE / gate values vs the unit json (json round trip is exact)


def gate_pairs(name, te):
    """= train_xgb.run_unit 1324-1326 (T1: no D0 training pairs, so no extra D0 gate pairs)."""
    rng = np.random.default_rng(C.SEED + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 10 ** 6)
    return te[rng.choice(len(te), size=min(C.ANTISYM_N, len(te)), replace=False)]


def prepare(Dd, Dg, u, Lp):
    """one dft XGB unit -> context with the 10 final-fit calls (evals: dft test / train / gate pairs + geo)."""
    name = TX.uname(u)
    js, pq = TX.UNITS / f"{name}.json", TX.UNITS / f"{name}.parquet"
    if not (js.is_file() and pq.is_file()):
        DC.not_ready(f"{js} / {pq} missing")
    rec = json.loads(js.read_text())
    if rec.get("smoke") or not rec.get("gates_ok"):
        DC.stop(f"{name}: smoke unit or gates not ok")
    if TX.L.sha256(pq) != rec.get("parquet_sha256"):
        DC.stop(f"{pq} changed after its json")
    sp = Dd.split(u["fold"])
    fpd = DC.fingerprint_diff(rec.get("fingerprint"), TX.fingerprint(Dd, u, sp, list(range(len(TX.GRID)))),
                              ("code_sha256",), ("train_xgb.py",))
    if not fpd["ok"]:
        DC.stop(f"{name}: fingerprint differs beyond code_sha256 (train_xgb.py): {fpd}")
    if rec.get("test_pairs_sha256") != sp["test_sha"] or rec.get("val_pairs_sha256") != sp["val_sha"]:
        DC.stop(f"{name}: test / validation pairs differ from the current split")
    tr, te = sp["train"], sp["test"]
    gp = gate_pairs(name, te)
    if int(rec.get("n_gate_pairs", -1)) != len(gp):
        DC.stop(f"{name}: {len(gp)} gate pairs, unit json {rec.get('n_gate_pairs')}")
    fin = TX.prep_fit(Dd, tr, dict(test=te, train=tr), gp)
    leak = TX.leak_check(Dd, fin, (te,), te)
    if not np.array_equal(np.asarray(fin["beta"], float), np.asarray(rec.get("beta"), float)):
        DC.stop(f"{name}: beta differs from the unit json")
    # the fit's x scaling, reconstructed (prep_fit keeps only its z): must give exactly prep_fit's test z
    sc = D1T.Scale(Dd.X[Dd.ra[tr]], Dd.X[Dd.rb[tr]], Dd.y[tr, 0])
    zt = sc.z(Dd.X[Dd.ra[te]], Dd.X[Dd.rb[te]])
    if not (np.array_equal(zt[0], fin["evals"]["test"][0]) and np.array_equal(zt[1], fin["evals"]["test"][1])):
        DC.stop(f"{name}: the reconstructed d1_train.Scale does not reproduce prep_fit's test z")
    keep, rows_g = DC.deploy_rows(Dd, Dg, te)
    dy_pairs = DC.check_same_pairs(Dd, Dg, te, keep, rows_g)
    zg = sc.z(Dg.X[Dg.ra[rows_g]], Dg.X[Dg.rb[rows_g]])
    on = [u["arm"] == "SH" and j in TX.H_IDX for j in range(NT)]
    base_g = np.stack([fin["beta"][j] * Dg.dH[rows_g, j] if on[j] else np.zeros(len(rows_g)) for j in range(NT)], 1)
    evals = dict(fin["evals"], geo=zg)
    cells, calls = {}, []
    for ti, t in enumerate(TX.TARGETS):
        ch = (rec.get("chosen") or {}).get(t) or {}
        gi = int(ch.get("grid_index", -1))
        if not (0 <= gi < len(TX.GRID)) or TX.norm(TX.GRID[gi]) != TX.norm(ch.get("cell")):
            DC.stop(f"{name} {t}: chosen cell {ch} is not GRID[{gi}]")
        fj = (rec.get("final") or {}).get(t) or {}
        if fj.get("grid_index") != gi or float(fj.get("sy", np.nan)) != float(fin["sy"][ti]) \
                or float(fj.get("beta", np.nan)) != float(fin["beta"][ti]):
            DC.stop(f"{name} {t}: final grid_index / s_y / beta differ from the unit json")
        cells[t] = gi
        calls.append((fin["zd"], fin["zm"], fin["ys"][ti], fin["sy"][ti], TX.GRID[gi], evals))
    return dict(u=u, name=name, rec=rec, pq=pq, tr=tr, te=te, gp=gp, fin=fin, keep=keep, rows_g=rows_g,
                base_g=base_g, cells=cells, calls=calls, fpd=fpd, leak=leak, dy_pairs=dy_pairs)


def finish(Dd, Dg, ctx, res, Lp):
    """fitted kcal predictions of one unit -> (record, deploy rows)."""
    u, rec, fin, te, tr = ctx["u"], ctx["rec"], ctx["fin"], ctx["te"], ctx["tr"]
    yd = np.zeros((len(te), NT))
    yg = np.zeros((len(ctx["rows_g"]), NT))
    per = {}
    for ti, (t, (kc, sec)) in enumerate(zip(TX.TARGETS, res)):
        yd[:, ti] = fin["base"]["test"][:, ti] + kc["test"]
        yg[:, ti] = ctx["base_g"][:, ti] + kc["geo"]
        ptr = fin["base"]["train"][:, ti] + kc["train"]
        tm = float(np.mean(np.abs(ptr - Dd.y[tr, ti])))
        g = TX.gate_eval(kc, fin["base"], ti)
        fj = rec["final"][t]
        gj = fj.get("gate_M1") or {}
        dg = max(abs(float(g[k]) - float(gj.get(k, np.nan))) for k in ("anti_max", "self_max", "model_anti_max",
                                                                           "model_self_max"))
        per[t] = dict(cell=TX.GRID[ctx["cells"][t]], train_mae=tm, train_mae_json=fj.get("train_mae"),
                      train_mae_abs_diff=abs(tm - float(fj.get("train_mae", np.nan))), gate_M1=g,
                      gate_M1_max_abs_diff_vs_json=dg, seconds=float(sec))
    beta = np.asarray(fin["beta"], float)
    mine = DC.pred_frame(Dd, te, u, yd[None], fin["base"]["test"], beta, DC.SRC_RUNG)
    dep = DC.pred_frame(Dg, ctx["rows_g"], u, yg[None], ctx["base_g"], beta, DC.RUNG_DEPLOY)
    if not np.isfinite(dep.yhat.to_numpy(float)).all():
        DC.stop(f"{ctx['name']}: non-finite deploy predictions")
    c_unit = DC.compare_unit(mine, pd.read_parquet(ctx["pq"]))
    c_lad = DC.compare_unit(mine, DC.ladder_rows(Lp, u["arm"], u["model"], u["fold"]))
    json_ok = all(np.isfinite(v["train_mae_abs_diff"]) and v["train_mae_abs_diff"] <= JSON_TOL
                  and np.isfinite(v["gate_M1_max_abs_diff_vs_json"]) and v["gate_M1_max_abs_diff_vs_json"] <= JSON_TOL
                  and v["gate_M1"]["ok"] for v in per.values())
    out = dict(unit=u, n_test=int(len(te)), n_deploy=int(len(ctx["rows_g"])), n_not_usable_in_geo=int((~ctx["keep"]).sum()),
               n_gate_pairs=int(len(ctx["gp"])), vs_l_predictions=c_lad, vs_unit_parquet=c_unit, targets=per,
               json_values_equal=bool(json_ok), y_dft_vs_geo_max_abs=ctx["dy_pairs"], fingerprint_diff=ctx["fpd"],
               leak=ctx["leak"], final_scale=fin["scale"], unit_host=rec.get("host"))
    return out, dep, te[ctx["keep"]]


def cmd_run():
    t0 = time.time()
    Lp, lp_info = DC.load_ladder_dft(("XGB",))
    DC.log(t0, f"L_predictions dft XGB rows {len(Lp)}; {TX.versions()}; n_jobs {TX.N_JOBS}")
    ctxs, data, feat = [], {}, {}
    for arm in TX.ARMS:
        Dd = TX.get_data(DC.SRC_RUNG, arm)
        Dg = TX.get_data(DC.IN_RUNG, arm)
        data[arm] = (Dd, Dg)
        feat[arm] = dict(n=len(Dd.feat_cols), equal=list(Dd.feat_cols) == list(Dg.feat_cols),
                         glob_cols_equal=list(Dd.glob_cols) == list(Dg.glob_cols), feat_cols_sha256=Dd.feat_cols_sha,
                         h_cols=dict(dft=Dd.hinfo["h_cols"], geo=Dg.hinfo["h_cols"]),
                         x182=dict(dft=Dd.inputs.get("x182"), geo=Dg.inputs.get("x182")))
        if not (feat[arm]["equal"] and feat[arm]["glob_cols_equal"]):
            DC.stop(f"XGB feature columns of rung geo differ from rung dft ({arm})")
        for k in range(TX.NF):
            ctxs.append(prepare(Dd, Dg, dict(rung=DC.SRC_RUNG, arm=arm, model=TX.MODEL, fold=k), Lp))
            DC.log(t0, f"{ctxs[-1]['name']}: prepared (test {len(ctxs[-1]['te'])}, deploy {len(ctxs[-1]['rows_g'])})")
    calls = [(i, ti) for i, c in enumerate(ctxs) for ti in range(NT)]
    order = sorted(range(len(calls)), key=lambda q: -TX.cost(TX.GRID[ctxs[calls[q][0]]["cells"][TX.TARGETS[calls[q][1]]]]))
    DC.log(t0, f"{len(calls)} final fits on {TX.N_JOBS} workers")
    res_o = TX.run_parallel([ctxs[calls[q][0]]["calls"][calls[q][1]] for q in order])
    res = [None] * len(calls)
    for q, r in zip(order, res_o):
        res[q] = r
    units, deps, kept = {}, [], {}
    for i, ctx in enumerate(ctxs):
        Dd, Dg = data[ctx["u"]["arm"]]
        rec, dep, te_kept = finish(Dd, Dg, ctx, [res[j] for j, (ii, _) in enumerate(calls) if ii == i], Lp)
        units[ctx["name"]] = rec
        deps.append(dep)
        kept[ctx["name"]] = sorted(np.asarray(Dd.P.index)[te_kept].astype(str).tolist())
        DC.log(t0, f"{ctx['name']}: |dyhat| vs L {rec['vs_l_predictions']['max_abs_yhat']:.3g}, json values equal "
                   f"{rec['json_values_equal']}")
    bad_lists = [f"f{k}" for k in range(TX.NF) if len({tuple(v) for n, v in kept.items() if n.endswith(f"__f{k}")}) != 1]
    if bad_lists:
        DC.stop(f"deployed pair lists differ between the S / SH XGB units of {bad_lists}")
    vs = [v["vs_l_predictions"] for v in units.values()] + [v["vs_unit_parquet"] for v in units.values()]
    dmax = max(v["max_abs_yhat"] for v in vs)
    pairs_ok = all(v["same_pairs"] and v["same_order"] for v in vs)
    others = {c: max(v[f"max_abs_{c}"] for v in vs) for c in ("y", "base", "beta", "yhat_seed_sd")}
    json_ok = all(v["json_values_equal"] for v in units.values())
    status = DC.gate_status(dmax, allow_fp32=False)
    passed = bool(status == "exact" and pairs_ok and json_ok and all(x <= DC.Y_TOL for x in others.values()))
    gate = dict(status=status, passed=passed, max_abs_yhat=dmax, max_abs_y_base_beta_sd=others,
                same_pairs_and_order=pairs_ok, json_values_equal=json_ok, tol_exact=DC.TOL_EXACT,
                rule="DP-2: 10 dft XGB units refitted (final fit, unit json cells) and predicted on their DFT test inputs vs "
                     "L_predictions.parquet / unit parquet: |dyhat| <= 1e-6 kcal/mol, y / base / beta <= 1e-12, same pairs "
                     "and order, train MAE and gate M-1' = unit json (1e-9); else fail (no deploy part)")
    out = dict(phase="GEO2b DEPLOY XGB (DP-1 .. DP-3)", gate=gate, features=feat, l_predictions=lp_info, units=units,
               versions=TX.versions(), n_jobs=TX.N_JOBS, env=DC.env_info(), code_sha256=DC.code_sha(),
               train_xgb_code_sha256=TX.CODE_SHA, decisions=str(DC.DECISIONS_MD),
               decisions_sha256=DC.sha256(DC.DECISIONS_MD) if DC.DECISIONS_MD.is_file() else None)
    if not passed:
        out.update(deploy=None, seconds=time.time() - t0)
        DC.write_json(DC.REG_XGB, out)
        DC.stop(f"XGB regression gate failed: {status}, max |dyhat| {dmax:.3g}, pairs ok {pairs_ok}, json {json_ok}, "
                f"{others} -> {DC.REG_XGB} (no deploy part written)")
    D = pd.concat(deps, ignore_index=True)[DC.PRED_COLS]
    if D.duplicated(["pair_id", "arm", "model", "target"]).any():
        DC.stop("duplicated XGB deploy rows")
    DC.write_parquet(DC.PART_XGB, D)
    out["deploy"] = dict(part=str(DC.PART_XGB), sha256=DC.sha256(DC.PART_XGB), n_rows=int(len(D)),
                         n_pairs_by_fold={int(k): int(g.pair_id.nunique()) for k, g in D.groupby("fold")},
                         rung=DC.RUNG_DEPLOY)
    out["seconds"] = time.time() - t0
    DC.write_json(DC.REG_XGB, out)
    DC.log(t0, f"gate {status} (max |dyhat| {dmax:.3g}); {len(D)} deploy rows -> {DC.PART_XGB}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        sys.exit(cmd_run())
    raise SystemExit(__doc__)
