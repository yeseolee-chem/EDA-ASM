#!/usr/bin/env python3
r"""predict_c1.py — DEPLOY-V2 (MLTS) Phase K C1 (no retraining; docs/specs/DEPLOY_V2_MLTS.md 10) and the C-6 regression
gate (8). The 40 LADDER dft units L__dft__{S,SH}__{AW,GNN,TF,XGB}__f{0..4} are applied to the fold-k test pairs of an
input rung: 'dft' (= the LADDER dft inputs themselves: C-6) or 'mlts' (the Phase H table $MLT/inputs/mlts.parquet +
tokens $MLT/fu2_mlts/tokens, registered in-process as rung 'mlts'). Compute nodes only; each command re-executes itself
in its environment (inputs_mlts.ensure_env).

  python predict_c1.py atoms <dft|mlts>   GPU, env_d1fu2: the 30 atom units (AW / GNN / TF; 5 saved seed models, saved
                                          std recomputed and required equal) -> $MLT/preds/parts/atoms_<in>.{parquet,json}
  python predict_c1.py xgb <dft|mlts>     CPU, env_espley: the 10 XGB units refitted exactly as LADDER fitted its final
                                          models (no XGB model files were saved) -> $MLT/preds/parts/xgb_<in>.{parquet,json}
  python predict_c1.py c6                 parts of 'dft' vs analysis/d1_ladder/results/L_predictions.parquet ->
                                          results/C6_report.json (exit 3 = STOP: max |dyhat| > 1e-4 kcal/mol)
  python predict_c1.py check [--src S]    CPU, env_d1fu2: Phase H source S (mlts | label) registered as rung 'mlts':
                                          data loading, layout check, deployable pairs per fold (no prediction)
  python predict_c1.py collect            parts of 'mlts' (C-6 must have passed) -> $MLT/preds/C1_predictions.parquet
                                          (L_predictions.parquet columns, rung 'mlts'), $MLT/preds/C1_representatives.json,
                                          results/C1_report.json
  exit 0 done, 2 not ready, 3 STOP.

Reused by import (never copied, never modified): analysis/geo2/deploy/predict_atoms.py unit_run (saved std check
std_equal, TA.final_kcal of the 5 seeds, the per-unit regression vs L_predictions / unit parquet) and layout_check;
predict_xgb.py prepare / finish (TX.prep_fit + d1_train.Scale reconstruction, TX.run_parallel, the same random_state);
deploy_common.py load_ladder_dft / compare_unit / gate_status / deploy_rows (DP-4: a fold-k test pair is predicted when
both ends are usable in the input rung) / pred_frame; analysis/d1_ladder/train_atoms.py and train_xgb.py (LadData /
XData, get_data); analysis/geo2/eval_geo2.py reps / load_units (REP / REP8). The in-process rung registration is the
GEO2b one: train_atoms.RUNGS / train_xgb.RUNGS['mlts'] (fu2_res = results/fu2_mlts, tokens, x182, tag 'mlts' ->
versions fu2-tokens-3-mlts / fu2-atoms-3-mlts) and GEO_INPUTS['mlts'] = the rung table (train_atoms.inputs_path).
Gates: atoms = DP-2 (exact <= 1e-6, else within_fp32_bound <= 1e-4; LADDER units ran on NVIDIA A10, fp32 differences
across GPU types ~3e-6 in GEO2b), XGB = exact; y / base / beta <= 1e-12; same pairs and order. C-6 = every one of the 40
units, deploy path output (input rung 'dft') vs L_predictions.parquet, max |dyhat| <= 1e-4 kcal/mol.
REP / REP8 (C1 = 'LADDER가 고른 것 그대로'): REP = per arm x target the lowest 5-fold mean validation kcal MAE of the dft
units (LADDER 13-1, ties AW -> GNN -> TF -> XGB), checked equal to analysis/d1_ladder/results/E1_report.json
representatives (dft); REP8 = per target over both arms (eval_geo2.py 73-84: ties SH before S, then the model order).
The mlts rung uses the dft choices unchanged (no selection on MLTS inputs).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402
import inputs_mlts as IM  # noqa: E402             (paths, ensure_env; stdlib only at import)

REPO = M.REPO
GEO2_DIR = REPO / "analysis" / "geo2"
GEO2_DEPLOY = GEO2_DIR / "deploy"
LAD_RES = REPO / "analysis" / "d1_ladder" / "results"
E1_REPORT = LAD_RES / "E1_report.json"
PREDS = M.MLT / "preds"
PARTS = PREDS / "parts"
C1_PRED = PREDS / "C1_predictions.parquet"
C1_REPS = PREDS / "C1_representatives.json"
C6_REPORT = M.RES / "C6_report.json"
C1_REPORT = M.RES / "C1_report.json"
RUNG_OUT = "mlts"
C6_TOL = 1e-4
IN_RUNGS = ("dft", "mlts")
T0 = time.time()


def log(msg):
    print(f"[C1 {time.time() - T0:7.0f}s] {msg}", flush=True)


def _geo2_import():
    for p in (str(GEO2_DEPLOY), str(GEO2_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def register_mlts(mod, kind, src="mlts"):
    """rung 'mlts' in train_atoms / train_xgb (in this process only; the GEO2b rung-'geo' pattern). src = the Phase H
    source whose outputs play rung 'mlts' (production: mlts; `check --src label` = the H-1 inputs, tag mltslab)."""
    S = IM.src_cfg(src)
    if kind == "atoms":
        mod.RUNGS["mlts"] = dict(fu2_res=S["fu2_res"], tokens=S["fu2_scr"] / "tokens", tag=S["tag"])
    else:
        mod.RUNGS["mlts"] = dict(fu2_res=S["fu2_res"], x182=S["x182"], tag=S["tag"])
    mod.GEO_INPUTS["mlts"] = S["table"]
    if not Path(S["table"]).is_file():
        M.not_ready(f"{S['table']} missing (inputs_mlts.py build {src})")
    return dict(rung="mlts", source=src, table=str(S["table"]), tokens=str(S["fu2_scr"] / "tokens"),
                fu2_res=str(S["fu2_res"]), x182=str(S["x182"]), tag=S["tag"], table_sha256=M.sha256(S["table"]))


def cmd_check(src="mlts"):
    """registration test (CPU, env_d1fu2): source `src` as rung 'mlts' -> train_atoms.LadData for both arms (token /
    table / label checks of train_atoms), predict_atoms.layout_check vs rung dft, deployable fold-k test pairs per
    fold (DP-4) -> $MLT/preds/check_<src>.json. No prediction."""
    IM.ensure_env("fu2")
    _geo2_import()
    import predict_atoms as PA
    DC, TA = PA.DC, PA.TA
    reg = register_mlts(TA, "atoms", src)
    out = dict(registered=reg, arms={}, written=IM.now())
    for arm in TA.ARMS:
        Dd, Dg = TA.get_data("dft", arm), TA.get_data("mlts", arm)
        try:
            lay = PA.layout_check(Dd, Dg)
        except SystemExit as e:                                     # a subset may lack elements (one-hot width)
            sub = set(Dg.elements) <= set(Dd.elements)
            lay = dict(ok=False, exit=e.code, elements=dict(dft=list(Dd.elements), mlts=list(Dg.elements)),
                       elements_subset=sub, note="atoms / predict need all 9 elements in the token set (one-hot "
                                                 "width d_in); a reaction subset is fine for this check only")
        per = {}
        for k in range(TA.NF):
            te = Dd.split(k)["test"]
            keep, rows_g = DC.deploy_rows(Dd, Dg, te)
            if len(rows_g):
                DC.check_same_pairs(Dd, Dg, te, keep, rows_g)
            per[k] = dict(n_test=int(len(te)), n_deployable=int(len(rows_g)))
        out["arms"][arm] = dict(layout_ok=bool(lay.get("ok")), n_usable_rxn=int(Dg.usable_rxn.sum()),
                                n_token_rxn=int(len(Dg.rids)), folds=per, h_cols=Dg.hinfo["h_cols"])
    M.write_json(PREDS / f"check_{src}.json", IM._clean(out))
    log(f"check {src}: {json.dumps(IM._clean(out['arms']))}")
    return 0


def part_paths(kind, in_rung):
    return PARTS / f"{kind}_{in_rung}.parquet", PARTS / f"{kind}_{in_rung}.json"


def _in_rung(x):
    if x not in IN_RUNGS:
        raise SystemExit(f"input rung must be one of {IN_RUNGS}")
    return x


# ================================================================ atom units (GPU, env_d1fu2)
def cmd_atoms(in_rung):
    IM.ensure_env("fu2")
    _geo2_import()
    import numpy as np
    import pandas as pd
    import predict_atoms as PA
    DC, TA, FT = PA.DC, PA.TA, PA.FT
    in_rung = _in_rung(in_rung)
    reg = register_mlts(TA, "atoms") if in_rung == "mlts" else None
    FT.require_gpu()
    Lp, lp_info = DC.load_ladder_dft(DC.ATOM_MODELS)
    log(f"atoms -> input rung {in_rung}; L_predictions dft atom rows {len(Lp)}; {FT.versions()}")
    units, deps, layout = {}, [], {}
    for arm in TA.ARMS:
        Dd = TA.get_data("dft", arm)
        Dg = Dd if in_rung == "dft" else TA.get_data("mlts", arm)
        if in_rung == "mlts":
            layout[arm] = PA.layout_check(Dd, Dg)
        for model in TA.MODELS:
            for k in range(TA.NF):
                u = dict(rung="dft", arm=arm, model=model, fold=k)
                rec, dep, _ = PA.unit_run(Dd, Dg, u, Lp)
                if in_rung == "dft":
                    rec["deploy_path_vs_l_predictions"] = DC.compare_unit(dep, DC.ladder_rows(Lp, arm, model, k))
                units[TA.uname(u)] = rec
                deps.append(dep)
                c = rec["vs_l_predictions"]
                log(f"{TA.uname(u)}: test {rec['n_test']}, predicted {rec['n_deploy']}, regression |dyhat| "
                    f"{c['max_abs_yhat']:.3g}")
    vs = [v["vs_l_predictions"] for v in units.values()] + [v["vs_unit_parquet"] for v in units.values()]
    dmax = max(v["max_abs_yhat"] for v in vs)
    pairs_ok = all(v["same_pairs"] and v["same_order"] for v in vs)
    others = {c: max(v[f"max_abs_{c}"] for v in vs) for c in ("y", "base", "beta")}
    status = DC.gate_status(dmax, allow_fp32=True)
    passed = bool(status != "fail" and pairs_ok and all(x <= DC.Y_TOL for x in others.values()))
    gpu = FT.versions().get("gpu")
    out = dict(phase="DEPLOY-V2 K C1 atoms", input_rung=in_rung, registered=reg, layout=layout,
               regression_gate=dict(status=status, passed=passed, max_abs_yhat=dmax, max_abs_y_base_beta=others,
                                    same_pairs_and_order=pairs_ok, gpu=gpu,
                                    unit_gpus=sorted({str(v["unit_gpu"]) for v in units.values()}),
                                    rule="geo2 DP-2: saved models + recomputed std on their DFT test inputs vs "
                                         "L_predictions / unit parquet: exact <= 1e-6, within_fp32_bound <= 1e-4"),
               l_predictions=lp_info, units=units, versions=FT.versions(), train_atoms_code_sha256=TA.CODE_SHA,
               predict_atoms_sha256=M.sha256(Path(PA.__file__).resolve()), written=IM.now(),
               seconds=time.time() - T0)
    pq_, js = part_paths("atoms", in_rung)
    if not passed:
        M.write_json(js, IM._clean(out))
        M.stop(f"atom regression gate failed: {status}, max |dyhat| {dmax:.3g}, pairs {pairs_ok}, {others}")
    D = pd.concat(deps, ignore_index=True)[DC.PRED_COLS]
    D["rung"] = RUNG_OUT if in_rung == "mlts" else "dft"
    if D.duplicated(["pair_id", "arm", "model", "target"]).any():
        M.stop("duplicated atom prediction rows (a pair in two folds?)")
    if not np.isfinite(D.yhat.to_numpy(float)).all():
        M.stop("non-finite atom predictions")
    M.write_parquet(pq_, D)
    out["part"] = dict(path=str(pq_), sha256=M.sha256(pq_), n_rows=int(len(D)),
                       n_pairs_by_fold={int(k): int(g.pair_id.nunique()) for k, g in D.groupby("fold")})
    M.write_json(js, IM._clean(out))
    log(f"atoms {in_rung}: gate {status} (max |dyhat| {dmax:.3g}), {len(D)} rows -> {pq_}")
    return 0


# ================================================================ XGB units (CPU, env_espley)
def cmd_xgb(in_rung):
    IM.ensure_env("espley")
    _geo2_import()
    import numpy as np
    import pandas as pd
    import predict_xgb as PX
    DC, TX = PX.DC, PX.TX
    in_rung = _in_rung(in_rung)
    reg = register_mlts(TX, "xgb") if in_rung == "mlts" else None
    Lp, lp_info = DC.load_ladder_dft(("XGB",))
    log(f"xgb -> input rung {in_rung}; L_predictions dft XGB rows {len(Lp)}; n_jobs {TX.N_JOBS}")
    ctxs, data, feat = [], {}, {}
    for arm in TX.ARMS:
        Dd = TX.get_data("dft", arm)
        Dg = Dd if in_rung == "dft" else TX.get_data("mlts", arm)
        data[arm] = (Dd, Dg)
        feat[arm] = dict(n=len(Dd.feat_cols), equal=list(Dd.feat_cols) == list(Dg.feat_cols),
                         glob_cols_equal=list(Dd.glob_cols) == list(Dg.glob_cols))
        if not (feat[arm]["equal"] and feat[arm]["glob_cols_equal"]):
            M.stop(f"XGB feature columns of rung {in_rung} differ from rung dft ({arm})")
        for k in range(TX.NF):
            ctxs.append(PX.prepare(Dd, Dg, dict(rung="dft", arm=arm, model=TX.MODEL, fold=k), Lp))
    calls = [(i, ti) for i, c in enumerate(ctxs) for ti in range(TX.NT)]
    order = sorted(range(len(calls)), key=lambda q: -TX.cost(TX.GRID[ctxs[calls[q][0]]["cells"][TX.TARGETS[calls[q][1]]]]))
    log(f"{len(calls)} final fits on {TX.N_JOBS} workers")
    res_o = TX.run_parallel([ctxs[calls[q][0]]["calls"][calls[q][1]] for q in order])
    res = [None] * len(calls)
    for q, r in zip(order, res_o):
        res[q] = r
    units, deps = {}, []
    for i, ctx in enumerate(ctxs):
        Dd, Dg = data[ctx["u"]["arm"]]
        rec, dep, _ = PX.finish(Dd, Dg, ctx, [res[j] for j, (ii, _) in enumerate(calls) if ii == i], Lp)
        if in_rung == "dft":
            u = ctx["u"]
            rec["deploy_path_vs_l_predictions"] = DC.compare_unit(dep, DC.ladder_rows(Lp, u["arm"], "XGB", u["fold"]))
        units[ctx["name"]] = rec
        deps.append(dep)
        log(f"{ctx['name']}: regression |dyhat| {rec['vs_l_predictions']['max_abs_yhat']:.3g}, json equal "
            f"{rec['json_values_equal']}")
    vs = [v["vs_l_predictions"] for v in units.values()] + [v["vs_unit_parquet"] for v in units.values()]
    dmax = max(v["max_abs_yhat"] for v in vs)
    pairs_ok = all(v["same_pairs"] and v["same_order"] for v in vs)
    others = {c: max(v[f"max_abs_{c}"] for v in vs) for c in ("y", "base", "beta", "yhat_seed_sd")}
    json_ok = all(v["json_values_equal"] for v in units.values())
    status = DC.gate_status(dmax, allow_fp32=False)
    passed = bool(status == "exact" and pairs_ok and json_ok and all(x <= DC.Y_TOL for x in others.values()))
    out = dict(phase="DEPLOY-V2 K C1 XGB", input_rung=in_rung, registered=reg, features=feat,
               regression_gate=dict(status=status, passed=passed, max_abs_yhat=dmax, max_abs_y_base_beta_sd=others,
                                    same_pairs_and_order=pairs_ok, json_values_equal=json_ok,
                                    rule="geo2 DP-2 XGB: final refit with the unit json cells, exact <= 1e-6"),
               l_predictions=lp_info, units=units, versions=TX.versions(), n_jobs=TX.N_JOBS,
               train_xgb_code_sha256=TX.CODE_SHA, predict_xgb_sha256=M.sha256(Path(PX.__file__).resolve()),
               written=IM.now(), seconds=time.time() - T0)
    pq_, js = part_paths("xgb", in_rung)
    if not passed:
        M.write_json(js, IM._clean(out))
        M.stop(f"XGB regression gate failed: {status}, max |dyhat| {dmax:.3g}, pairs {pairs_ok}, json {json_ok}")
    D = pd.concat(deps, ignore_index=True)[DC.PRED_COLS]
    D["rung"] = RUNG_OUT if in_rung == "mlts" else "dft"
    if D.duplicated(["pair_id", "arm", "model", "target"]).any():
        M.stop("duplicated XGB prediction rows")
    if not np.isfinite(D.yhat.to_numpy(float)).all():
        M.stop("non-finite XGB predictions")
    M.write_parquet(pq_, D)
    out["part"] = dict(path=str(pq_), sha256=M.sha256(pq_), n_rows=int(len(D)))
    M.write_json(js, IM._clean(out))
    log(f"xgb {in_rung}: gate {status} (max |dyhat| {dmax:.3g}), {len(D)} rows -> {pq_}")
    return 0


# ================================================================ C-6 / collect (reactot)
def _load_part(kind, in_rung):
    import pandas as pd
    pq_, js = part_paths(kind, in_rung)
    if not (pq_.is_file() and js.is_file()):
        M.not_ready(f"{pq_} / {js} missing (predict_c1.py {kind} {in_rung})")
    rec = M.read_json(js)
    if (rec.get("part") or {}).get("sha256") != M.sha256(pq_):
        M.stop(f"{pq_} changed after {js.name}")
    return pd.read_parquet(pq_), rec


def cmd_c6():
    IM.ensure_env("reactot")
    _geo2_import()
    import numpy as np
    import pandas as pd
    import deploy_common as DC
    Lp, lp_info = DC.load_ladder_dft(DC.MODELS)
    parts, recs = [], {}
    for kind in ("atoms", "xgb"):
        D, r = _load_part(kind, "dft")
        parts.append(D)
        recs[kind] = r
    D = pd.concat(parts, ignore_index=True)
    D["pair_id"] = D.pair_id.astype(str)
    units, worst = {}, (0.0, None)
    for arm in DC.ARMS:
        for model in DC.MODELS:
            for k in range(DC.NF):
                mine = D[(D.arm == arm) & (D.model == model) & (D.fold == k)]
                c = DC.compare_unit(mine, DC.ladder_rows(Lp, arm, model, k))
                name = f"L__dft__{arm}__{model}__f{k}"
                units[name] = c
                if c["max_abs_yhat"] > worst[0] or worst[1] is None:
                    worst = (c["max_abs_yhat"], name)
    dmax = max(v["max_abs_yhat"] for v in units.values())
    pairs_ok = all(v["same_pairs"] and v["same_order"] for v in units.values()) and len(units) == 40
    others = {c: max(v[f"max_abs_{c}"] for v in units.values()) for c in ("y", "base", "beta", "yhat_seed_sd")}
    n_rows = dict(mine=int(len(D)), ladder=int(len(Lp)))
    passed = bool(np.isfinite(dmax) and dmax <= C6_TOL and pairs_ok and n_rows["mine"] == n_rows["ladder"]
                  and others["y"] <= DC.Y_TOL and others["base"] <= C6_TOL and others["beta"] <= DC.Y_TOL)
    by_model = {mo: max(v["max_abs_yhat"] for n, v in units.items() if f"__{mo}__" in n) for mo in DC.MODELS}
    rep = dict(spec="DEPLOY_V2_MLTS.md 8 C-6", written=IM.now(), tol_kcal=C6_TOL, passed=passed,
               max_abs_yhat=dmax, worst_unit=worst[1], max_abs_by_model=by_model, max_abs_other=others,
               same_pairs_and_order_all_40=pairs_ok, n_units=len(units), n_rows=n_rows,
               gpu=(recs["atoms"].get("regression_gate") or {}).get("gpu"),
               unit_gpus=(recs["atoms"].get("regression_gate") or {}).get("unit_gpus"),
               atoms_gate=recs["atoms"].get("regression_gate"), xgb_gate=recs["xgb"].get("regression_gate"),
               definition="predict_c1.py atoms / xgb with input rung 'dft' (the C1 deploy path: the dft unit's saved "
                          "models + std / refit applied to the input rung's test pairs) vs L_predictions.parquet, all "
                          "40 dft units, every column (yhat, y, base, beta, yhat_seed_sd); base may differ by the "
                          "GPU fp32 bound only through beta * dH (beta exact)",
               l_predictions=lp_info, units=units)
    M.write_json(C6_REPORT, IM._clean(rep))
    log(f"C-6: max |dyhat| {dmax:.3g} kcal/mol ({worst[1]}), pairs ok {pairs_ok} -> passed {passed}")
    if not passed:
        print(f"STOP: C-6 regression failed (results/C6_report.json)", file=sys.stderr)
        return 3
    return 0


def representatives():
    """REP / REP8 of the dft units (eval_geo2.reps on L_report + L_xgb_report), REP checked against E1_report."""
    _geo2_import()
    import eval_geo2 as EG
    rep, rep8, m = EG.reps(EG.load_units(EG.BASE_UNITS))
    e1 = (M.read_json(E1_REPORT).get("representatives") or {}).get("dft") or {}
    diff = [(a, t, rep.get(("dft", a, t)), (e1.get(a) or {}).get(t, {}).get("model")) for a in ("S", "SH")
            for t in EG.TARGETS if rep.get(("dft", a, t)) != (e1.get(a) or {}).get(t, {}).get("model")]
    if diff:
        M.stop(f"REP of the dft units differs from E1_report.json representatives: {diff[:5]}")
    out = dict(rule_REP="LADDER 13-1: per rung x arm x target the lowest 5-fold mean val kcal MAE, ties AW -> GNN -> TF "
                        "-> XGB (= E1_report.json representatives, checked)",
               rule_REP8="eval_geo2.py 73-84: per rung x target over both arms, ties SH before S, then the model order",
               REP={r: {a: {t: rep[("dft", a, t)] for t in EG.TARGETS} for a in ("S", "SH")} for r in ("dft", RUNG_OUT)},
               REP8={r: {t: list(rep8[("dft", t)]) for t in EG.TARGETS} for r in ("dft", RUNG_OUT)},
               note=f"rung {RUNG_OUT} = the dft choices unchanged (C1: no reselection)",
               sources=[str(p) for p in EG.BASE_UNITS] + [str(E1_REPORT)])
    return out


def cmd_collect():
    IM.ensure_env("reactot")
    import numpy as np
    import pandas as pd
    if not C6_REPORT.is_file() or not M.read_json(C6_REPORT).get("passed"):
        M.not_ready(f"{C6_REPORT} missing or not passed (C-6 first)")
    parts, recs = [], {}
    for kind in ("atoms", "xgb"):
        D, r = _load_part(kind, "mlts")
        if not (r.get("regression_gate") or {}).get("passed"):
            M.stop(f"{kind} mlts part without a passed regression gate")
        parts.append(D)
        recs[kind] = r
    D = pd.concat(parts, ignore_index=True)
    if set(D.rung) != {RUNG_OUT} or D.duplicated(["pair_id", "arm", "model", "target"]).any():
        M.stop("collect: rung label / duplicated rows")
    lists = {}
    for k, g in D.groupby("fold"):
        s = {(a, mo): tuple(sorted(set(x.pair_id))) for (a, mo), x in g.groupby(["arm", "model"])}
        lists[int(k)] = len(set(s.values())) == 1 and len(s) == 8
    if not all(lists.values()):
        M.stop(f"collect: the 8 units of a fold predict different pair lists: {lists}")
    Lc = pd.read_parquet(LAD_RES / "L_predictions.parquet", columns=["pair_id", "rung", "model", "arm", "target"])
    Lc = Lc[(Lc.rung == "dft")]
    n_dft_pairs = int(Lc.pair_id.nunique())
    reps = representatives()
    M.write_parquet(C1_PRED, D)
    M.write_json(C1_REPS, reps)
    rep = dict(spec="DEPLOY_V2_MLTS.md 10 C1", written=IM.now(), predictions=str(C1_PRED), sha256=M.sha256(C1_PRED),
               n_rows=int(len(D)), n_pairs=int(D.pair_id.nunique()), n_dft_test_pairs=n_dft_pairs,
               n_pairs_by_fold={int(k): int(g.pair_id.nunique()) for k, g in D.groupby("fold")},
               same_pair_list_per_fold=lists, representatives=str(C1_REPS),
               gates=dict(atoms=recs["atoms"].get("regression_gate"), xgb=recs["xgb"].get("regression_gate")),
               c6=dict(path=str(C6_REPORT), sha256=M.sha256(C6_REPORT)),
               note="fold-k test pairs predicted by the fold-k dft units when both ends are usable in rung mlts "
                    "(geo2 DP-4); the MLTS structure of a fold-k reaction must come from FT_k (A-4; checked by C-2)")
    M.write_json(C1_REPORT, IM._clean(rep))
    log(f"collect: {len(D)} rows, {rep['n_pairs']} pairs -> {C1_PRED}")
    return 0


def main(a):
    if a[:1] == ["atoms"] and len(a) == 2:
        return cmd_atoms(a[1])
    if a[:1] == ["xgb"] and len(a) == 2:
        return cmd_xgb(a[1])
    if a == ["c6"]:
        return cmd_c6()
    if a == ["collect"]:
        return cmd_collect()
    if a == ["reps"]:
        IM.ensure_env("reactot")
        r = representatives()
        M.write_json(C1_REPS, r)
        log(f"reps -> {C1_REPS}: REP8 {r['REP8']['dft']}")
        return 0
    if a[:1] == ["check"] and len(a) in (1, 3) and (len(a) == 1 or a[1] == "--src"):
        return cmd_check(a[2] if len(a) == 3 else "mlts")
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
