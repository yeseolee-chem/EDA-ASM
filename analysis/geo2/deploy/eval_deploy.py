#!/usr/bin/env python3
r"""eval_deploy.py — GEO2b DEPLOY evaluation, D1 only (analysis/geo2/results/DECISIONS_deploy.md DP-4 .. DP-6):
  (a) 'dft'        the LADDER dft rung on its DFT inputs (analysis/d1_ladder/results/L_predictions.parquet),
  (b) 'dft_on_geo' the same 40 units on the geo-rung inputs (deployment; $GEO2B/deploy/L_predictions_deploy.parquet),
  (c) 'geo'        the GEO2b geo-trained units on the geo inputs (analysis/d1_ladder/results/L_predictions_geo.parquet D1),
all on the common pair set (D1 T1 test pairs usable in dft AND geo = the pairs (b) predicts). Descriptive; the only rule
applied is the LADDER goal check (spec 13-2) for (b) (and, descriptively, for (a) / (c) on the same pairs).
env_espley (analysis/geo2/jobs/m3_cpu.slurm M3ENV=espley; = LADDER evaluate.py E1-1). Nothing was run when written.

  python eval_deploy.py selftest   LADDER / GEO2b files only: (i) the pipeline on the full LADDER dft predictions
                                   reproduces results/E2_metrics.csv, E2_goal.csv, E2_bend_rates.json (dft, own);
                                   (ii) b := a (relabelled, on A & C) gives zero differences and identical tables;
                                   (iii) cached bootstrap = eval_geo2.boot_diff -> results/DEPLOY_selftest.json (exit 3)
  python eval_deploy.py collect    parts/{atoms,xgb}.parquet (both regression gates passed, compat ok) ->
                                   $GEO2B/deploy/L_predictions_deploy.parquet + results/DEPLOY_collect.json
  python eval_deploy.py run        -> results/DEPLOY_metrics.csv (case x selection x target x position: n, MAE, core
                                   bootstrap CI), DEPLOY_compare.csv (d = MAE(b) - MAE(a) main, MAE(b) - MAE(c)
                                   descriptive; CI, p), DEPLOY_goal.csv, DEPLOY_metrics_ladder.csv (LADDER metrics_e1
                                   format incl. LINE / null, R2_fam), DEPLOY_bends.csv, DEPLOY_bend_rates.json,
                                   DEPLOY_report.json
Reused (imported, unchanged): analysis/d1_ladder/evaluate.py (LE) fu2_eval 395-473, label_info 481-497, prep_S 655-669,
  read_lab 672-678, ref_set 681-690, model_frames 739-750, check_pair_attrs 753-760, derived_frames 790-811, gate_pairs
  814-829, check_line 832-854, annotate_all 861-873, metrics_e1 876-909, met_index 912-918, bends_e1 921-937, goal_e1
  940-991; analysis/geo2/eval_geo2.py (EG) load_units 58-70, reps 73-84, pairs_frame 87-98, select 113-130, boot_diff
  41-54; analysis/geo2/eval_d0d1.py (ED) load_units_geo 131-160, add_bend_passed 239-248, goal_rep8 251-294,
  boot_mae_ci 447-460, _num_close / _same / _clean / _stop3. rung_part below = eval_d0d1.d1_part 297-351 with the rung
  as an argument (d1_part is bound to rung 'geo').
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy_common as DC  # noqa: E402

sys.path.insert(0, str(DC.GEO2_DIR))
import eval_geo2 as EG  # noqa: E402
import eval_d0d1 as ED  # noqa: E402

sys.path.insert(0, str(DC.LAD_DIR))
import evaluate as LE  # noqa: E402

TARGETS, ARMS, MODELS = DC.TARGETS, DC.ARMS, DC.MODELS
RUNGS3 = (DC.SRC_RUNG, DC.RUNG_DEPLOY, DC.IN_RUNG)              # (a), (b), (c)
CASE_OF = {v: k for k, v in DC.CASES.items()}
SELS_ALL = list(EG.SELS) + [(m, a) for a in ARMS for m in MODELS]
POS = ("all", "p", "m", "o", "direct")
HEAD_SELS = [EG.SEL_NAME[s] for s in EG.SELS]                   # REP8, REP·SH, REP·S, ENS4·SH, ENS4·S
OUT_FILES = dict(metrics="DEPLOY_metrics.csv", compare="DEPLOY_compare.csv", goal="DEPLOY_goal.csv",
                 metrics_ladder="DEPLOY_metrics_ladder.csv", bends="DEPLOY_bends.csv",
                 bend_rates="DEPLOY_bend_rates.json", report="DEPLOY_report.json", selftest="DEPLOY_selftest.json")
E2_TOL, LINE_TOL, CROSS_TOL = 1e-12, 1e-8, 1e-9                 # = eval_d0d1 selftest tolerances; cross = its 1e-9
if LE.N_BOOT != EG.N_BOOT or LE.SEED != EG.SEED or tuple(LE.GOAL_TARGETS) != tuple(EG.MAIN):
    DC.stop("bootstrap / goal constants of evaluate.py and eval_geo2.py differ")


def sel_name(sel):
    return EG.SEL_NAME.get(tuple(sel)) or f"{sel[0]}·{sel[1]}"


def sel_arm(sel):
    return "S+SH" if sel[0] == "REP8" else sel[1]


def sel_source(sel, rung, t, rep, rep8):
    if sel[0] == "REP8":
        a, m = rep8[(rung, t)]
        return f"{m}·{a}"
    if sel[0] == "REP":
        return rep[(rung, sel[1], t)]
    return sel[0]


def mi_key(sel, rung, t, rep8):
    """the LADDER metrics key (rung, arm, model) of a selection (REP8 -> its chosen arm and model)."""
    if sel[0] == "REP8":
        a, m = rep8[(rung, t)]
        return rung, a, m
    if sel[0] in ("REP", "ENS4"):
        return rung, sel[1], sel[0]
    return rung, sel[1], sel[0]


# ---------------------------------------------------------------- selections (DP-5), before any test file is read
def selections():
    """REP / REP8: (a) and (b) from the dft unit validation records (L_report.json + L_xgb_report.json units, = LADDER
    E2), (c) from the geo unit records (L_report_geo.json + L_xgb_report_geo.json, = GEO2b E). STOP when the dft REP is
    not the LADDER E2 choice."""
    Ul = EG.load_units(EG.BASE_UNITS)
    Ud = Ul[Ul.rung == DC.SRC_RUNG]
    Ug = ED.load_units_geo(EG.GEO_UNITS)
    U = pd.concat([Ud, Ud.assign(rung=DC.RUNG_DEPLOY), Ug], ignore_index=True)
    rep, rep8, mval = EG.reps(U)
    e2 = DC.read_json(LE.L.RES / "E2_report.json")["representatives"][DC.SRC_RUNG]
    bad = [f"{a}|{t}: {rep[(DC.SRC_RUNG, a, t)]} vs E2 {e2[a][t]['model']}" for a in ARMS for t in TARGETS
           if rep[(DC.SRC_RUNG, a, t)] != e2[a][t]["model"]]
    if bad:
        DC.stop(f"dft REP differs from LADDER E2_report.json representatives: {bad[:5]}")
    same = all(rep[(DC.RUNG_DEPLOY, a, t)] == rep[(DC.SRC_RUNG, a, t)] for a in ARMS for t in TARGETS) and \
        all(tuple(rep8[(DC.RUNG_DEPLOY, t)]) == tuple(rep8[(DC.SRC_RUNG, t)]) for t in TARGETS)
    if not same:
        DC.stop("REP / REP8 of dft_on_geo differ from dft")
    info = {CASE_OF[r]: dict(rung=r, rep={f"{a}|{t}": rep[(r, a, t)] for a in ARMS for t in TARGETS},
                             rep8={t: list(rep8[(r, t)]) for t in TARGETS},
                             mean_val={f"{a}|{m}|{t}": float(mval[(r, a, m, t)]) for a in ARMS for m in MODELS
                                       for t in TARGETS})
            for r in RUNGS3}
    info["rule"] = ("REP: per rung x arm x target the lowest 5-fold mean val_kcal_mae (ties AW, GNN, TF, XGB); REP8: over "
                    "both arms (ties SH first, then that order); (b) uses the dft choices (dft unit validation), (c) the "
                    "geo unit records; no re-selection on test pairs")
    info["dft_rep_equals_ladder_e2"] = True
    return rep, rep8, info


# ---------------------------------------------------------------- inputs
def context():
    EV, finfo = LE.fu2_eval()
    lab = LE.label_info(LE.DEFAULT_LABELS)
    S = LE.prep_S(pd.read_csv(LE.S_FOLDS, dtype={"pair_id": str}))
    Y = LE.read_lab(lab["path"])
    F = EG.pairs_frame()
    return dict(EV=EV, fu2_evaluate=finfo, labels=lab, S=S, Y=Y, F=F)


def norm_pred(P):
    P = P[DC.PRED_COLS].copy()
    P["pair_id"] = P.pair_id.astype(str)
    P["fold"] = P.fold.astype(int)
    P["ref_kind"] = P.ref_kind.fillna("").astype(str)
    P["pos"] = P.pos.fillna("").astype(str)
    return P.reset_index(drop=True)


def fold_check(P, S, name):
    """a D1 test row's fold = its S_folds.csv fold_T1 (T1 test = fold_T1 == k); else STOP."""
    f = S.fold_T1.reindex(P.pair_id).to_numpy()
    bad = ~(np.isfinite(f.astype(float)) & (f.astype(float) == P.fold.to_numpy(float)))
    if bad.any():
        DC.stop(f"{name}: {int(bad.sum())} rows whose fold != S_folds.csv fold_T1 (e.g. {P.pair_id[bad].head(3).tolist()})")


def load_cases(ctx, read_b=True):
    """(a) LADDER dft rows, (b) the deploy rows (read_b), (c) the GEO2b geo D1 rows; sha256 = their reports."""
    lr, gr = DC.read_json(DC.L_REPORT), DC.read_json(DC.GEO_REPORT)
    info = dict(a=dict(path=str(DC.L_PRED), sha256=DC.sha256(DC.L_PRED)),
                c=dict(path=str(DC.GEO_PRED), sha256=DC.sha256(DC.GEO_PRED)))
    if info["a"]["sha256"] != lr.get("predictions_sha256"):
        DC.stop(f"{DC.L_PRED} changed after {DC.L_REPORT.name}")
    if info["c"]["sha256"] != gr.get("predictions_sha256"):
        DC.stop(f"{DC.GEO_PRED} changed after {DC.GEO_REPORT.name}")
    La = pd.read_parquet(DC.L_PRED, columns=DC.PRED_COLS)
    Pa = norm_pred(La[La.rung == DC.SRC_RUNG])
    del La
    Gc = pd.read_parquet(DC.GEO_PRED, columns=DC.PRED_COLS)
    Pc = norm_pred(Gc[Gc.ref_kind.fillna("").astype(str) != "d0"])
    del Gc
    if set(Pc.rung) != {DC.IN_RUNG}:
        DC.stop(f"{DC.GEO_PRED.name} D1 rows hold rungs {sorted(set(Pc.rung))}")
    Pb = None
    if read_b:
        info["b"] = dict(path=str(DC.DEPLOY_PRED), sha256=DC.sha256(DC.DEPLOY_PRED))
        Pb = norm_pred(pd.read_parquet(DC.DEPLOY_PRED, columns=DC.PRED_COLS))
        if set(Pb.rung) != {DC.RUNG_DEPLOY}:
            DC.stop(f"{DC.DEPLOY_PRED.name} holds rungs {sorted(set(Pb.rung))}")
    for nm, P in (("a", Pa), ("b", Pb), ("c", Pc)):
        if P is not None:
            fold_check(P, ctx["S"], nm)
    return Pa, Pb, Pc, info


def keyset(P):
    q = P[(P.arm == "S") & (P.model == "AW") & (P.target == TARGETS[0])]
    return set(zip(q.fold.astype(int), q.pair_id))


def common_set(Pa, Pb, Pc, csv=True):
    """DP-4: common = (fold, pair) predicted in (a), (b) and (c); (b) must be exactly that set (and = common_pairs.csv)."""
    A, B, Cc = keyset(Pa), keyset(Pb), keyset(Pc)
    com = A & B & Cc
    rec = dict(n_a=len(A), n_b=len(B), n_c=len(Cc), n_common=len(com), a_not_in_b=len(A - B), c_not_in_b=len(Cc - B),
               b_not_in_a=len(B - A), b_not_in_c=len(B - Cc))
    if B != com:
        DC.stop(f"deployed pairs outside the dft or geo test lists: {rec}")
    if csv:
        CP = pd.read_csv(DC.COMMON_CSV, dtype={"pair_id": str})
        if set(zip(CP.fold.astype(int), CP.pair_id)) != B:
            DC.stop(f"deployed pairs != {DC.COMMON_CSV} (compat_deploy.py)")
        rec["common_csv_sha256"] = DC.sha256(DC.COMMON_CSV)
    return com, rec


# ---------------------------------------------------------------- the LADDER E1 / E2 computation on one rung
def line_from_sh(fr, rung):
    """= eval_d0d1.line_from_sh (E-3) for any rung: LINE = the AW·SH base column (check_line then requires every SH
    model to carry the same base / beta and the S rows none)."""
    f = fr[(rung, "SH", "AW")]
    line = f.assign(yhat=f.base.to_numpy(float), source="LINE")[["pair_id", "fold", "target", "y", "yhat", "base",
                                                                  "beta", "source"]]
    betas = {int(k): {t: float(g[g.target == t].beta.iloc[0]) for t in TARGETS if (g.target == t).any()}
             for k, g in line.groupby("fold")}
    return line.reset_index(drop=True), betas


def rung_part(P1, rung, rep, rep8, ctx):
    """= eval_d0d1.d1_part (297-351) with the rung as an argument: pair gates, LINE / ENS4 / REP / null frames, metrics_e1,
    bends_e1, goal_e1 (+ REP8 goal rows and bend_passed)."""
    t0 = time.time()
    EV, S, Y = ctx["EV"], ctx["S"], ctx["Y"]
    LE.check_pair_attrs(P1, S)
    rungs = (rung,)
    fr = LE.model_frames(P1, rungs)
    base = fr[(rung, "S", "AW")]
    base = base[base.target == TARGETS[0]]
    tests = {int(k): g.pair_id.astype(str).tolist() for k, g in base.groupby("fold")}
    ref = {rung: LE.ref_set(tests, rung)}
    gm = EV.pair_sets_gate({f"{rung}|{a}|{m}": fr[(rung, a, m)] for a in ARMS for m in MODELS}, ref[rung], Y)
    lines = {rung: line_from_sh(fr, rung)}
    fr = LE.derived_frames(fr, lines, rep, rungs)
    ga = LE.gate_pairs(EV, fr, ref, Y, {}, rungs)
    lchk = LE.check_line(fr, lines, rungs)
    A = LE.annotate_all(EV, fr, S, None, rungs)
    Mt = LE.metrics_e1(EV, A)
    MI = LE.met_index(Mt)
    Bd, rates, fams = LE.bends_e1(EV, A, S, rep)
    goal, verdict, concl = LE.goal_e1(MI, rates, rungs)
    g8, verdict8 = ED.goal_rep8(MI, rates, rep8, rungs)
    goal = pd.concat([ED.add_bend_passed(goal), g8], ignore_index=True)
    return dict(Mt=Mt, MI=MI, Bd=Bd, rates=rates, fams=fams, goal=goal, verdict=verdict, verdict8=verdict8, concl=concl,
                gates=dict(models=gm, all=ga, n_test_pairs=len(ref[rung])), line_check=lchk, line_betas=lines[rung][1],
                seconds=time.time() - t0)


# ---------------------------------------------------------------- tables on the common P_main reference pairs
def p_main(F, ids):
    s = set(F[(F.ref_kind == "reference") & F.in_P_main.astype(bool)].pair_id.astype(str))
    return sorted(set(ids) & s)


def metrics_sel(Pall, F, rep, rep8, rungs, pset):
    """DP-6: case x selection x target x position: n, MAE, RMSE, core bootstrap CI (eval_d0d1.boot_mae_ci = eval_geo2
    boot_diff(e, 0, core)), null MAE."""
    Fi = F.set_index("pair_id")
    pos = Fi.pos.reindex(pset).fillna("").astype(str).to_numpy()
    core = Fi.core.reindex(pset).astype(str).to_numpy()
    rows = []
    for rung in rungs:
        for sel in SELS_ALL:
            for t in TARGETS:
                d = EG.select(Pall, rung, sel, t, rep, rep8).reindex(pset)
                if d.yhat.isna().any() or d.y.isna().any():
                    DC.stop(f"{rung} {sel_name(sel)} {t}: predictions missing on the common P_main pairs")
                e, y = (d.yhat - d.y).to_numpy(float), d.y.to_numpy(float)
                for p in POS:
                    m = np.ones(len(e), bool) if p == "all" else pos == p
                    if not m.any():
                        continue
                    mae, lo, hi = ED.boot_mae_ci(e[m], core[m])
                    rows.append(dict(case=CASE_OF[rung], rung=rung, sel=sel_name(sel), model=sel[0], arm=sel_arm(sel),
                                     source=sel_source(sel, rung, t, rep, rep8), target=t, pos=p, n=int(m.sum()),
                                     n_cores=int(len(set(core[m]))), mae=mae, rmse=float(np.sqrt(np.mean(e[m] ** 2))),
                                     mae_ci_lo=lo, mae_ci_hi=hi, mae_null=float(np.abs(y[m]).mean())))
    return pd.DataFrame(rows)


def compare_sel(Pall, F, rep, rep8, x, b, pset, name):
    """DP-6 paired difference d = MAE(x) - MAE(b) on the same pairs (eval_geo2.boot_diff, core resampling)."""
    Fi = F.set_index("pair_id")
    pos = Fi.pos.reindex(pset).fillna("").astype(str).to_numpy()
    core = Fi.core.reindex(pset).astype(str).to_numpy()
    rows = []
    for sel in SELS_ALL:
        for t in TARGETS:
            dx = EG.select(Pall, x, sel, t, rep, rep8).reindex(pset)
            db = EG.select(Pall, b, sel, t, rep, rep8).reindex(pset)
            if dx.yhat.isna().any() or db.yhat.isna().any():
                DC.stop(f"compare {name} {sel_name(sel)} {t}: predictions missing")
            if DC.maxdiff(dx.y, db.y) > DC.Y_TOL:
                DC.stop(f"compare {name} {sel_name(sel)} {t}: y differs between {x} and {b}")
            ex, eb = (dx.yhat - dx.y).to_numpy(float), (db.yhat - db.y).to_numpy(float)
            for p in POS:
                m = np.ones(len(ex), bool) if p == "all" else pos == p
                if not m.any():
                    continue
                d, lo, hi, pv = EG.boot_diff(ex[m], eb[m], core[m])
                rows.append(dict(comparison=name, x=x, b=b, case_x=CASE_OF[x], case_b=CASE_OF[b], sel=sel_name(sel),
                                 model=sel[0], arm=sel_arm(sel), source_x=sel_source(sel, x, t, rep, rep8),
                                 source_b=sel_source(sel, b, t, rep, rep8), target=t, pos=p, n=int(m.sum()),
                                 n_cores=int(len(set(core[m]))), mae_x=float(np.abs(ex[m]).mean()),
                                 mae_b=float(np.abs(eb[m]).mean()), d=d, ci_lo=lo, ci_hi=hi, p=pv))
    return pd.DataFrame(rows)


def cross_check(M, parts, rep8):
    """my P_main x p rows (eval_geo2 select) = the LADDER metrics_e1 rows of the same rung / selection (n exact; MAE and
    CI <= 1e-9; = the eval_d0d1 'e_metrics_vs_e3' check, extended to every selection)."""
    Mi = M[M.pos == "p"].set_index(["rung", "sel", "target"])
    bad, n = [], 0
    for rung, part in parts.items():
        for sel in SELS_ALL:
            for t in TARGETS:
                r, a, m = mi_key(sel, rung, t, rep8)
                x = part["MI"][(r, a, m, "own", t, "P_main", "p")]
                y = Mi.loc[(rung, sel_name(sel), t)]
                n += 1
                if int(x["n"]) != int(y["n"]) or any(abs(float(x[c]) - float(y[c])) > CROSS_TOL
                                                     for c in ("mae", "mae_ci_lo", "mae_ci_hi")):
                    bad.append(f"{rung}|{sel_name(sel)}|{t}")
    return dict(ok=not bad, n_compared=n, problems=bad[:20])


def evaluate_cases(Pa, Pb, Pc, rep, rep8, ctx, common):
    ids = sorted({p for _, p in common})
    sub = {DC.SRC_RUNG: Pa, DC.RUNG_DEPLOY: Pb, DC.IN_RUNG: Pc}
    sub = {r: P[P.pair_id.isin(ids)].reset_index(drop=True) for r, P in sub.items()}
    for r, P in sub.items():
        if keyset(P) != set(common):
            DC.stop(f"rung {r}: (fold, pair) list on the common set != common")
    parts = {}
    for r in RUNGS3:
        parts[r] = rung_part(sub[r], r, rep, rep8, ctx)
        print(f"[deploy-eval] rung {r}: LADDER pipeline done ({parts[r]['seconds']:.0f} s)", flush=True)
    Pall = pd.concat([sub[r] for r in RUNGS3], ignore_index=True)
    pset = p_main(ctx["F"], ids)
    M = metrics_sel(Pall, ctx["F"], rep, rep8, RUNGS3, pset)
    Cmp = pd.concat([compare_sel(Pall, ctx["F"], rep, rep8, DC.RUNG_DEPLOY, DC.SRC_RUNG, pset, "b-a"),
                     compare_sel(Pall, ctx["F"], rep, rep8, DC.RUNG_DEPLOY, DC.IN_RUNG, pset, "b-c")],
                    ignore_index=True)
    cross = cross_check(M, parts, rep8)
    if not cross["ok"]:
        DC.stop(f"selection tables disagree with the LADDER metrics_e1 rows: {cross['problems'][:5]}")
    return parts, M, Cmp, cross, pset


# ---------------------------------------------------------------- summaries
def goal_summary(part, rung):
    v, v8 = part["verdict"], part["verdict8"][f"{rung}|REP8"]
    G = part["goal"]
    G = G[(G.level == "target") & G.target.isin(list(LE.GOAL_TARGETS))]
    det = {}
    for r in G.itertuples(index=False):
        det[f"{r.model}·{r.arm}|{r.target}" if r.model != "REP8" else f"REP8|{r.target}"] = dict(
            source=r.source, para_mae=r.para_mae, para_mae_ci=[r.para_mae_ci_lo, r.para_mae_ci_hi], n_para=r.n_para,
            sens=r.sens, n_bent_dft=r.n_bent_dft, fpr=r.fpr, n_straight_dft=r.n_straight_dft, crit_mae=r.crit_mae,
            crit_sens=r.crit_sens, crit_fpr=r.crit_fpr, passed=r.passed, surely_passed=r.surely_passed)
    out = {"REP8": dict(role="primary with REP (GEO2.md 9-1)", passed=v8["passed"], surely_passed=v8["surely_passed"],
                        mae_part=v8["mae_passed"], bend_part=v8["bend_passed"], choices=v8["choices"])}
    for a in ARMS:
        for m in ("REP", "ENS4") + MODELS:
            x = v[f"{rung}|{a}|{m}"]
            out[f"{m}·{a}"] = dict(role=x["role"], passed=x["passed"], surely_passed=x["surely_passed"])
    return dict(by_selection=out, detail=det,
                rule="LADDER 13-2: oi and cpcm both para MAE <= 0.13, bend sensitivity >= 0.7, FPR <= 0.1 (point "
                     "estimates on the common pairs; surely = para MAE CI upper <= 0.13); REP primary, ENS4 secondary, "
                     "models descriptive")


def headline(M, Cmp):
    out = {}
    h = M[(M.pos == "p") & M.sel.isin(HEAD_SELS)]
    for r in h.itertuples(index=False):
        out.setdefault(r.case, {}).setdefault(r.sel, {})[r.target] = dict(mae=r.mae, ci=[r.mae_ci_lo, r.mae_ci_hi],
                                                                         n=r.n, source=r.source)
    c = Cmp[(Cmp.pos == "p") & Cmp.sel.isin(HEAD_SELS)]
    for r in c.itertuples(index=False):
        out.setdefault(r.comparison, {}).setdefault(r.sel, {})[r.target] = dict(d=r.d, ci=[r.ci_lo, r.ci_hi], p=r.p,
                                                                               mae_x=r.mae_x, mae_b=r.mae_b)
    return out


def ladder_full_reference():
    """(a) on its own full pair set as LADDER reported it (E2_metrics.csv, dft own, P_main para; REP and ENS4)."""
    EM = pd.read_csv(LE.L.RES / "E2_metrics.csv", keep_default_na=False, na_values=[""], float_precision="round_trip")
    EM = EM[(EM.rung == DC.SRC_RUNG) & (EM.pair_set == "own") & (EM.set == "P_main") & (EM.pos == "p")
            & EM.model.isin(["REP", "ENS4"])]
    return {f"{m}·{a}": {r.target: dict(mae=r.mae, ci=[r.mae_ci_lo, r.mae_ci_hi], n=r.n)
                         for r in EM[(EM.model == m) & (EM.arm == a)].itertuples(index=False)}
            for m in ("REP", "ENS4") for a in ARMS}


def write_outputs(parts, M, Cmp, outputs):
    def csv(name, df):
        p = DC.write_csv(DC.RES / name, df)
        outputs[str(p)] = DC.sha256(p)
    csv(OUT_FILES["metrics"], M)
    csv(OUT_FILES["compare"], Cmp)
    csv(OUT_FILES["goal"], pd.concat([parts[r]["goal"].assign(case=CASE_OF[r]) for r in RUNGS3], ignore_index=True))
    csv(OUT_FILES["metrics_ladder"], pd.concat([parts[r]["Mt"].assign(case=CASE_OF[r]) for r in RUNGS3],
                                               ignore_index=True))
    csv(OUT_FILES["bends"], pd.concat([parts[r]["Bd"].assign(case=CASE_OF[r]) for r in RUNGS3], ignore_index=True))
    rates = {k: v for r in RUNGS3 for k, v in parts[r]["rates"].items()}
    p = DC.write_json(DC.RES / OUT_FILES["bend_rates"],
                      dict(key="rung|arm|model|pair_set (own = the common pairs of that rung)", rates=rates,
                           families={r: parts[r]["fams"] for r in RUNGS3},
                           note="d1_eval.bend_rates on FU2 bends_tab rows (= E2_bend_rates.json format)"))
    outputs[str(p)] = DC.sha256(p)


# ---------------------------------------------------------------- commands
def cmd_collect():
    """parts -> L_predictions_deploy.parquet (gates passed, compat ok; one pair list per fold for every unit = the common
    set; y = L_predictions dft y; S rows without a baseline, one SH baseline per pair / target)."""
    t0 = time.time()
    ra, rx, cj = (DC.read_json(p) if p.is_file() else DC.not_ready(f"{p} missing")
                  for p in (DC.REG_ATOMS, DC.REG_XGB, DC.COMPAT_JSON))
    for nm, r, part in (("atoms", ra, DC.PART_ATOMS), ("xgb", rx, DC.PART_XGB)):
        g, d = r.get("gate") or {}, r.get("deploy") or {}
        if not g.get("passed"):
            DC.stop(f"{nm} regression gate not passed ({g.get('status')})")
        if d.get("part") != str(part) or not part.is_file() or DC.sha256(part) != d.get("sha256"):
            DC.stop(f"{part} missing or changed after its regression report")
    if not cj.get("ok") or not DC.COMMON_CSV.is_file() or \
            DC.sha256(DC.COMMON_CSV) != (cj.get("pairs") or {}).get("common_csv_sha256"):
        DC.stop(f"{DC.COMPAT_JSON.name} not ok or {DC.COMMON_CSV} changed")
    P = norm_pred(pd.concat([pd.read_parquet(DC.PART_ATOMS), pd.read_parquet(DC.PART_XGB)], ignore_index=True))
    bad = []
    if set(P.rung) != {DC.RUNG_DEPLOY}:
        bad.append(f"rungs {sorted(set(P.rung))}")
    if set(zip(P.arm, P.model)) != {(a, m) for a in ARMS for m in MODELS} or set(P.target) != set(TARGETS):
        bad.append("arm x model / targets incomplete")
    if P.duplicated(["pair_id", "arm", "model", "target"]).any():
        bad.append("duplicated rows")
    if not np.isfinite(P.yhat.to_numpy(float)).all():
        bad.append("non-finite yhat")
    CP = pd.read_csv(DC.COMMON_CSV, dtype={"pair_id": str})
    com = set(zip(CP.fold.astype(int), CP.pair_id))
    for (a, m, t), g in P.groupby(["arm", "model", "target"]):
        if set(zip(g.fold, g.pair_id)) != com:
            bad.append(f"{a}|{m}|{t}: (fold, pair) list != common_pairs.csv")
            break
    Lp, _ = DC.load_ladder_dft(MODELS)
    Lq = Lp[(Lp.arm == "S") & (Lp.model == "AW")].set_index(["pair_id", "target"])
    J = P.set_index(["pair_id", "target"]).join(Lq[["y", "fold"]], rsuffix="_lad", how="left")
    dy = DC.maxdiff(J.y.to_numpy(float), J.y_lad.to_numpy(float))
    if dy > DC.Y_TOL or (J.fold.to_numpy() != J.fold_lad.to_numpy()).any():
        bad.append(f"y / fold differ from the LADDER dft rows (max |dy| {dy})")
    s = P[P.arm == "S"]
    if ((s.base != 0) | (s.beta != 0)).any():
        bad.append("S rows with a baseline")
    sh = P[P.arm == "SH"].groupby(["pair_id", "target"]).agg(b0=("base", "min"), b1=("base", "max"), e0=("beta", "min"),
                                                           e1=("beta", "max"))
    db = float(np.nanmax(np.abs(sh.b1 - sh.b0).to_numpy(float))) if len(sh) else 0.0
    dbe = float(np.nanmax((np.abs(sh.e1 - sh.e0) / np.maximum(1.0, np.abs(sh.e0))).to_numpy(float))) if len(sh) else 0.0
    if db > LE.BASE_TOL or dbe > LE.BASE_TOL:
        bad.append(f"SH baselines differ between models (base {db:.3g}, beta {dbe:.3g}; LADDER E1-5 tolerance)")
    if bad:
        DC.stop("deploy parts: " + "; ".join(bad))
    DC.write_parquet(DC.DEPLOY_PRED, P)
    rep = dict(phase="GEO2b DEPLOY collect", predictions=str(DC.DEPLOY_PRED), sha256=DC.sha256(DC.DEPLOY_PRED),
               n_rows=int(len(P)), n_pairs=int(P.pair_id.nunique()),
               n_pairs_by_fold={int(k): int(g.pair_id.nunique()) for k, g in P.groupby("fold")},
               parts={str(DC.PART_ATOMS): DC.sha256(DC.PART_ATOMS), str(DC.PART_XGB): DC.sha256(DC.PART_XGB)},
               gates=dict(atoms={k: ra["gate"].get(k) for k in ("status", "passed", "max_abs_yhat_vs_l_predictions",
                                                                "batch_bs32_vs_bs256_seed0_max", "gpu", "unit_gpus")},
                          xgb={k: rx["gate"].get(k) for k in ("status", "passed", "max_abs_yhat")}),
               checks=dict(y_vs_ladder_max_abs=dy, sh_base_max_abs_spread=db, sh_beta_max_rel_spread=dbe),
               common_csv_sha256=DC.sha256(DC.COMMON_CSV), env=DC.env_info(), code_sha256=DC.code_sha(),
               seconds=time.time() - t0)
    DC.write_json(DC.COLLECT_JSON, rep)
    print(f"[deploy-collect] {len(P)} rows, {rep['n_pairs']} pairs -> {DC.DEPLOY_PRED}", flush=True)
    return 0


def cmd_run():
    t0 = time.time()
    if not DC.COLLECT_JSON.is_file() or not DC.DEPLOY_PRED.is_file():
        DC.not_ready(f"{DC.COLLECT_JSON} / {DC.DEPLOY_PRED} missing (eval_deploy.py collect)")
    col = DC.read_json(DC.COLLECT_JSON)
    if DC.sha256(DC.DEPLOY_PRED) != col.get("sha256"):
        DC.stop(f"{DC.DEPLOY_PRED} changed after {DC.COLLECT_JSON.name}")
    rep, rep8, sel_info = selections()                  # DP-5: unit records only, before any prediction file
    ctx = context()
    Pa, Pb, Pc, pinfo = load_cases(ctx, read_b=True)
    com, crec = common_set(Pa, Pb, Pc, csv=True)
    parts, M, Cmp, cross, pset = evaluate_cases(Pa, Pb, Pc, rep, rep8, ctx, com)
    outputs = {}
    write_outputs(parts, M, Cmp, outputs)
    Fi = ctx["F"].set_index("pair_id")
    report = dict(
        phase="GEO2b DEPLOY evaluation (D1 only; DECISIONS_deploy.md DP-4 .. DP-6)", verdict="none (descriptive); the "
        "only rule applied is the LADDER 13-2 goal check, reported for (b) (and (a), (c) on the same pairs)",
        cases={k: dict(rung=v) for k, v in DC.CASES.items()}, date=time.strftime("%F %T"),
        n=dict(crec, common_p_main=len(pset), common_p_main_para=int(Fi.pos.reindex(pset).eq("p").sum())),
        selections=sel_info, goal_b=goal_summary(parts[DC.RUNG_DEPLOY], DC.RUNG_DEPLOY),
        goal_a_common=goal_summary(parts[DC.SRC_RUNG], DC.SRC_RUNG), goal_c_common=goal_summary(parts[DC.IN_RUNG],
                                                                                                DC.IN_RUNG),
        headline_para=headline(M, Cmp), ladder_full_set_para_e2=ladder_full_reference(), cross_check=cross,
        gates={r: parts[r]["gates"] for r in RUNGS3}, line_check={r: parts[r]["line_check"] for r in RUNGS3},
        line_betas={r: parts[r]["line_betas"] for r in RUNGS3}, conclusion_13_3_a=parts[DC.SRC_RUNG]["concl"],
        inputs=dict(predictions=pinfo, collect=dict(path=str(DC.COLLECT_JSON), sha256=DC.sha256(DC.COLLECT_JSON)),
                    units=[str(p) for p in EG.BASE_UNITS + EG.GEO_UNITS],
                    decisions=dict(path=str(DC.DECISIONS_MD), sha256=DC.sha256(DC.DECISIONS_MD))),
        fu2_evaluate=ctx["fu2_evaluate"], labels=ctx["labels"], outputs=outputs, env=DC.env_info(),
        code_sha256=DC.code_sha(), seconds=time.time() - t0)
    p = DC.write_json(DC.RES / OUT_FILES["report"], report)
    g = report["goal_b"]["by_selection"]
    print(json.dumps(dict(n=report["n"], goal_b={k: g[k]["passed"] for k in ("REP8", "REP·S", "REP·SH", "ENS4·S",
                                                                              "ENS4·SH")}), indent=1, default=str),
          flush=True)
    print(f"[deploy-eval] -> {p} ({time.time() - t0:.0f} s)", flush=True)
    return 0


def _rows_equal(A, B, keys, cols, tol, line_tol=None):
    """count of rows not in both + per column count of values that differ (eval_d0d1._num_close, relative tol; LINE rows
    absolute line_tol when given)."""
    J = A.merge(B, on=keys, how="outer", suffixes=("", "_ref"), indicator=True)
    out = {"rows_not_in_both": int((J["_merge"] != "both").sum())}
    Jb = J[J["_merge"] == "both"]
    isl = (Jb.model == "LINE").to_numpy() if ("model" in Jb.columns and line_tol is not None) else np.zeros(len(Jb), bool)
    for c in cols:
        out[c] = int(sum(not (ED._num_close(x, y, line_tol, rel=False) if li else ED._num_close(x, y, tol))
                         for x, y, li in zip(Jb[c].tolist(), Jb[f"{c}_ref"].tolist(), isl)))
    return out


def cmd_selftest():
    t0 = time.time()
    ck, det = {}, {}
    rep, rep8, sel_info = selections()
    ctx = context()
    Pa, _, Pc, _ = load_cases(ctx, read_b=False)
    # (i) the pipeline on the full LADDER dft predictions = LADDER E2 (dft, own)
    full = rung_part(Pa, DC.SRC_RUNG, rep, rep8, ctx)
    EM = pd.read_csv(LE.L.RES / "E2_metrics.csv", keep_default_na=False, na_values=[""], float_precision="round_trip")
    EM = EM[(EM.rung == DC.SRC_RUNG) & (EM.pair_set == "own")]
    num = ["n", "mae", "rmse", "mae_null", "mae_rel_null", "n_cores", "mae_ci_lo", "mae_ci_hi", "n_fam", "r2_fam_median",
           "r2_fam_q1", "r2_fam_q3", "rho_err_median", "source"]
    k = ["arm", "model", "target", "set", "pos"]
    nb = _rows_equal(full["Mt"][k + num], EM[k + num], k, num, E2_TOL, LINE_TOL)
    ck["i_metrics_equal_E2_dft"], det["i_metrics"] = all(v == 0 for v in nb.values()), nb
    G2 = pd.read_csv(LE.L.RES / "E2_goal.csv", keep_default_na=False, na_values=[""], float_precision="round_trip")
    G2 = G2[(G2.rung == DC.SRC_RUNG) & G2.level.isin(["target", "rung_arm"])]
    G3 = full["goal"][full["goal"].level.isin(["target", "rung_arm"]) & (full["goal"].model != "REP8")]
    gc = ["passed", "surely_passed", "para_mae", "para_mae_ci_hi", "sens", "fpr"]
    gk = ["level", "arm", "model", "target"]
    ng = _rows_equal(G3[gk + gc], G2[gk + gc], gk, gc, E2_TOL)
    ck["i_goal_equal_E2_dft"], det["i_goal"] = all(v == 0 for v in ng.values()) and len(G3) > 0, ng
    BR = DC.read_json(LE.L.RES / "E2_bend_rates.json")["rates"]
    rb, rl = [], []
    for key, v in full["rates"].items():
        if not ED._same(ED._clean(v), BR.get(key)):
            (rl if "|LINE|" in key else rb).append(key)
    ck["i_bend_rates_equal_E2_dft"], det["i_bend_rates"], det["i_bend_rates_LINE_differ"] = not rb, rb[:10], rl
    E2v = DC.read_json(LE.L.RES / "E2_report.json")
    det["i_e2_verdict_dft"] = {k2: v for k2, v in (E2v.get("verdict") or {}).items() if str(k2).startswith("dft|")}
    # (ii) b := a (relabelled) on A & C: every difference 0, identical tables
    A, Cc = keyset(Pa), keyset(Pc)
    com = A & Cc
    ids = {p for _, p in com}
    Pb = Pa[Pa.pair_id.isin(ids)].assign(rung=DC.RUNG_DEPLOY).reset_index(drop=True)
    parts, M, Cmp, cross, pset = evaluate_cases(Pa, Pb, Pc, rep, rep8, ctx, com)
    z = Cmp[Cmp.comparison == "b-a"]
    ck["ii_b_minus_a_zero"] = bool(len(z) and (z.d == 0).all() and (z.ci_lo == 0).all() and (z.ci_hi == 0).all())
    ma, mb = M[M.case == "a"].drop(columns=["case", "rung"]), M[M.case == "b"].drop(columns=["case", "rung"])
    ck["ii_metrics_b_equal_a"] = bool(len(ma) and ma.reset_index(drop=True).equals(mb.reset_index(drop=True)))
    ta = parts[DC.SRC_RUNG]["Mt"].drop(columns=["rung", "pair_set_name"]).reset_index(drop=True)
    tb = parts[DC.RUNG_DEPLOY]["Mt"].drop(columns=["rung", "pair_set_name"]).reset_index(drop=True)
    ck["ii_ladder_metrics_b_equal_a"] = bool(ta.equals(tb))
    ga = parts[DC.SRC_RUNG]["goal"]
    gb = parts[DC.RUNG_DEPLOY]["goal"]
    ga = ga[ga.level.isin(["target", "rung_arm", "rung"])][gk + gc].reset_index(drop=True)
    gb = gb[gb.level.isin(["target", "rung_arm", "rung"])][gk + gc].reset_index(drop=True)
    ck["ii_goal_b_equal_a"] = bool(ga.equals(gb))
    ck["ii_cross_check"] = bool(cross["ok"])
    ck["ii_b_minus_c_rows"] = bool((Cmp.comparison == "b-c").sum() == len(z) > 0)
    det["ii"] = dict(n_common=len(com), n_p_main=len(pset), cross=cross, n_compare_rows=int(len(Cmp)))
    # (iii) cached bootstrap (eval_d0d1.boot_mae_ci) = eval_geo2.boot_diff(e, 0, core)
    Fi = ctx["F"].set_index("pair_id")
    bd = []
    Pall = pd.concat([Pa[Pa.pair_id.isin(ids)], Pb, Pc[Pc.pair_id.isin(ids)]], ignore_index=True)
    for sel, t in ((("REP8",), "oi"), (("ENS4", "SH"), "cpcm"), (("GNN", "S"), "barrier")):
        d = EG.select(Pall, DC.SRC_RUNG, sel, t, rep, rep8).reindex(pset)
        e = (d.yhat - d.y).to_numpy(float)
        cores = Fi.core.reindex(pset).astype(str).to_numpy()
        a_ = ED.boot_mae_ci(e, cores)
        b_ = EG.boot_diff(e, np.zeros(len(e)), cores)[:3]
        bd.append(max(abs(u - v) for u, v in zip(a_, b_)))
    ck["iii_boot_cached_equal"], det["iii_boot_max_diff"] = bool(max(bd) <= 1e-12), max(bd)
    rep_out = dict(phase="GEO2b DEPLOY evaluation selftest", checks=ck, detail=det, all_pass=all(ck.values()),
                   selections=sel_info, env=DC.env_info(), code_sha256=DC.code_sha(), seconds=time.time() - t0)
    DC.write_json(DC.RES / OUT_FILES["selftest"], rep_out)
    for k2, v in ck.items():
        print(f"[deploy-selftest] {k2}: {'ok' if v else 'FAIL'}", flush=True)
    print(f"[deploy-selftest] {'all checks passed' if rep_out['all_pass'] else 'FAILED'} ({time.time() - t0:.0f} s)",
          flush=True)
    return 0 if rep_out["all_pass"] else 3


if __name__ == "__main__":
    cmds = {"selftest": cmd_selftest, "collect": cmd_collect, "run": cmd_run}
    if len(sys.argv) == 2 and sys.argv[1] in cmds:
        sys.exit(ED._stop3(cmds[sys.argv[1]]))
    raise SystemExit(__doc__)
