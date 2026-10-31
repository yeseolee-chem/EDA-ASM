#!/usr/bin/env python3
"""GEO2b Phase E: energy (8 EDA channels + c_ghost + barrier) of the geo rung against dft / g2 (docs/specs/GEO2.md 9,
GEO1B.md 5-2: no g1 rung, pair bins from GEO1b E_rxn.csv instead of the G1-derived same_ts_g1 class).

  python eval_geo2.py selftest   # g2 predictions relabelled as rung 'geo': must reproduce E2 numbers exactly, diff = 0
  python eval_geo2.py run        # disabled (exit 2) under the GEO2b decision D0D1: no comparison verdict; the
                                 # evaluation is eval_d0d1.py run, which reuses this module's functions
Reads only; any pandas + pyarrow + numpy + scipy env (LADDER evaluate env 'espley' or reactot).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
LAD_RES = REPO / "analysis" / "d1_ladder" / "results"
F_PAIRS = REPO / "analysis" / "d1_pair" / "results" / "F_pairs.csv"
GEO1B_RXN = REPO / "analysis" / "geo1b" / "results" / "E_rxn.csv"            # rxn_id -> bin (G2 vs label mapped RMSD)
BIN_ORDER = {"low": 0, "mid": 1, "high": 2}
BASE_PRED = LAD_RES / "L_predictions.parquet"
BASE_UNITS = (LAD_RES / "L_report.json", LAD_RES / "L_xgb_report.json")
GEO_PRED = LAD_RES / "L_predictions_geo.parquet"                       # stage-3 collect (spec 8-1)
GEO_UNITS = (LAD_RES / "L_report_geo.json", LAD_RES / "L_xgb_report_geo.json")
E2_REPORT, E2_METRICS = LAD_RES / "E2_report.json", LAD_RES / "E2_metrics.csv"
TARGETS = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "c_ghost", "barrier"]
MAIN = ("oi", "cpcm")
MODELS = ("AW", "GNN", "TF", "XGB")                 # LADDER 13-1 tie order
ARMS = ("SH", "S")                                  # REP8 tie order: SH first (spec 9-1)
N_BOOT, SEED = 2000, 20261019                       # = d1pair_common N_BOOT / SEED (LADDER spec 11)
GOAL_MAE = 0.13                                     # LADDER 13-2
APP_T = (0.41, 1.0, 1.36)                           # RT ln2, 1 kcal/mol, RT ln10 at 298 K (kcal/mol)


# ---------------------------------------------------------------- statistics (= d1_eval.boot_diff, copied verbatim)
def boot_diff(dx, db, cores, n_boot=N_BOOT, seed=SEED):
    """d = MAE(x) - MAE(b) with the cores resampled: (d, ci_lo, ci_hi, p)."""
    ex, eb = np.abs(dx), np.abs(db)
    uc, inv = np.unique(cores, return_inverse=True)
    sx, sb, cnt = np.bincount(inv, ex), np.bincount(inv, eb), np.bincount(inv)
    rng = np.random.default_rng(seed)
    k = len(uc)
    w = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)])
    num = w @ cnt
    dstar = (w @ sx) / num - (w @ sb) / num
    d = ex.mean() - eb.mean()
    lo, hi = np.percentile(dstar, [2.5, 97.5])
    p = min(1.0, 2 * min((1 + (dstar <= 0).sum()) / (n_boot + 1), (1 + (dstar >= 0).sum()) / (n_boot + 1)))
    return float(d), float(lo), float(hi), float(p)


# ---------------------------------------------------------------- inputs
def load_units(paths):
    """unit json records (L_report / L_xgb_report 'units') -> rows rung, arm, model, fold, target, val."""
    rows = []
    for p in paths:
        for name, u in json.loads(Path(p).read_text())["units"].items():
            _, rung, arm, model, f = name.split("__")
            for t, v in (u.get("val_kcal_mae") or {}).items():
                rows.append(dict(rung=rung, arm=arm, model=model, fold=int(f[1:]), target=t, val=float(v)))
    U = pd.DataFrame(rows)
    n = U.groupby(["rung", "arm", "model", "target"]).fold.nunique()
    if (n != 5).any():
        raise SystemExit(f"STOP: units without 5 folds: {n[n != 5].index.tolist()[:5]}")
    return U


def reps(U):
    """REP (LADDER 13-1: per rung x arm x target the lowest 5-fold mean val kcal MAE, ties AW -> GNN -> TF -> XGB) and
    REP8 (per rung x target over both arms; ties SH before S, then the model order)."""
    m = U.groupby(["rung", "arm", "model", "target"]).val.mean()
    rep, rep8 = {}, {}
    for (rung, arm, t), s in m.groupby(level=["rung", "arm", "target"]):
        v = {k[2]: float(x) for k, x in s.items()}
        rep[(rung, arm, t)] = min((v[mo], MODELS.index(mo), mo) for mo in v)[2]
    for (rung, t), s in m.groupby(level=["rung", "target"]):
        v = {(k[1], k[2]): float(x) for k, x in s.items()}
        rep8[(rung, t)] = min((x, ARMS.index(a), MODELS.index(mo), a, mo) for (a, mo), x in v.items())[3:]
    return rep, rep8, m


def pairs_frame():
    F = pd.read_csv(F_PAIRS)
    keep = ["pair_id", "ref_kind", "a_rxn", "b_rxn", "pos", "in_P_main", "core", "dipole", "skeleton", "series", "fam",
            "cfg", "orient", "face", "c3", "Xb"]
    F = F[[c for c in keep if c in F.columns]].copy()
    if GEO1B_RXN.is_file():
        B = pd.read_csv(GEO1B_RXN, usecols=["rxn_id", "bin"]).set_index("rxn_id").bin
        ba, bb = B.reindex(F.a_rxn).fillna("na").values, B.reindex(F.b_rxn).fillna("na").values
    else:
        ba = bb = np.array(["na"] * len(F))
    F["cls"] = ["na" if "na" in (x, y) else max((x, y), key=BIN_ORDER.get) for x, y in zip(ba, bb)]
    return F


def load_preds(selftest):
    L = pd.read_parquet(BASE_PRED)
    if selftest:
        G = L[L.rung == "g2"].copy()
        G["rung"] = "geo"
    else:
        G = pd.read_parquet(GEO_PRED)
        if set(G.rung) != {"geo"} or list(G.columns) != list(L.columns):
            raise SystemExit("STOP: L_predictions_geo.parquet must hold rung 'geo' only with the LADDER columns")
    return pd.concat([L, G], ignore_index=True)


def select(L, rung, sel, t, rep, rep8):
    """per-pair test prediction of one selection: ('REP', arm) | ('REP8',) | ('ENS4', arm) | (model, arm)."""
    d = L[(L.rung == rung) & (L.target == t)]
    if sel[0] == "REP8":
        arm, mo = rep8[(rung, t)]
        d = d[(d.arm == arm) & (d.model == mo)]
    elif sel[0] == "REP":
        d = d[(d.arm == sel[1]) & (d.model == rep[(rung, sel[1], t)])]
    elif sel[0] == "ENS4":
        d = d[d.arm == sel[1]].groupby("pair_id").agg(yhat=("yhat", "mean"), y=("y", "first"), n=("model", "nunique"))
        if (d.n != 4).any():
            raise SystemExit(f"STOP: ENS4 {rung} {sel} {t}: a pair without 4 model predictions")
        return d[["y", "yhat"]]
    else:
        d = d[(d.arm == sel[1]) & (d.model == sel[0])]
    if d.pair_id.duplicated().any():
        raise SystemExit(f"STOP: duplicated pair predictions {rung} {sel} {t}")
    return d.set_index("pair_id")[["y", "yhat"]]


SELS = [("REP8",), ("REP", "SH"), ("REP", "S"), ("ENS4", "SH"), ("ENS4", "S")]
SEL_NAME = {("REP8",): "REP8", ("REP", "SH"): "REP·SH", ("REP", "S"): "REP·S", ("ENS4", "SH"): "ENS4·SH",
            ("ENS4", "S"): "ENS4·S"}


def common_pairs(L, F, rungs):
    """P_main reference pairs predicted in every rung of 'rungs' (all targets, all models)."""
    ref = set(F[(F.ref_kind == "reference") & F.in_P_main.astype(bool)].pair_id)
    s = ref
    for r in rungs:
        s = s & set(L[L.rung == r].pair_id)
    return sorted(s)


# ---------------------------------------------------------------- tables
def metrics(L, F, rep, rep8, rungs, pset, name):
    Fi = F.set_index("pair_id")
    rows = []
    for rung in rungs:
        for sel in SELS:
            for t in TARGETS:
                d = select(L, rung, sel, t, rep, rep8).reindex(pset)
                if d.yhat.isna().any():
                    raise SystemExit(f"STOP: {rung} {sel} {t} lacks predictions on {name}")
                e = (d.yhat - d.y).values
                pos, core = Fi.pos.reindex(pset).values, Fi.core.reindex(pset).values
                for p in ("all", "p", "m", "o", "direct"):
                    m = np.ones(len(e), bool) if p == "all" else pos == p
                    if not m.any():
                        continue
                    r = dict(pair_set=name, rung=rung, sel=SEL_NAME[sel], target=t, pos=p, n=int(m.sum()),
                             mae=float(np.abs(e[m]).mean()))
                    if p == "p":
                        _, lo, hi, _ = boot_diff(e[m], np.zeros(m.sum()), core[m])
                        r.update(n_cores=int(len(set(core[m]))), mae_ci_lo=lo, mae_ci_hi=hi)
                    rows.append(r)
    return pd.DataFrame(rows)


def compare(L, F, rep, rep8, x, b, pset, name):
    """d = MAE(x) - MAE(b), para pairs of pset, core bootstrap (LADDER compare convention)."""
    Fi = F.set_index("pair_id")
    para = [p for p in pset if Fi.at[p, "pos"] == "p"]
    rows = []
    for sel in SELS:
        for t in TARGETS:
            dx = select(L, x, sel, t, rep, rep8).reindex(para)
            db = select(L, b, sel, t, rep, rep8).reindex(para)
            ex, eb = (dx.yhat - dx.y).values, (db.yhat - db.y).values
            d, lo, hi, p = boot_diff(ex, eb, Fi.core.reindex(para).values)
            rows.append(dict(pair_set=name, x=x, b=b, sel=SEL_NAME[sel], target=t, n=len(para), mae_x=float(np.abs(ex).mean()),
                             mae_b=float(np.abs(eb).mean()), d=d, ci_lo=lo, ci_hi=hi, p=p))
    return pd.DataFrame(rows)


def recovery(M):
    """(MAE_g2 - MAE_geo) / (MAE_g2 - MAE_dft) on P_023 para (REP8 and per-arm REP)."""
    m = M[(M.pair_set == "P_023") & (M.pos == "p")].pivot_table(index=["sel", "target"], columns="rung", values="mae")
    m["recovery"] = (m["g2"] - m["geo"]) / (m["g2"] - m["dft"])
    return m.reset_index()


def by_class(L, F, rep, rep8, pset):
    Fi = F.set_index("pair_id")
    para = [p for p in pset if Fi.at[p, "pos"] == "p"]
    rows = []
    for rung in ("dft", "g2", "geo"):
        if rung not in set(L.rung):
            continue
        for t in TARGETS:
            d = select(L, rung, ("REP8",), t, rep, rep8).reindex(para)
            e = np.abs(d.yhat - d.y).values
            for cls in ("low", "mid", "high", "na"):
                m = Fi.cls.reindex(para).values == cls
                rows.append(dict(rung=rung, sel="REP8", target=t, cls=cls, n=int(m.sum()), mae=float(e[m].mean())))
    return pd.DataFrame(rows)


def app_metrics(L, F, rep, rep8, pset):
    """barrier application metrics (post hoc in REPORT_D1LADDER 3; here fixed before the geo results): within-family
    order accuracy for true |gap| >= 0.41 / 1.0 / 1.36 kcal/mol, family Spearman median, regio change MAE / direction."""
    from scipy.stats import spearmanr
    Fi = F.set_index("pair_id")
    rows = []
    for rung in ("dft", "g2", "geo"):
        if rung not in set(L.rung):
            continue
        for sel in (("REP8",), ("ENS4", "SH"), ("ENS4", "S")):
            d = select(L, rung, sel, "barrier", rep, rep8).reindex(pset).join(Fi, how="left")
            acc = {t: [] for t in APP_T}
            rho = []
            for _, h in d.groupby("fam"):
                y, p = np.r_[h.y.values, 0.0], np.r_[h.yhat.values, 0.0]   # the family reference (H) = 0
                if len(y) >= 8:
                    rho.append(spearmanr(y, p)[0])
                i, j = np.triu_indices(len(y), 1)
                dy, dp = y[i] - y[j], p[i] - p[j]
                for t in APP_T:
                    k = np.abs(dy) >= t
                    acc[t].append(np.sign(dy[k]) == np.sign(dp[k]))
            key = ["dipole", "skeleton", "series", "cfg", "face", "c3", "Xb", "pos"]
            G = d.assign(o=d.orient.fillna("_"))
            rg = []
            for kk, h in G.groupby(key, dropna=False):
                if len(h) == 2 and h.o.nunique() == 2:
                    h = h.sort_values("o")
                    rg.append((kk[-1], h.y.iloc[0] - h.y.iloc[1], h.yhat.iloc[0] - h.yhat.iloc[1]))
            rg = pd.DataFrame(rg, columns=["pos", "dy", "dp"])
            e = (rg.dp - rg.dy).abs()
            big = rg.dy.abs() >= APP_T[0]
            rows.append(dict(rung=rung, sel=SEL_NAME[sel], n_pairs=len(d),
                             **{f"order_acc_{t}": float(np.concatenate(acc[t]).mean()) for t in APP_T},
                             spearman_fam_median=float(np.median(rho)) if rho else np.nan, regio_n=len(rg),
                             regio_mae=float(e.mean()) if len(rg) else np.nan,
                             regio_dir_acc_2x=float((np.sign(rg.dp[big]) == np.sign(rg.dy[big])).mean()) if big.any()
                             else np.nan))
    return pd.DataFrame(rows)


def verdict(C):
    """spec 10-1: REP8, P_23 para, geo vs g2: both oi and cpcm ci_hi < 0 -> improved; one -> partial; none -> not."""
    c = C[(C.pair_set == "P_23") & (C.sel == "REP8") & (C.target.isin(MAIN))].set_index("target")
    k = int((c.ci_hi < 0).sum())
    return dict(rule="REP8 para P_23: ci_hi(MAE_geo - MAE_g2) < 0 for oi and cpcm", n_main_improved=k,
                verdict={2: "개선", 1: "부분", 0: "불가"}[k],
                detail={t: {kk: float(c.at[t, kk]) for kk in ("mae_x", "mae_b", "d", "ci_lo", "ci_hi")} for t in MAIN})


def goal(M):
    """LADDER 13-2 MAE part (oi, cpcm para MAE <= 0.13) for geo; bend sensitivity / FPR come from evaluate.py e3 (spec 9-2)."""
    m = M[(M.pair_set == "own_geo") & (M.pos == "p") & (M.target.isin(MAIN)) & (M.rung == "geo")]
    return {s: {t: float(g[g.target == t].mae.iloc[0]) for t in MAIN} | {"mae_ok": bool((g.mae <= GOAL_MAE).all())}
            for s, g in m.groupby("sel")}


# ---------------------------------------------------------------- selftest checks against E2
def selftest_checks(M, C, rep):
    E2 = json.loads(E2_REPORT.read_text())["representatives"]
    bad = []
    for (rung, arm, t), mo in rep.items():
        if rung in E2 and E2[rung][arm][t]["model"] != mo:
            bad.append(f"REP {rung} {arm} {t}: {mo} != E2 {E2[rung][arm][t]['model']}")
    EM = pd.read_csv(E2_METRICS)
    EM = EM[(EM.pair_set == "own") & (EM.set == "P_main") & (EM.pos == "p")]
    own = M[(M.pair_set == "own_g2") & (M.pos == "p") & (M.rung == "g2")]
    n_cmp = 0
    for s_my, s_e2, arm in (("REP·SH", "REP", "SH"), ("REP·S", "REP", "S"), ("ENS4·SH", "ENS4", "SH"),
                            ("ENS4·S", "ENS4", "S")):
        for t in TARGETS:
            a = own[(own.sel == s_my) & (own.target == t)]
            b = EM[(EM.rung == "g2") & (EM.arm == arm) & (EM.model == s_e2) & (EM.target == t)]
            if len(a) != 1 or len(b) != 1:
                bad.append(f"missing row {s_my} {t}")
                continue
            n_cmp += 1
            if int(a.n.iloc[0]) != int(b.n.iloc[0]) or abs(a.mae.iloc[0] - b.mae.iloc[0]) > 1e-9 or \
                    abs(a.mae_ci_lo.iloc[0] - b.mae_ci_lo.iloc[0]) > 1e-9 or abs(a.mae_ci_hi.iloc[0] - b.mae_ci_hi.iloc[0]) > 1e-9:
                bad.append(f"E2 mismatch {s_my} {t}: {a[['n', 'mae', 'mae_ci_lo', 'mae_ci_hi']].values} vs "
                           f"{b[['n', 'mae', 'mae_ci_lo', 'mae_ci_hi']].values}")
    z = C[(C.pair_set == "P_23")]
    if (z.d.abs() > 0).any() or (z.ci_lo != 0).any() or (z.ci_hi != 0).any():
        bad.append("geo == g2 copy but some difference != 0")
    return dict(n_e2_rows_compared=n_cmp, problems=bad[:30], all_pass=not bad)


# ---------------------------------------------------------------- main
def main(mode):
    selftest = mode == "selftest"
    L = load_preds(selftest)
    F = pairs_frame()
    U = load_units(BASE_UNITS)
    if selftest:
        G = U[U.rung == "g2"].copy()
        G["rung"] = "geo"
        U = pd.concat([U, G], ignore_index=True)
    else:
        U = pd.concat([U, load_units(GEO_UNITS)], ignore_index=True)
    rep, rep8, mval = reps(U)
    P23 = common_pairs(L, F, ("g2", "geo"))
    P023 = common_pairs(L, F, ("dft", "g2", "geo"))
    own = {r: common_pairs(L, F, (r,)) for r in ("dft", "g2", "geo")}
    M = pd.concat([metrics(L, F, rep, rep8, ("g2", "geo"), P23, "P_23"),
                   metrics(L, F, rep, rep8, ("dft", "g2", "geo"), P023, "P_023")]
                  + [metrics(L, F, rep, rep8, (r,), own[r], f"own_{r}") for r in own], ignore_index=True)
    C = pd.concat([compare(L, F, rep, rep8, "geo", "g2", P23, "P_23"),
                   compare(L, F, rep, rep8, "geo", "dft", P023, "P_023")], ignore_index=True)
    out = RES / ("selftest" if selftest else "")
    out.mkdir(parents=True, exist_ok=True)
    M.to_csv(out / "E_metrics.csv", index=False)
    C.to_csv(out / "E_compare.csv", index=False)
    recovery(M).to_csv(out / "E_recovery.csv", index=False)
    by_class(L, F, rep, rep8, P023).to_csv(out / "E_bin.csv", index=False)
    app_metrics(L, F, rep, rep8, P023).to_csv(out / "E_app.csv", index=False)
    rep_out = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in rep.items()}
    rep8_out = {f"{k[0]}|{k[1]}": list(v) for k, v in rep8.items()}
    report = dict(mode=mode, n_pairs=dict(P_23=len(P23), P_023=len(P023), **{f"own_{r}": len(v) for r, v in own.items()}),
                  n_para=dict(P_23=int(F.set_index("pair_id").pos.reindex(P23).eq("p").sum()),
                              P_023=int(F.set_index("pair_id").pos.reindex(P023).eq("p").sum())),
                  representatives=rep_out, rep8=rep8_out, verdict=verdict(C), goal_mae=goal(M))
    if selftest:
        report["selftest"] = selftest_checks(M, C, rep)
    (out / "E_report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ("n_pairs", "verdict")} | ({"selftest": report.get("selftest")}
                                                                        if selftest else {}), indent=1, default=float))
    return 3 if selftest and not report["selftest"]["all_pass"] else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:                     # GEO2b decision D0D1 (user 2026-10-31): no geo-vs-g2 verdict
        print("NOT READY: 'run' (the GEO2.md 9-10 comparison with dft / g2 and its verdict) is dropped by the user "
              "decision D0D1; the GEO2b evaluation is 'eval_d0d1.py run' (results/DECISIONS_ZE.md E-8)", file=sys.stderr)
        sys.exit(2)
    if sys.argv[1:] == ["selftest"]:
        sys.exit(main(sys.argv[1]))
    raise SystemExit(__doc__)
