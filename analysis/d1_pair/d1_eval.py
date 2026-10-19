#!/usr/bin/env python3
"""d1_eval.py — D1PAIR Phase E (§5-2 .. §5-4, §10-1): metrics, tests, stage verdicts, Hammett-level evaluation.

  python d1_eval.py all        # every E_* output under results/ (exit 3 if an input is missing)

Inputs: results/M_predictions.parquet + M_units.csv (Phase M), S_folds.csv, F_pairs.csv, f0_d1.parquet + f0_report.json.
Arms: null (0), A (f0 representative difference; every f0 model too, appendix rows), B / C (6 models each), P, B_ref.
Representative g model (§3-3): per split x arm x target, lowest mean over the outer folds of the inner-CV MAE (ties: model
order). All metrics pool the test pairs of every outer fold. Primary pairs = P_main (reference); anchor pairs (A_main) and
P_sens are reported separately.
Hammett level (reference P_main test pairs, para, sigma_p known; hammett_eval.family_series with min 12 points incl. H):
R2_fam = 1 - sum (yhat - y)^2 / sum (y - ybar)^2 on y = -dd / 1.3642 with H at (0, 0); rho error |rho_hat - rho|
(hammett_eval.rho_r2 slope); bend / V calls = hammett_eval.calls on DFT and predicted values of each channel.
Tests: core cluster bootstrap (2,000, rng 20261019; the cores of the test pairs resampled with replacement), d = MAE(X) -
MAE(B); 95 % percentile CI; two-sided p = min(1, 2 min((1 + #{d* <= 0}) / (B + 1), (1 + #{d* >= 0}) / (B + 1))); Holm over
the 9 primary targets (rev6_common.holm). Sensitivities with Clopper-Pearson 95 % CI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1pair_common as C  # noqa: E402
import hammett_eval as HE  # noqa: E402

CH = list(HE.CH)                       # barrier + 8 channels (appendix D order)
ARM_ORDER = ["null", "A", "B", "C", "P", "B_ref"]


# ================================================================ loading
def load():
    S = pd.read_csv(C.RES / "S_folds.csv")
    F = pd.read_csv(C.RES / "F_pairs.csv", usecols=["pair_id"] + [f"dd_{t}" for t in C.TARGETS])
    S = S.merge(F, on="pair_id", how="left", validate="1:1").set_index("pair_id")
    M = pd.read_parquet(C.RES / "M_predictions.parquet")
    U = pd.read_csv(C.RES / "M_units.csv")
    f0r = json.loads((C.RES / "f0_report.json").read_text())
    f0 = pd.read_parquet(C.RES / "f0_d1.parquet")
    return S, M, U, f0r, f0


def representatives(U):
    """{(split, arm, target): model} by the mean inner-CV MAE over the outer folds."""
    rep = {}
    g = U.groupby(["split", "arm", "target", "model"]).inner_cv_mae.mean().reset_index()
    for (sp, arm, t), d in g.groupby(["split", "arm", "target"]):
        d = d.assign(o=d.model.map({m: i for i, m in enumerate(C.MODELS)})).sort_values(["inner_cv_mae", "o"])
        rep[(sp, arm, t)] = d.iloc[0].model
    return rep


def baseline_preds(S, f0, f0r, test_ids):
    """arm A (representative f0 and every f0 model) and null predictions for the given pair ids."""
    P = f0.pivot_table(index="rxn_id", columns=["target", "model"], values="yhat")
    out = []
    a, b = S.loc[test_ids, "a_rxn"].to_numpy(), S.loc[test_ids, "b_rxn"].to_numpy()
    for t in C.TARGETS:
        y = S.loc[test_ids, f"dd_{t}"].to_numpy(float)
        for m in C.MODELS:
            d = P[(t, m)].reindex(a).to_numpy() - P[(t, m)].reindex(b).to_numpy()
            out.append(pd.DataFrame(dict(pair_id=test_ids, arm="A", model=m, target=t, y=y, yhat=d,
                                         rep=m == f0r["representative"][t])))
        out.append(pd.DataFrame(dict(pair_id=test_ids, arm="null", model="0", target=t, y=y, yhat=0.0, rep=True)))
    return pd.concat(out, ignore_index=True)


def rep_table(S, M, U, f0, f0r):
    """long table: split, arm, model, target, pair_id, y, yhat, rep flag (B/C/P/B_ref from M; A/null per split)."""
    reps = representatives(U)
    M = M.copy()
    M["rep"] = [reps.get((sp, a, t)) == m for sp, a, t, m in zip(M.split, M.arm, M.target, M.model)]
    parts = [M[["split", "arm", "model", "target", "pair_id", "y", "yhat", "rep", "fold"]]]
    for sp in sorted(M.split.unique()):
        ids = sorted(M[M.split == sp].pair_id.unique())
        bl = baseline_preds(S, f0, f0r, ids)
        fold = S.loc[bl.pair_id, f"fold_{sp}"].to_numpy()
        parts.append(bl.assign(split=sp, fold=fold))
    L = pd.concat(parts, ignore_index=True)
    L = L.join(S[["ref_kind", "in_P_main", "in_P_sens", "in_A_main", "pos", "core", "fam", "sigma", "Xb", "series",
                  "d0_overlap", "hidden_V_fam", "hidden_bend_fam", "t2_exclude"]], on="pair_id")
    return L, reps


# ================================================================ metrics
def mae(e):
    return float(np.mean(np.abs(e))) if len(e) else np.nan


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan


def hammett_frame(d, col):
    """pair rows -> the frame hammett_eval.family_series expects (fam, pos, sigma, Xb, col)."""
    return d[["fam", "pos", "sigma", "Xb"]].assign(**{col: d["yhat"].to_numpy() if col == "pred" else d["y"].to_numpy()})


def family_eval(d):
    """per para family of one (split, arm, model, target) table: R2_fam, rho error (P_main reference pairs)."""
    d = d[d.in_P_main & (d.ref_kind == "reference") & (d.pos == "p") & d.sigma.notna()]
    out = []
    for f, g in d.groupby("fam"):
        sy = HE.family_series(hammett_frame(g, "y"), f, "y", C.MIN_X_FAM)
        sp = HE.family_series(hammett_frame(g, "pred"), f, "pred", C.MIN_X_FAM)
        if sy is None:
            continue
        x, y, _ = sy
        yh = sp[1]
        r2 = 1 - ((yh - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rho, _ = HE.rho_r2(x, y)
        rho_h, _ = HE.rho_r2(x, yh)
        out.append(dict(fam=f, n_points=len(x), r2_fam=float(r2), rho=float(rho), rho_hat=float(rho_h),
                        rho_err=float(abs(rho_h - rho))))
    return pd.DataFrame(out)


def metrics(L, U):
    rows = []
    trn = U.groupby(["split", "arm", "model", "target"]).train_mae.mean()
    for (sp, arm, m, t), d in L.groupby(["split", "arm", "model", "target"]):
        rep = bool(d.rep.iloc[0])
        sets = {"P_main": d[d.in_P_main & (d.ref_kind == "reference")], "A_main": d[d.in_A_main & (d.ref_kind == "anchor")],
                "P_sens": d[d.in_P_sens & (d.ref_kind == "reference")],
                "P_main_no_d0_overlap": d[d.in_P_main & (d.ref_kind == "reference") & ~d.d0_overlap.astype(bool)]}
        fe = family_eval(d)
        for sname, s in sets.items():
            for pos in ["all"] + C.POSITIONS:
                q = s if pos == "all" else s[s.pos == pos]
                e = (q.yhat - q.y).to_numpy(float)
                r = dict(split=sp, arm=arm, model=m, target=t, rep=rep, set=sname, pos=pos, n=len(q), mae=mae(e), rmse=rmse(e),
                         train_mae=float(trn.get((sp, arm, m, t), np.nan)))
                if sname == "P_main" and pos == "all" and len(fe):
                    r.update(n_fam=len(fe), r2_fam_median=float(fe.r2_fam.median()), r2_fam_q1=float(fe.r2_fam.quantile(0.25)),
                             r2_fam_q3=float(fe.r2_fam.quantile(0.75)), rho_err_median=float(fe.rho_err.median()))
                rows.append(r)
    return pd.DataFrame(rows)


# ================================================================ bootstrap tests
def boot_diff(dx, db, cores, n_boot=C.N_BOOT, seed=C.SEED):
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


def clopper(k, n, a=0.05):
    from scipy.stats import beta
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def paired(L, sp, t, x, b, sset="P_main", extra=None):
    """aligned errors of two (arm, model|rep) selections on the test pairs of a split / set."""
    d = L[(L.split == sp) & (L.target == t)]
    if sset == "P_main":
        d = d[d.in_P_main & (d.ref_kind == "reference")]
    elif sset == "P_sens":
        d = d[d.in_P_sens & (d.ref_kind == "reference")]
    elif sset == "no_d0":
        d = d[d.in_P_main & (d.ref_kind == "reference") & ~d.d0_overlap.astype(bool)]
    if extra is not None:
        d = d[extra(d)]

    def pick(sel):
        arm, model = sel
        q = d[(d.arm == arm) & ((d.rep) if model == "rep" else (d.model == model))]
        return q.set_index("pair_id")
    qx, qb = pick(x), pick(b)
    ids = qx.index.intersection(qb.index)
    return (qx.loc[ids, "yhat"] - qx.loc[ids, "y"]).to_numpy(float), (qb.loc[ids, "yhat"] - qb.loc[ids, "y"]).to_numpy(float), \
        qx.loc[ids, "core"].to_numpy(), qx.loc[ids, "fold"].to_numpy(), qb.loc[ids, "fold"].to_numpy()


def fold_wins(ex, eb, fold):
    w = 0
    for k in np.unique(fold):
        m = fold == k
        w += int(np.mean(np.abs(eb[m])) < np.mean(np.abs(ex[m])))
    return w, len(np.unique(fold))


def primary_tests(L):
    rows = []
    for t in C.PRIMARY_TARGETS:
        ea, eb, cores, fa, fb = paired(L, "T1", t, ("A", "rep"), ("B", "rep"))
        d, lo, hi, p = boot_diff(ea, eb, cores)
        wins, nf = fold_wins(ea, eb, fb)
        rows.append(dict(target=t, n_pairs=len(ea), n_cores=len(np.unique(cores)), mae_A=float(np.mean(np.abs(ea))),
                         mae_B=float(np.mean(np.abs(eb))), d_A_minus_B=d, ci_lo=lo, ci_hi=hi, p=p, folds_B_lower=wins, n_folds=nf))
    T = pd.DataFrame(rows)
    T["p_holm"] = C.holm(T.p.to_numpy())
    T["significant"] = (T.p_holm < C.ALPHA) & (T.ci_lo > 0)
    return T


def secondary_tests(L, reps):
    rows = []

    def add(kind, sp, t, x, b, sset="P_main", extra=None, label=None):
        ex, eb, cores, _, fb = paired(L, sp, t, x, b, sset, extra)
        if not len(ex):
            return
        d, lo, hi, p = boot_diff(ex, eb, cores)
        rows.append(dict(comparison=kind, label=label or f"{x[0]}:{x[1]} vs {b[0]}:{b[1]}", split=sp, set=sset, target=t,
                         n_pairs=len(ex), mae_x=float(np.mean(np.abs(ex))), mae_B=float(np.mean(np.abs(eb))),
                         d_x_minus_B=d, ci_lo=lo, ci_hi=hi, p=p))
    for t in C.TARGETS:
        for x in (("C", "rep"), ("null", "0"), ("P", "KRR")):
            add(f"B vs {x[0]}", "T1", t, x, ("B", "rep"))
        add("B_KRR vs B_ref_KRR", "T1", t, ("B_ref", "KRR"), ("B", "KRR"))
        for i, m1 in enumerate(C.MODELS):
            for m2 in C.MODELS[i + 1:]:
                add("model pair (B)", "T1", t, ("B", m1), ("B", m2), label=f"B:{m1} vs B:{m2}")
        for sp in ("LODO", "LOSO"):
            add(f"{sp}: B vs A", sp, t, ("A", "rep"), ("B", "rep"))
            add(f"{sp}: B vs C", sp, t, ("C", "rep"), ("B", "rep"))
            ser = "R1" if sp == "LODO" else "R2"
            add(f"{sp} {ser} pairs: B vs A", sp, t, ("A", "rep"), ("B", "rep"), extra=lambda d, s=ser: d.series == s)
        add("P_sens: B vs A", "T1", t, ("A", "rep"), ("B", "rep"), sset="P_sens")
        add("no D0 overlap: B vs A", "T1", t, ("A", "rep"), ("B", "rep"), sset="no_d0")
        add("T2: B vs A", "T2", t, ("A", "rep"), ("B", "rep"))
    # sum of the 9 channel predictions (each with its B representative) vs the direct barrier
    for sp in ("T1",):
        d = L[(L.split == sp) & (L.arm == "B") & L.rep & L.in_P_main & (L.ref_kind == "reference")]
        w = d.pivot_table(index="pair_id", columns="target", values="yhat")
        y = d[d.target == "barrier"].set_index("pair_id").y
        ids = w.index.intersection(y.index)
        es = w.loc[ids, C.CHANNELS9].sum(axis=1).to_numpy() - y.loc[ids].to_numpy()
        ed = w.loc[ids, "barrier"].to_numpy() - y.loc[ids].to_numpy()
        cores = d[d.target == "barrier"].set_index("pair_id").loc[ids, "core"].to_numpy()
        dd, lo, hi, p = boot_diff(es, ed, cores)
        rows.append(dict(comparison="sum-of-9 vs direct barrier (B rep)", label="B:sum9 vs B:direct", split=sp, set="P_main",
                         target="barrier", n_pairs=len(ids), mae_x=float(np.mean(np.abs(es))), mae_B=float(np.mean(np.abs(ed))),
                         d_x_minus_B=dd, ci_lo=lo, ci_hi=hi, p=p))
    return pd.DataFrame(rows)


# ================================================================ bends
def bends(L, splits=("T1", "T2")):
    rows = []
    for sp in splits:
        d = L[(L.split == sp) & (L.arm == "B") & L.rep & L.in_P_main & (L.ref_kind == "reference") & (L.pos == "p")
              & L.sigma.notna()]
        for f, g in d.groupby("fam"):
            for c in CH:
                q = g[g.target == c]
                sy = HE.family_series(hammett_frame(q, "y"), f, "y", C.MIN_X_FAM)
                if sy is None:
                    continue
                sp_ = HE.family_series(hammett_frame(q, "pred"), f, "pred", C.MIN_X_FAM)
                x, y, names = sy
                Y = np.c_[y, sp_[1]]
                ok, v, gain, psi, Lsl, Rsl = HE.calls(x, Y, names)
                rows.append(dict(split=sp, fam=f, target=c, n_points=len(x), bent_dft=bool(ok[0]), bent_pred=bool(ok[1]),
                                 V_dft=bool(v[0]), V_pred=bool(v[1]), gain_dft=float(gain[0]), gain_pred=float(gain[1]),
                                 psi_dft=float(psi[0]), psi_pred=float(psi[1]), left_dft=float(Lsl[0]), right_dft=float(Rsl[0]),
                                 left_pred=float(Lsl[1]), right_pred=float(Rsl[1]),
                                 hidden_V_fam=bool(q.hidden_V_fam.iloc[0]), hidden_bend_fam=bool(q.hidden_bend_fam.iloc[0])))
    return pd.DataFrame(rows)


def bend_rates(Bd, sp, fams=None):
    out = {}
    for c in CH:
        q = Bd[(Bd.split == sp) & (Bd.target == c)]
        if fams is not None:
            q = q[q.fam.isin(fams)]
        tp, pos = int((q.bent_dft & q.bent_pred).sum()), int(q.bent_dft.sum())
        fp, neg = int((~q.bent_dft & q.bent_pred).sum()), int((~q.bent_dft).sum())
        vtp, vpos = int((q.V_dft & q.V_pred).sum()), int(q.V_dft.sum())
        out[c] = dict(n_fam=len(q), sens=tp / pos if pos else np.nan, sens_ci=clopper(tp, pos), n_bent_dft=pos,
                      fpr=fp / neg if neg else np.nan, fpr_ci=clopper(fp, neg), n_straight_dft=neg,
                      V_sens=vtp / vpos if vpos else np.nan, V_sens_ci=clopper(vtp, vpos), n_V_dft=vpos,
                      V_fp=int((~q.V_dft & q.V_pred).sum()))
    return out


# ================================================================ stages / flow
def stages(Mt, Pt, Br):
    rows = []
    for t in C.TARGETS:
        b = Mt[(Mt.split == "T1") & (Mt.arm == "B") & Mt.rep & (Mt.set == "P_main") & (Mt.pos == "all") & (Mt.target == t)]
        nl = Mt[(Mt.split == "T1") & (Mt.arm == "null") & (Mt.set == "P_main") & (Mt.pos == "all") & (Mt.target == t)]
        bp = Mt[(Mt.split == "T1") & (Mt.arm == "B") & Mt.rep & (Mt.set == "P_main") & (Mt.pos == "p") & (Mt.target == t)]
        pt = Pt[Pt.target == t]
        s0 = (bool(b.mae.iloc[0] < nl.mae.iloc[0]) and bool(pt.significant.iloc[0])) if len(pt) else None
        r2 = float(b.r2_fam_median.iloc[0]) if "r2_fam_median" in b and b.r2_fam_median.notna().any() else np.nan
        s1 = bool(r2 >= C.STAGE1_R2) if np.isfinite(r2) else None
        s2 = None
        if t in C.STAGE2_TARGETS:
            br = Br.get(t, {})
            s2 = bool(bp.mae.iloc[0] <= C.STAGE2_MAE and (br.get("sens") or 0) >= C.STAGE2_SENS
                      and (br.get("fpr") if br.get("fpr") == br.get("fpr") else 1) <= C.STAGE2_FPR)
        rows.append(dict(target=t, model_B=b.model.iloc[0], mae_B=float(b.mae.iloc[0]), mae_null=float(nl.mae.iloc[0]),
                         stage0=s0, r2_fam_median=r2, stage1=s1, mae_para_B=float(bp.mae.iloc[0]),
                         bend_sens=Br.get(t, {}).get("sens"), bend_fpr=Br.get(t, {}).get("fpr"), stage2=s2))
    return pd.DataFrame(rows)


def data_flow(S):
    p = json.loads((C.RES / "P_report.json").read_text())
    f = json.loads((C.RES / "F_report.json").read_text())
    g = json.loads((C.RES / "G_report.json").read_text())
    rows = [("D0", "labels_all.json rows", 5265), ("D0", "f0 training rows (rev 6 appendix A)", C.N_D0),
            ("D1", "labels_d1.json rows", p["gates"]["P1_n_labels"]["got"]),
            ("D1", "excluded (D'', ex > 20, flags)", p["gates"]["P1_excluded"]["got"]),
            ("D1", "kept rows", p["gates"]["P1_kept"]["got"]), ("D1", "G1 targets (pairs' reactions)", g["n"]),
            ("D1", "G1 ok (first attempt)", g["n_ok_first"]), ("D1", "G1 recovered", g["n_recovered"]),
            ("D1", "G1 ok", g["n_ok"]), ("D1", "182 features ok", f["n_feat"])]
    fl = f["pairs"]["flow"]
    rows += [("pairs", k, v) for k, v in fl.items() if not isinstance(v, dict)]
    rows += [("pairs", "T_main (P_main + A_main)", int(S.in_T_main.sum())),
             ("pairs", "D1 reactions in T_main pairs", f["pairs"]["d1_reactions_in_T_main_pairs"])]
    return pd.DataFrame(rows, columns=["data", "step", "n"])


def input_gate():
    """Phase E refuses inputs that are incomplete, STOPped or changed after PREREG (exit 3)."""
    import os
    import d1_split as SP
    import d1_train as T
    need = [C.RES / f for f in ("M_predictions.parquet", "M_units.csv", "S_folds.csv", "F_pairs.csv", "f0_d1.parquet",
                                "M_report.json", "f0_report.json", "S_report.json", "P_report.json", "F_report.json",
                                "G_report.json", "P4_report.json", "G_prep_report.json", "G0_report.json", "F0_report.json")]
    bad = [f"missing {p.name}" for p in need if not p.is_file()]
    if bad:
        print("Phase E input gate:\n  " + "\n  ".join(bad), file=sys.stderr); sys.exit(3)
    pre = SP.gate_prereg()
    mrep = json.loads((C.RES / "M_report.json").read_text())
    f0r = json.loads((C.RES / "f0_report.json").read_text())
    if mrep.get("stop") or mrep.get("n_missing") or mrep.get("n_leak") or mrep.get("m1_fail"):
        bad.append(f"M_report stop={mrep.get('stop')} missing={mrep.get('n_missing')} leak={mrep.get('n_leak')} "
                   f"m1_fail={mrep.get('m1_fail')}")
    if C.sha256(C.RES / "M_predictions.parquet") != mrep["predictions"]["sha256"]:
        bad.append("M_predictions.parquet differs from M_report.json")
    if C.sha256(C.RES / "M_units.csv") != mrep.get("units", {}).get("sha256"):
        bad.append("M_units.csv differs from M_report.json")
    if f0r["representative"] != pre["f0_representative"]:
        bad.append("f0 representative differs from PREREG")
    os.environ.pop("D1PAIR_MODELS", None)
    exp = {T.uname(u) for g in T.GROUPS for u in T.unit_list(g)}
    U = pd.read_csv(C.RES / "M_units.csv")
    got = {T.uname(dict(split=r.split, arm=r.arm, model=r.model, target=r.target, fold=int(r.fold))) for r in U.itertuples()}
    if got != exp:
        bad.append(f"units: {len(exp - got)} missing, {len(got - exp)} unexpected")
    M = pd.read_parquet(C.RES / "M_predictions.parquet", columns=["split", "arm", "model", "target", "pair_id"])
    for sp, d in M.groupby("split"):
        ids = set(d.pair_id)
        for k, g in d.groupby(["arm", "model", "target"]):
            if set(g.pair_id) != ids or g.pair_id.duplicated().any():
                bad.append(f"{sp} {k}: test pairs differ from the split's test set"); break
    if bad:
        print("Phase E input gate:\n  " + "\n  ".join(bad), file=sys.stderr); sys.exit(3)
    return pre


def cmd_all():
    pre = input_gate()
    S, M, U, f0r, f0 = load()
    L, reps = rep_table(S, M, U, f0, f0r)
    Mt = metrics(L, U)
    C.write_atomic(C.RES / "E_metrics.csv", lambda t: Mt.to_csv(t, index=False))
    Pt = primary_tests(L)
    C.write_atomic(C.RES / "E_primary_tests.csv", lambda t: Pt.to_csv(t, index=False))
    St = secondary_tests(L, reps)
    Bd = bends(L)
    br1 = bend_rates(Bd, "T1")
    hidden_v = sorted(S.loc[S.hidden_V_fam.astype(bool), "fam"].unique())
    hidden_b = sorted(S.loc[S.hidden_bend_fam.astype(bool), "fam"].unique())
    t2 = []
    for sp in ("T1", "T2"):
        for name, fams in (("hidden_V", hidden_v), ("hidden_bend", hidden_b)):
            for c, r in bend_rates(Bd, sp, fams).items():
                mm = L[(L.split == sp) & (L.arm == "B") & L.rep & (L.target == c) & L.fam.isin(fams) & L.in_P_main]
                t2.append(dict(comparison=f"T2 vs T1 {name}", split=sp, target=c, n_fam=r["n_fam"], sens=r["sens"],
                               sens_ci_lo=r["sens_ci"][0], sens_ci_hi=r["sens_ci"][1], fpr=r["fpr"], V_sens=r["V_sens"],
                               V_sens_ci_lo=r["V_sens_ci"][0], V_sens_ci_hi=r["V_sens_ci"][1], n_V_dft=r["n_V_dft"],
                               mae_x=mae((mm.yhat - mm.y).to_numpy(float)), n_pairs=len(mm)))
    St = pd.concat([St, pd.DataFrame(t2)], ignore_index=True)
    C.write_atomic(C.RES / "E_secondary_tests.csv", lambda t: St.to_csv(t, index=False))
    C.write_atomic(C.RES / "E_bends.csv", lambda t: Bd.to_csv(t, index=False))
    Sg = stages(Mt, Pt, br1)
    C.write_atomic(C.RES / "E_stages.csv", lambda t: Sg.to_csv(t, index=False))
    C.write_json_safe(C.RES / "E_representative.json",
                      {f"{sp}|{a}|{t}": m for (sp, a, t), m in sorted(reps.items())} | {f"f0|{t}": m for t, m in f0r["representative"].items()})
    Fl = data_flow(S)
    C.write_atomic(C.RES / "E_data_flow.csv", lambda t: Fl.to_csv(t, index=False))
    import d1_split as SP
    pre = json.loads(SP.PREREG_JSON.read_text())
    E = dict(inputs_sha256={str(p): C.sha256(p) for p in (C.RES / "M_predictions.parquet", C.RES / "M_units.csv",
                                                          C.RES / "S_folds.csv", C.RES / "F_pairs.csv", C.RES / "f0_d1.parquet")},
             prereg_inputs_sha256=pre["inputs_sha256"], bend_rates_T1=br1,
             gates={k: json.loads((C.RES / f).read_text()).get("stop") for k, f in
                    (("P", "P_report.json"), ("P4", "P4_report.json"), ("G_prep", "G_prep_report.json"), ("G0", "G0_report.json"),
                     ("G", "G_report.json"), ("F", "F_report.json"), ("F0chk", "F0_report.json"), ("f0", "f0_report.json"),
                     ("S", "S_report.json"), ("M", "M_report.json"))},
             n_test_P_main_T1=int(L[(L.split == "T1") & (L.arm == "null") & (L.target == "barrier") & L.in_P_main
                                    & (L.ref_kind == "reference")].shape[0]))
    C.write_json_safe(C.RES / "E.json", E)
    print(Pt.to_string()); print(Sg.to_string())
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["all"]:
        sys.exit(cmd_all())
    raise SystemExit(__doc__)
