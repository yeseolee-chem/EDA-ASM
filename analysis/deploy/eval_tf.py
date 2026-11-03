#!/usr/bin/env python3
r"""eval_tf.py — DEPLOY-MODEL Phase E: lf . S . TF cross-validation predictions vs LADDER g2 . S . TF
(docs/specs/DEPLOY_MODEL.md 12, G-C of 10). CPU, env espley or reactot (pandas, scipy); jobs/cpu_py.slurm.

  python eval_tf.py selftest   the evaluation code on the LADDER g2 . S . TF predictions alone (results/L_predictions.parquet
                               rung g2, arm S, model TF, also relabelled as 'lf'): must reproduce the spec 0-4 numbers
                               (para MAE on 1,658 P_main para pairs; continuous g AUC on 89 para families), the E2 para
                               CIs, the E2 bend calls on the P_main families, DEPLOY_MODEL_W 0-1 hidden-bend rates, and
                               G-C machinery on identical inputs (d = 0, CI = [0, 0], pass) -> results/E_selftest.json,
                               tables in $DEP/E_selftest/. Exit 3 when a check fails.
  python eval_tf.py run        results/K_predictions.parquet (lf) vs the LADDER g2 . S . TF rows -> results/E_metrics.csv,
                               E_compare.csv, E_bends.csv, E_bend_rates.json, E_ad.json, E_report.json (G-C verdict).
                               Exit 0 = written (also when G-C fails: E_report G_C.pass false; train_tf.py final then
                               refuses), 2 = inputs missing, 3 = STOP (inconsistent inputs).

Rules (fixed before any lf result; PREREG_DEPLOY.md)
  pair attributes  D1PAIR S_folds.csv (core, fam, sigma, Xb, in_P_main, ...); pos / ref_kind must equal the predictions.
  E_metrics        model (lf, g2) x pair set (own = the model's test pairs, common = pairs with both predictions) x
                   evaluation set (P_main = reference & in_P_main [primary]; test_all = every test pair) x target x pos
                   (all, p, m, o, direct): n, MAE, RMSE; P_main x p: core bootstrap 95 % CI of the MAE (= eval_geo2.boot_diff
                   (e, 0, core), 2,000 resamples, seed 20261019; = LADDER E2 para CI).
  E_compare / G-C  common P_main pairs, pos p (decision) and m (descriptive): d = MAE_lf - MAE_g2 with eval_geo2.boot_diff
                   (core bootstrap), relative r = d / MAE_g2 with its CI from the SAME resamples (boot_rel). G-C passes iff
                   the 95 % upper bound of r <= +5 % for each of the 9 primary targets (d1 d2 elst pauli oi disp cpcm cds
                   barrier).
  E_bends          families = hammett_eval.family_series (>= 12 points incl. H at (0, 0)) on the REFERENCE test pairs
                   with sigma (= P_sens reference pairs; 89 para families for LADDER g2 . S . TF, spec 0-4), pos p (and m,
                   descriptive, same rule with sigma_m); per family x target (hammett_eval.CH): hammett_eval.calls on
                   [DFT y, prediction] -> bend / V call, AICc gain g (bent_gain), psi, arm slopes. Sets own and common
                   (common: identical families and DFT calls for lf and g2, side by side).
  E_bend_rates     per model x set x pos x target: sensitivity / false-positive rate / V sensitivity with Clopper-Pearson
                   CIs (d1_eval.bend_rates), continuous g AUC (DFT call = truth, predicted g = score; Mann-Whitney, ties
                   1/2) with a family bootstrap 95 % CI (2,000, seed 20261019; common set: paired lf - g2 difference);
                   hidden-bend families (DFT: barrier straight & >= 1 of the 8 channels bent; prediction: the same rule on
                   the predicted calls): base rate, sensitivity, FPR, precision (Clopper-Pearson).
  E_ad             per target the 95th percentile (numpy linear) of the lf test yhat_seed_sd -> applicability threshold.
Reused code: analysis/geo2/eval_geo2.py boot_diff (41-54); analysis/d1_pair/hammett_eval.py family_series (60-67), calls
(37-50), bent_gain (19-34), CH; analysis/d1_pair/d1_eval.py bend_rates (287-300), clopper (167-173).
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
import dep_common as C  # noqa: E402

sys.path.insert(0, str(C.REPO / "analysis" / "d1_pair"))
sys.path.insert(0, str(C.REPO / "analysis" / "geo2"))
import hammett_eval as HE  # noqa: E402
import d1_eval as E9  # noqa: E402
import eval_geo2 as EG  # noqa: E402

TARGETS = list(C.TARGETS)
PRIMARY = list(C.PRIMARY)
CH = list(HE.CH)                                      # barrier + 8 channels
CH8 = [c for c in CH if c != "barrier"]
POSITIONS = ("all", "p", "m", "o", "direct")
MIN_X = 12
GC_REL_MAX = 0.05
AD_PCT = 95.0
L_PRED = C.REPO / "analysis" / "d1_ladder" / "results" / "L_predictions.parquet"
E2_METRICS = C.REPO / "analysis" / "d1_ladder" / "results" / "E2_metrics.csv"
E2_BENDS = C.REPO / "analysis" / "d1_ladder" / "results" / "E2_bends.csv"
S_FOLDS = C.REPO / "analysis" / "d1_pair" / "results" / "S_folds.csv"
K_PRED = C.RES / "K_predictions.parquet"
SELF_DIR = C.DEP / "E_selftest"
# spec 0-4 (LADDER g2 . S . TF, D1 T1 test, 1,658 para pairs / 89 para families) and DEPLOY_MODEL_W 0-1
SPEC_MAE = dict(oi=1.014, cpcm=0.299, pauli=1.735, elst=0.879, d1=0.374, d2=0.298, disp=0.131, cds=0.027, barrier=0.327)
SPEC_AUC = dict(oi=0.845, cpcm=0.787, pauli=0.775, elst=0.757, d1=0.916, d2=0.686, cds=0.466, barrier=0.834)
SPEC_HIDDEN = dict(sens=0.65, fpr=0.21, ppv=0.57, base=0.29)
SPEC_N = dict(para_pairs=1658, para_families=89)
RATE_RULES = dict(
    families="hammett_eval.family_series (>= 12 points incl. H) on the reference test pairs with sigma, per position",
    calls="hammett_eval.calls on [DFT y, prediction]: bend (AICc gain >= 4, robust to one point), V call",
    rates="d1_eval.bend_rates: sensitivity / FPR / V sensitivity, Clopper-Pearson 95 % CI",
    auc_g="continuous g AUC: DFT bend call = truth, predicted AICc gain g = score (Mann-Whitney, ties 1/2), family "
          "bootstrap 95 % CI (2,000 resamples, seed 20261019); common set: paired lf - g2 difference on the same resamples",
    hidden="hidden-bend family = barrier straight & >= 1 of the 8 channels bent (DFT calls; prediction: predicted calls); "
           "base rate, sensitivity, FPR, precision with Clopper-Pearson CIs",
    sets="own = the model's test pairs, common = pairs predicted by both models (same families, same DFT calls)")
ATTR = ["pair_id", "a_rxn", "b_rxn", "core", "dipole", "fam", "pos", "Xb", "sigma", "ref_kind", "in_P_main", "in_P_sens",
        "in_A_main", "hidden_V_fam", "hidden_bend_fam"]


# ================================================================ inputs
def s_folds():
    S = pd.read_csv(S_FOLDS)
    S["pair_id"] = S.pair_id.astype(str)
    S["pos"] = S.pos.fillna("").astype(str)
    S["ref_kind"] = S.ref_kind.fillna("").astype(str)
    return S[[c for c in ATTR if c in S.columns]].set_index("pair_id")


def ladder_g2():
    L = pd.read_parquet(L_PRED)
    g = L[(L.rung == "g2") & (L.arm == "S") & (L.model == "TF")].copy()
    if g.empty:
        C.stop(f"{L_PRED}: no rung g2 / arm S / model TF rows")
    return g


def frame(pred, name, S):
    """predictions of one model -> long frame with the S_folds attributes; consistency checks (STOP)."""
    d = pred[["pair_id", "fold", "target", "y", "yhat", "yhat_seed_sd", "pos", "ref_kind"]].copy()
    d["pair_id"] = d.pair_id.astype(str)
    if d.duplicated(["pair_id", "target"]).any():
        C.stop(f"{name}: duplicated (pair_id, target) predictions")
    if set(d.target) != set(TARGETS):
        C.stop(f"{name}: targets {sorted(set(d.target))} != {TARGETS}")
    miss = sorted(set(d.pair_id) - set(S.index))
    if miss:
        C.stop(f"{name}: {len(miss)} pairs not in S_folds.csv: {miss[:5]}")
    a = S.reindex(d.pair_id)
    for c in ("pos", "ref_kind"):
        if (a[c].to_numpy() != d[c].to_numpy()).any():
            C.stop(f"{name}: column {c} differs from S_folds.csv")
    d = d.drop(columns=["pos", "ref_kind"]).join(S, on="pair_id")
    d["model"] = name
    return d


def check_y(A, B):
    m = A[["pair_id", "target", "y"]].merge(B[["pair_id", "target", "y"]], on=["pair_id", "target"])
    dy = float(np.abs(m.y_x - m.y_y).max()) if len(m) else 0.0
    if dy > 1e-12:
        C.stop(f"labels differ between the two prediction sets on common pairs (max {dy})")
    return dy


# ================================================================ statistics
def boot_rel(dx, db, cores, n_boot=C.N_BOOT, seed=C.SEED):
    """relative difference r = (MAE(x) - MAE(b)) / MAE(b) with the core resamples of eval_geo2.boot_diff (same rng use,
    same weight matrix): (r, ci_lo, ci_hi, d-percentiles for the consistency check)."""
    ex, eb = np.abs(dx), np.abs(db)
    uc, inv = np.unique(cores, return_inverse=True)
    sx, sb, cnt = np.bincount(inv, ex), np.bincount(inv, eb), np.bincount(inv)
    rng = np.random.default_rng(seed)
    k = len(uc)
    w = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)])
    num = w @ cnt
    mx, mb = (w @ sx) / num, (w @ sb) / num
    with np.errstate(divide="ignore", invalid="ignore"):
        rstar = (mx - mb) / mb
    r = (ex.mean() - eb.mean()) / eb.mean() if eb.mean() > 0 else np.nan
    lo, hi = np.nanpercentile(rstar, [2.5, 97.5]) if np.isfinite(rstar).any() else (np.nan, np.nan)
    dlo, dhi = np.percentile(mx - mb, [2.5, 97.5])
    return float(r), float(lo), float(hi), (float(dlo), float(dhi))


def auc(lab, score):
    """Mann-Whitney AUC (ties 1/2); NaN without both classes."""
    lab = np.asarray(lab, dtype=bool)
    s = np.asarray(score, dtype=float)
    npos, nneg = int(lab.sum()), int((~lab).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    from scipy.stats import rankdata
    rk = rankdata(s)
    return float((rk[lab].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def boot_auc(lab, score, idx):
    """AUC over the family resamples idx [n_boot, n_fam] -> array."""
    lab, score = np.asarray(lab, dtype=bool), np.asarray(score, dtype=float)
    return np.array([auc(lab[i], score[i]) for i in idx])


def pct_ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def rate(k, n):
    return dict(k=int(k), n=int(n), value=k / n if n else np.nan, ci=list(E9.clopper(k, n)))


# ================================================================ tables
def eval_mask(d, es):
    if es == "P_main":
        return (d.ref_kind == "reference").to_numpy() & d.in_P_main.astype(bool).to_numpy()
    return np.ones(len(d), dtype=bool)


def metrics(A, sets):
    """E_metrics rows: sets = {pair_set name: {model: pair ids}}."""
    rows = []
    for ps, by_model in sets.items():
        for model, ids in by_model.items():
            d0 = A[(A.model == model) & A.pair_id.isin(ids)]
            for es in ("P_main", "test_all"):
                d1 = d0[eval_mask(d0, es)]
                for t in TARGETS:
                    q = d1[d1.target == t]
                    e = (q.yhat - q.y).to_numpy(float)
                    for p in POSITIONS:
                        m = np.ones(len(q), dtype=bool) if p == "all" else (q.pos == p).to_numpy()
                        if not m.any():
                            continue
                        r = dict(model=model, pair_set=ps, eval_set=es, target=t, pos=p, n=int(m.sum()),
                                 mae=float(np.abs(e[m]).mean()), rmse=float(np.sqrt((e[m] ** 2).mean())))
                        if p == "p" and es == "P_main":
                            cores = q.core.to_numpy()[m]
                            _, lo, hi, _ = EG.boot_diff(e[m], np.zeros(int(m.sum())), cores)
                            r.update(n_cores=int(len(set(cores))), mae_ci_lo=lo, mae_ci_hi=hi)
                        rows.append(r)
    return pd.DataFrame(rows)


def compare(A, ids, x="lf", b="g2", thr=GC_REL_MAX):
    """E_compare rows on the common P_main pairs, pos p (G-C) and m (descriptive)."""
    rows = []
    for pos in ("p", "m"):
        for t in TARGETS:
            qx = A[(A.model == x) & (A.target == t) & A.pair_id.isin(ids)]
            qx = qx[eval_mask(qx, "P_main") & (qx.pos == pos).to_numpy()].set_index("pair_id")
            qb = A[(A.model == b) & (A.target == t)].set_index("pair_id").reindex(qx.index)
            if not len(qx):
                continue
            if qb.yhat.isna().any():
                C.stop(f"compare: {b} lacks predictions on common pairs ({t}, {pos})")
            ex, eb = (qx.yhat - qx.y).to_numpy(float), (qb.yhat - qb.y).to_numpy(float)
            cores = qx.core.to_numpy()
            d, lo, hi, p = EG.boot_diff(ex, eb, cores)
            r, rlo, rhi, (dlo, dhi) = boot_rel(ex, eb, cores)
            if abs(dlo - lo) > 1e-12 or abs(dhi - hi) > 1e-12:
                C.stop(f"boot_rel resamples differ from eval_geo2.boot_diff ({t}, {pos})")
            prim = t in PRIMARY and pos == "p"
            rows.append(dict(x=x, b=b, pos=pos, target=t, primary=prim, n=int(len(qx)), n_cores=int(len(set(cores))),
                             mae_x=float(np.abs(ex).mean()), mae_b=float(np.abs(eb).mean()), d=d, d_ci_lo=lo,
                             d_ci_hi=hi, p=p, rel=r, rel_ci_lo=rlo, rel_ci_hi=rhi,
                             gc_pass=(bool(rhi <= thr) if prim else None)))
    return pd.DataFrame(rows)


def gc_threshold():
    """the pre-registered G-C threshold (results/prereg_deploy.json gates.G_C.rel_ci_hi_max; default +5 %)."""
    import prereg_deploy as PR
    return PR.gc_threshold()


def gc_verdict(Cm, thr=None):
    thr = GC_REL_MAX if thr is None else float(thr)
    c = Cm[(Cm.pos == "p") & Cm.target.isin(PRIMARY)].set_index("target")
    miss = [t for t in PRIMARY if t not in c.index]
    per = {t: dict(rel=float(c.at[t, "rel"]), rel_ci_hi=float(c.at[t, "rel_ci_hi"]), mae_lf=float(c.at[t, "mae_x"]),
                   mae_g2=float(c.at[t, "mae_b"]), pass_=bool(c.at[t, "rel_ci_hi"] <= thr)) for t in c.index
           if t in PRIMARY}
    ok = not miss and all(v["pass_"] for v in per.values())
    return {"pass": bool(ok), "rule": "for each of the 9 primary targets: 95 % upper bound of (MAE_lf - MAE_g2) / MAE_g2 "
            f"<= +{thr * 100:g} % (common para P_main pairs, core bootstrap 2,000, seed 20261019)", "threshold": thr,
            "n_pairs": int(c.n.iloc[0]) if len(c) else 0, "missing": miss,
            "per_target": {t: {("pass" if k == "pass_" else k): v for k, v in r.items()} for t, r in per.items()},
            "failed": [t for t, v in per.items() if not v["pass_"]]}


def family_series_pos(P, fam, col, pos, min_x=MIN_X):
    """hammett_eval.family_series (60-67) with the position as a parameter (pos 'p' calls the original)."""
    if pos == "p":
        return HE.family_series(P, fam, col, min_x)
    t = P[(P.fam == fam) & (P.pos == pos) & P.sigma.notna()].drop_duplicates("Xb")
    if len(t) + 1 < min_x:
        return None
    x = np.r_[t.sigma.to_numpy(float), 0.0]
    y = np.r_[-t[col].to_numpy(float) / HE.K, 0.0]
    return x, y, list(t.Xb) + ["H"]


def bends(A, model, ids, pos, pmain_only=False):
    """per family x target: DFT / predicted bend + V calls, gains (hammett_eval.calls on [y, yhat])."""
    d = A[(A.model == model) & A.pair_id.isin(ids) & (A.ref_kind == "reference") & (A.pos == pos) & A.sigma.notna()]
    if pmain_only:
        d = d[d.in_P_main.astype(bool)]
    rows = []
    for f, g in d.groupby("fam"):
        for c in CH:
            q = g[g.target == c]
            base = q[["fam", "pos", "sigma", "Xb"]]
            sy = family_series_pos(base.assign(v=q.y.to_numpy()), f, "v", pos)
            if sy is None:
                continue
            sp = family_series_pos(base.assign(v=q.yhat.to_numpy()), f, "v", pos)
            x, y, names = sy
            ok, v, gain, psi, Ls, Rs = HE.calls(x, np.c_[y, sp[1]], names)
            rows.append(dict(model=model, pos=pos, fam=f, target=c, n_points=len(x), bent_dft=bool(ok[0]),
                             bent_pred=bool(ok[1]), V_dft=bool(v[0]), V_pred=bool(v[1]), gain_dft=float(gain[0]),
                             gain_pred=float(gain[1]), psi_dft=float(psi[0]), psi_pred=float(psi[1]),
                             left_dft=float(Ls[0]), right_dft=float(Rs[0]), left_pred=float(Ls[1]), right_pred=float(Rs[1]),
                             hidden_V_fam=bool(q.hidden_V_fam.iloc[0]) if "hidden_V_fam" in q else None,
                             hidden_bend_fam=bool(q.hidden_bend_fam.iloc[0]) if "hidden_bend_fam" in q else None))
    return pd.DataFrame(rows)


def hidden(B):
    """hidden-bend families: barrier straight & >= 1 of the 8 channels bent (DFT calls; prediction: predicted calls)."""
    w = B.pivot_table(index="fam", columns="target", values=["bent_dft", "bent_pred"], aggfunc="first")
    hd = (~w[("bent_dft", "barrier")].astype(bool)) & w["bent_dft"][CH8].astype(bool).any(axis=1)
    hp = (~w[("bent_pred", "barrier")].astype(bool)) & w["bent_pred"][CH8].astype(bool).any(axis=1)
    return pd.DataFrame(dict(hidden_dft=hd, hidden_pred=hp))


def hidden_rates(H):
    tp = int((H.hidden_dft & H.hidden_pred).sum())
    fp = int((~H.hidden_dft & H.hidden_pred).sum())
    fn = int((H.hidden_dft & ~H.hidden_pred).sum())
    tn = int((~H.hidden_dft & ~H.hidden_pred).sum())
    return dict(n_fam=int(len(H)), tp=tp, fp=fp, fn=fn, tn=tn, base=rate(tp + fn, len(H)), sens=rate(tp, tp + fn),
                fpr=rate(fp, fp + tn), ppv=rate(tp, tp + fp))


def bend_rates(B, boot_idx):
    """per target: d1_eval.bend_rates (sens, FPR, V sensitivity, Clopper-Pearson) + continuous g AUC (family bootstrap)."""
    fams = sorted(B.fam.unique())
    br = E9.bend_rates(B.assign(split="T1"), "T1")
    out = {}
    for c in CH:
        q = B[B.target == c].set_index("fam").reindex(fams)
        a = auc(q.bent_dft.to_numpy(bool), q.gain_pred.to_numpy(float))
        bs = boot_auc(q.bent_dft.to_numpy(bool), q.gain_pred.to_numpy(float), boot_idx)
        out[c] = dict(br.get(c, {}), auc_g=a, auc_g_ci=list(pct_ci(bs)), auc_boot=bs)
    return out, fams


def boot_index(n, n_boot=C.N_BOOT, seed=C.SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n)) if n else np.zeros((n_boot, 0), dtype=int)


def rates_block(A, sets, models, pos):
    """E_bends rows + E_bend_rates entries for one position over the pair sets."""
    allB, rec = [], {}
    for ps, by_model in sets.items():
        Bm = {m: bends(A, m, by_model[m], pos) for m in models if m in by_model}
        rec[ps] = {}
        fam_sets = {m: sorted(B.fam.unique()) for m, B in Bm.items()}
        if ps == "common" and len({tuple(v) for v in fam_sets.values()}) > 1:
            C.stop(f"common set: the models' {pos} families differ")
        for m, B in Bm.items():
            idx = boot_index(B.fam.nunique())
            r, fams = bend_rates(B, idx)
            H = hidden(B)
            rec[ps][m] = dict(n_fam=len(fams), targets={c: {k: v for k, v in r[c].items() if k != "auc_boot"} for c in CH},
                              hidden=hidden_rates(H), _boot={c: r[c]["auc_boot"] for c in CH})
            allB.append(B.assign(pair_set=ps))
        if ps == "common" and len(Bm) == 2:
            a, b = models
            diff = {}
            for c in CH:
                da = rec[ps][a]["_boot"][c] - rec[ps][b]["_boot"][c]
                diff[c] = dict(d=rec[ps][a]["targets"][c]["auc_g"] - rec[ps][b]["targets"][c]["auc_g"],
                               ci=list(pct_ci(da)))
            rec[ps][f"auc_diff_{a}_minus_{b}"] = diff
        for m in Bm:
            rec[ps][m].pop("_boot", None)
    return (pd.concat(allB, ignore_index=True) if allB else pd.DataFrame()), rec


def ad_thresholds(A, model="lf"):
    d = A[A.model == model]
    th = {t: float(np.percentile(d[d.target == t].yhat_seed_sd.to_numpy(float), AD_PCT)) for t in TARGETS}
    return dict(rule=f"{AD_PCT:.0f}th percentile (numpy linear) of the {model} test-pair yhat_seed_sd per target "
                     "(SD over the 5 final seeds of each fold); out_of_range = yhat_sd > threshold",
                thresholds=th, n_pairs=int(d.pair_id.nunique()), model=model)


# ================================================================ pipeline
def evaluate(lf, g2, thr=GC_REL_MAX):
    S = s_folds()
    A = pd.concat([frame(lf, "lf", S), frame(g2, "g2", S)], ignore_index=True)
    dy = check_y(A[A.model == "lf"], A[A.model == "g2"])
    own = {m: sorted(A[A.model == m].pair_id.unique()) for m in ("lf", "g2")}
    common = sorted(set(own["lf"]) & set(own["g2"]))
    sets = {"own": own, "common": {"lf": common, "g2": common}}
    M = metrics(A, sets)
    Cm = compare(A, common, thr=thr)
    gc = gc_verdict(Cm, thr)
    Bp, rp = rates_block(A, sets, ("lf", "g2"), "p")
    Bm, rm = rates_block(A, sets, ("lf", "g2"), "m")
    B = pd.concat([Bp, Bm], ignore_index=True)
    return dict(A=A, M=M, C=Cm, B=B, rates=dict(p=rp, m=rm), gc=gc, ad=ad_thresholds(A, "lf"),
                ad_g2=ad_thresholds(A, "g2"), n=dict(own_lf=len(own["lf"]), own_g2=len(own["g2"]), common=len(common)),
                dy=dy)


def summary(R):
    M = R["M"]
    pm = M[(M.eval_set == "P_main") & (M.pos == "p")]
    s = {}
    for ps in ("own", "common"):
        s[f"para_mae_{ps}"] = {m: {t: float(g[g.target == t].mae.iloc[0]) for t in TARGETS if (g.target == t).any()}
                               for m, g in pm[pm.pair_set == ps].groupby("model")}
        s[f"para_n_{ps}"] = {m: int(g.n.iloc[0]) for m, g in pm[pm.pair_set == ps].groupby("model")}
        rp = R["rates"]["p"].get(ps, {})
        s[f"auc_g_{ps}"] = {m: {c: v["auc_g"] for c, v in rp[m]["targets"].items()} for m in ("lf", "g2") if m in rp}
        s[f"n_para_families_{ps}"] = {m: rp[m]["n_fam"] for m in ("lf", "g2") if m in rp}
        s[f"hidden_{ps}"] = {m: {k: (v["value"] if isinstance(v, dict) else v) for k, v in rp[m]["hidden"].items()}
                             for m in ("lf", "g2") if m in rp}
    return s


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def code_sha():
    return {"analysis/deploy/eval_tf.py": C.sha256(Path(__file__).resolve()),
            "analysis/d1_pair/hammett_eval.py": C.sha256(Path(HE.__file__).resolve()),
            "analysis/d1_pair/d1_eval.py": C.sha256(Path(E9.__file__).resolve()),
            "analysis/geo2/eval_geo2.py": C.sha256(Path(EG.__file__).resolve())}


def write_tables(R, out_dir):
    C.write_csv(out_dir / "E_metrics.csv", R["M"])
    C.write_csv(out_dir / "E_compare.csv", R["C"])
    C.write_csv(out_dir / "E_bends.csv", R["B"])
    C.write_json(out_dir / "E_bend_rates.json", _clean(dict(rules=RATE_RULES, rates=R["rates"])))
    C.write_json(out_dir / "E_ad.json", _clean(dict(R["ad"], g2_reference=R["ad_g2"]["thresholds"])))


def cmd_run():
    if not K_PRED.is_file():
        C.not_ready(f"{K_PRED} missing (train_tf.py collect)")
    K = pd.read_parquet(K_PRED)
    if set(K.rung) != {C.RUNG} or set(K.arm) != {"S"} or set(K.model) != {"TF"}:
        C.stop("K_predictions.parquet must hold rung lf / arm S / model TF only")
    R = evaluate(K, ladder_g2(), gc_threshold())
    write_tables(R, C.RES)
    kr = C.RES / "K_report.json"
    rep = dict(G_C=R["gc"], summary=summary(R), n_pairs=R["n"], ad=R["ad"],
               inputs_sha256=dict(K_predictions=C.sha256(K_PRED), L_predictions=C.sha256(L_PRED),
                                  S_folds=C.sha256(S_FOLDS), K_report=C.sha256(kr) if kr.is_file() else None),
               outputs_sha256={f: C.sha256(C.RES / f) for f in ("E_metrics.csv", "E_compare.csv", "E_bends.csv",
                                                                 "E_bend_rates.json", "E_ad.json")},
               code_sha256=code_sha(), labels_max_abs_diff_common=R["dy"])
    C.write_json(C.RES / "E_report.json", _clean(rep))
    print(json.dumps(_clean(dict(G_C=dict(passed=R["gc"]["pass"], failed=R["gc"]["failed"]), n_pairs=R["n"])), indent=1))
    return 0


# ================================================================ selftest
def _close3(v, spec):
    return v is not None and np.isfinite(v) and abs(v - spec) <= 0.0005 + 1e-12


def cmd_selftest():
    g2 = ladder_g2()
    lf = g2.copy()
    lf["rung"] = "lf"
    R = evaluate(lf, g2)
    write_tables(R, SELF_DIR)
    checks, detail = {}, {}
    M = R["M"]
    own = M[(M.model == "g2") & (M.pair_set == "own") & (M.eval_set == "P_main") & (M.pos == "p")].set_index("target")
    detail["para_mae"] = {t: dict(value=float(own.at[t, "mae"]), spec=SPEC_MAE[t], n=int(own.at[t, "n"]))
                          for t in SPEC_MAE}
    checks["spec0-4 para MAE (3 dp) and n = 1,658"] = all(_close3(v["value"], v["spec"]) and v["n"] == SPEC_N["para_pairs"]
                                                          for v in detail["para_mae"].values())
    E2 = pd.read_csv(E2_METRICS)
    E2 = E2[(E2.rung == "g2") & (E2.arm == "S") & (E2.model == "TF") & (E2.pair_set == "own") & (E2.set == "P_main")
            & (E2.pos == "p")].set_index("target")
    dev = {t: max(abs(own.at[t, k] - E2.at[t, k]) for k in ("mae", "mae_ci_lo", "mae_ci_hi")) for t in TARGETS}
    detail["E2_para_ci_max_abs_diff"] = dev
    detail["E2_n_cores"] = {t: [int(own.at[t, "n_cores"]), int(E2.at[t, "n_cores"])] for t in TARGETS}
    checks["E2 para MAE + CI reproduced (<= 1e-9)"] = all(v <= 1e-9 for v in dev.values()) and \
        all(a == b for a, b in detail["E2_n_cores"].values())
    rp = R["rates"]["p"]["own"]["g2"]
    detail["auc_g"] = {c: dict(value=rp["targets"][c]["auc_g"], ci=rp["targets"][c]["auc_g_ci"],
                               spec=SPEC_AUC.get(c)) for c in CH}
    detail["n_para_families"] = rp["n_fam"]
    checks["spec0-4 continuous g AUC (3 dp) on 89 families"] = rp["n_fam"] == SPEC_N["para_families"] and \
        all(_close3(detail["auc_g"][c]["value"], SPEC_AUC[c]) for c in SPEC_AUC)
    hd = rp["hidden"]
    detail["hidden"] = {k: dict(value=hd[k]["value"], k=hd[k]["k"], n=hd[k]["n"], spec=SPEC_HIDDEN[k])
                        for k in SPEC_HIDDEN}
    checks["W 0-1 hidden-bend rates (2 dp)"] = all(abs(round(v["value"], 2) - v["spec"]) < 1e-9
                                                    for v in detail["hidden"].values())
    # bends loop vs LADDER E2_bends on the P_main families (FU2 bends_tab rule)
    A = R["A"]
    Bpm = bends(A, "g2", sorted(A[A.model == "g2"].pair_id.unique()), "p", pmain_only=True)
    E2b = pd.read_csv(E2_BENDS)
    E2b = E2b[(E2b.rung == "g2") & (E2b.arm == "S") & (E2b.model == "TF") & (E2b.pair_set == "own")]
    m = Bpm.merge(E2b, on=["fam", "target"], suffixes=("", "_e2"), how="outer", indicator=True)
    same = bool((m._merge == "both").all())
    calls_eq = same and all((m[c] == m[f"{c}_e2"]).all() for c in ("bent_dft", "bent_pred", "V_dft", "V_pred",
                                                                   "n_points"))
    gmax = float(np.nanmax(np.abs(m.gain_pred - m.gain_pred_e2))) if same else np.inf
    detail["E2_bends"] = dict(n_rows=int(len(m)), same_rows=same, calls_equal=bool(calls_eq), gain_max_abs_diff=gmax,
                              n_families=int(Bpm.fam.nunique()))
    checks["bend calls = LADDER E2_bends on the P_main families"] = bool(calls_eq and gmax <= 1e-9)
    # family_series_pos('p'-rule written out) = hammett_eval.family_series
    d = A[(A.model == "g2") & (A.ref_kind == "reference") & (A.pos == "p") & A.sigma.notna() & (A.target == "oi")]
    eq = True
    for f in sorted(d.fam.unique()):
        P = d[["fam", "pos", "sigma", "Xb"]].assign(v=d.y.to_numpy())
        a = HE.family_series(P, f, "v", MIN_X)
        t = P[(P.fam == f) & (P.pos == "p") & P.sigma.notna()].drop_duplicates("Xb")
        b = None if len(t) + 1 < MIN_X else (np.r_[t.sigma.to_numpy(float), 0.0], np.r_[-t.v.to_numpy(float) / HE.K, 0.0])
        eq &= (a is None and b is None) or (a is not None and b is not None and np.array_equal(a[0], b[0])
                                            and np.array_equal(a[1], b[1]))
    checks["family_series_pos rule = hammett_eval.family_series"] = bool(eq)
    # G-C machinery on identical inputs
    Cm = R["C"]
    cz = Cm[Cm.pos == "p"]
    checks["identical inputs: d = 0, CI = [0, 0], r = 0, G-C pass"] = bool(
        (cz.d.abs() == 0).all() and (cz.d_ci_lo == 0).all() and (cz.d_ci_hi == 0).all() and (cz.rel.abs() == 0).all()
        and R["gc"]["pass"])
    diffs = R["rates"]["p"]["common"].get("auc_diff_lf_minus_g2", {})
    checks["identical inputs: paired AUC difference 0"] = all(abs(v["d"]) == 0 and v["ci"] == [0.0, 0.0]
                                                              for v in diffs.values() if v["d"] == v["d"])
    rep = dict(all_pass=bool(all(checks.values())), checks=checks, detail=detail, gc=R["gc"], n_pairs=R["n"],
               ad_g2=R["ad_g2"], tables=str(SELF_DIR), code_sha256=code_sha(),
               inputs_sha256=dict(L_predictions=C.sha256(L_PRED), S_folds=C.sha256(S_FOLDS),
                                  E2_metrics=C.sha256(E2_METRICS), E2_bends=C.sha256(E2_BENDS)))
    C.write_json(C.RES / "E_selftest.json", _clean(rep))
    print(json.dumps(_clean(dict(all_pass=rep["all_pass"], checks=checks,
                                 para_mae={t: round(v["value"], 4) for t, v in detail["para_mae"].items()},
                                 auc={c: (None if not np.isfinite(v["value"]) else round(v["value"], 4))
                                      for c, v in detail["auc_g"].items()}, n_fam=detail["n_para_families"],
                                 hidden={k: round(v["value"], 4) for k, v in detail["hidden"].items()})), indent=1))
    return 0 if rep["all_pass"] else 3


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["selftest"]:
        sys.exit(cmd_selftest())
    if a == ["run"]:
        sys.exit(cmd_run())
    print(__doc__)
    sys.exit(2)
