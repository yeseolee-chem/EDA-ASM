#!/usr/bin/env python3
r"""eval_w.py — DEPLOY-MODEL-W Phase E (evaluation) and D (decision) (docs/specs/DEPLOY_MODEL_W.md 7, 10, 11).
CPU, env_d1fu2 or espley (pandas, scipy); analysis/deploy/jobs/cpu_py.slurm submitted from analysis/deploy/w.
Exit 0 written, 2 inputs missing, 3 STOP.

  python eval_w.py run      results/K_predictions.parquet (lf, DEPLOY_MODEL K), results/w/K_predictions_w.parquet
                            (lfV, lfW), $DEPW/within_pred/*.parquet (Phase P) -> results/w/E_H.csv, E_ratio.csv,
                            E_mae.csv, E_bends_w.csv, E_bend_rates_w.json, E_ad_w.json, E_report_w.json;
                            $DEPW/eval/E_graph_pred.parquet (graph readout per pair, I2-2)
  python eval_w.py decide   results/w/E_report_w.json -> results/w/D_decision.json

Combinations: lf-direct, lf-graph, lfV-direct, lfV-graph (interpretation only), lfW-direct, lfW-graph.
Graph readout of pair (a, ref) in group G (m = |G|, ref included): (1/m) sum_{j in G} [g(a, j) - g(ref, j)] from the
Phase P unordered pair predictions (w_common.graph_values), per seed, then the mean / SD over the 5 seeds (= yhat /
yhat_seed_sd of the direct predictions).
Rules (PREREG_W, fixed before any lfV / lfW result)
  families  para families of the lf test predictions: hammett_eval.family_series rule (reference test pairs with sigma,
            pos p, one point per Xb, >= 12 points incl. H) = eval_tf.bends; the three models / six combinations must have
            the same families and points (else STOP); DFT calls from the same points (calls identical across combinations).
  H         H = mean_{t in PRIMARY} mean_f [ r_ft / s_t ], r_ft = RMS of the least-squares line residuals of (sigma, e = yhat
            - y [kcal/mol]) + (0, 0) (w_common.line_rms), s_t = SD (ddof 0) of the DFT y of every evaluation point (fixed,
            model independent). Core bootstrap 2,000 (seed 20261019): eval_geo2.boot_diff resampling of the cores
            (weights w_c), H* = mean_t [sum_f w_c(f) r_ft / sum_f w_c(f)] / s_t; ratios on the same resamples.
  A-1       R = H_lfW-direct / H_lf-direct, 95 % upper bound < 1.00
  A-2       each of the 9 primary targets: (MAE_lfW - MAE_lf) / MAE_lf on the para P_main pairs, 95 % upper bound <= +5 %
            (eval_tf.boot_rel = eval_geo2.boot_diff core resamples; = DEPLOY_MODEL G-C)
  grade     A-1 fails -> 'no improvement'; else R <= 0.70 -> 'goal reached', else 'partial improvement'
  secondary for lf and lfW: R_g = H_graph / H_direct upper bound < 1.00 and the A-2 non-inferiority graph vs direct
            -> readout graph, else direct (applied to the adopted model)
  descriptive  hidden-bend sensitivity / FPR / PPV (Clopper-Pearson + family bootstrap CI, paired difference vs
            lf-direct on the same family resamples), per-channel sensitivity / FPR, continuous g AUC (eval_tf.bend_rates),
            H per target, R lfW / lfV and lfV / lf, the expected PPV of the spec 0-1 simulation table at R.
Reused code (import): eval_tf.frame, check_y, s_folds, bends, hidden, hidden_rates, bend_rates, boot_index, boot_rel,
auc, pct_ci, rate, ad_thresholds, family_series_pos, MIN_X; eval_geo2.boot_diff; hammett_eval.family_series.
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
import w_common as W  # noqa: E402

C = W.C
import eval_tf as ET  # noqa: E402

HE, EG = ET.HE, ET.EG
MODELS = ("lf", "lfV", "lfW")
COMBOS = ("lf-direct", "lf-graph", "lfV-direct", "lfV-graph", "lfW-direct", "lfW-graph")
DECISION_COMBOS = ("lf-direct", "lf-graph", "lfW-direct", "lfW-graph")
PRIMARY = list(W.PRIMARY)
REL_MAX = 0.05
GOAL_R = 0.70
DIRECT_TOL_WARN, DIRECT_TOL_STOP = 1e-4, 1e-3
# spec 0-1 simulation (harmful error shrunk, shape fixed): R -> PPV (R = 1: the LADDER g2 . S . TF value)
SIM_PPV = ((0.20, 0.93), (0.30, 0.81), (0.70, 0.73), (1.00, 0.57))
K_LF = C.RES / "K_predictions.parquet"
K_W = W.RESW / "K_predictions_w.parquet"
OUT = ("E_H.csv", "E_ratio.csv", "E_mae.csv", "E_bends_w.csv", "E_bend_rates_w.json", "E_ad_w.json")


# ================================================================ graph readout from Phase P
def graph_predictions(Wp, T, fold):
    """Phase P rows (gid, a_rxn, b_rxn, seed, target, g) of one (model, fold) + the fold's test groups T -> graph readout
    per test pair: pair_id, fold, target, yhat, yhat_seed_sd, n_seeds, m. Every node pair of every group must be present
    once per (seed, target) (else STOP)."""
    T = T.sort_values(["gid", "a_rxn"]).reset_index(drop=True)
    nodes = []
    for g, q in T.groupby("gid", sort=True):
        ref = int(q.b_rxn.iloc[0])
        for r in sorted(int(x) for x in q.a_rxn) + [ref]:
            nodes.append((int(g), r, ref))
    N = pd.DataFrame(nodes, columns=["gid", "rxn", "ref"])
    N["node"] = np.arange(len(N))
    m = N.groupby("gid").size()
    gidx = {g: i for i, g in enumerate(m.index)}
    node_group = N.gid.map(gidx).to_numpy()
    ref_node = N[N.rxn == N.ref].set_index("gid").node.reindex(m.index).to_numpy()
    key = pd.MultiIndex.from_frame(N[["gid", "rxn"]])
    Wp = Wp.copy()
    Wp["seed"] = Wp.seed.astype(int)
    wide = Wp.pivot_table(index=["gid", "a_rxn", "b_rxn"], columns=["seed", "target"], values="g", aggfunc="first")
    cnt = Wp.groupby(["gid", "a_rxn", "b_rxn", "seed", "target"]).size()
    if (cnt != 1).any():
        W.stop(f"fold {fold}: duplicated Phase P rows")
    expect = int((m * (m - 1) // 2).sum())
    if len(wide) != expect or wide.isna().any().any():
        W.stop(f"fold {fold}: Phase P pairs incomplete ({len(wide)} of {expect}, NaN {int(wide.isna().sum().sum())})")
    idx = wide.index.to_frame(index=False)
    u = key.get_indexer(pd.MultiIndex.from_frame(idx[["gid", "a_rxn"]].rename(columns={"a_rxn": "rxn"})))
    v = key.get_indexer(pd.MultiIndex.from_frame(idx[["gid", "b_rxn"]].rename(columns={"b_rxn": "rxn"})))
    if (u < 0).any() or (v < 0).any():
        W.stop(f"fold {fold}: Phase P reactions outside the test groups")
    seeds = sorted(set(wide.columns.get_level_values(0)))
    cols = [(s, t) for s in seeds for t in W.TARGETS]
    G = wide[cols].to_numpy(float)
    gh = W.graph_values(u, v, G, node_group, m.to_numpy(), ref_node).reshape(len(N), len(seeds), len(W.TARGETS))
    Tn = T.merge(N.rename(columns={"rxn": "a_rxn"})[["gid", "a_rxn", "node"]], on=["gid", "a_rxn"], how="left")
    vals = gh[Tn.node.to_numpy()]                                         # [n_pairs, seeds, 10]
    mm = m.reindex(Tn.gid).to_numpy()
    rows = [pd.DataFrame(dict(pair_id=Tn.pair_id.to_numpy(), fold=int(fold), target=t, yhat=vals[:, :, j].mean(1),
                              yhat_seed_sd=vals[:, :, j].std(1), n_seeds=len(seeds), m=mm))
            for j, t in enumerate(W.TARGETS)]
    direct = Wp.merge(T[["gid", "a_rxn", "b_rxn"]], on=["gid", "a_rxn", "b_rxn"])        # (a, ref) rows
    dmean = direct.groupby(["gid", "a_rxn", "b_rxn", "target"]).g.mean().rename("g_direct").reset_index()
    dmean = dmean.merge(T[["pair_id", "gid", "a_rxn", "b_rxn"]], on=["gid", "a_rxn", "b_rxn"])[["pair_id", "target",
                                                                                                  "g_direct"]]
    return pd.concat(rows, ignore_index=True).merge(dmean, on=["pair_id", "target"], how="left")


def load_graph(m, k, source="lf"):
    import within as WI
    p = W.PRED_W / f"D__{m}__S__TF__f{k}.parquet"
    js = p.with_suffix(".json")
    if not (p.is_file() and js.is_file()):
        W.not_ready(f"{p} missing (within_predict.py run)")
    if W.read_json(js).get("parquet_sha256") != W.sha256(p):
        W.stop(f"{p} changed after its json")
    return graph_predictions(pd.read_parquet(p), WI.load_test_groups(k, source=source), k), W.sha256(p)


# ================================================================ statistics
def boot_weights(cores, n_boot=W.N_BOOT, seed=W.SEED):
    """the core resample weight matrix of eval_geo2.boot_diff (same rng calls): (unique cores, w [n_boot, k])."""
    uc = np.unique(cores)
    rng = np.random.default_rng(seed)
    k = len(uc)
    return uc, np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)])


def family_points(A, combo, pos="p"):
    """evaluation points per family (eval_tf.bends selection: reference test pairs with sigma at pos, one per Xb in
    pair order, >= MIN_FAM_EVAL incl. H) -> {fam: DataFrame pair_id, core, sigma, y_<t>, yhat_<t>}."""
    d = A[(A.model == combo) & (A.ref_kind == "reference") & (A.pos == pos) & A.sigma.notna()]
    y = d.pivot_table(index="pair_id", columns="target", values="y", aggfunc="first")
    yh = d.pivot_table(index="pair_id", columns="target", values="yhat", aggfunc="first")
    at = d.drop_duplicates("pair_id").set_index("pair_id")[["fam", "core", "sigma", "Xb"]]
    first = d[d.target == W.TARGETS[0]]                       # pair order of the frame (= family_series row order)
    out = {}
    for f, q in first.groupby("fam", sort=True):
        t = q.drop_duplicates("Xb")
        if len(t) + 1 < W.MIN_FAM_EVAL:
            continue
        ids = t.pair_id.to_numpy()
        fr = at.loc[ids, ["core", "sigma", "Xb"]].copy()
        for tt in W.TARGETS:
            fr[f"y_{tt}"] = y.loc[ids, tt].to_numpy(float)
            fr[f"yhat_{tt}"] = yh.loc[ids, tt].to_numpy(float)
        out[f] = fr
    return out


def h_table(fams, s_t):
    """r_ft for every family x primary target -> DataFrame fam, core, n_points, target, r."""
    rows = []
    for f, fr in fams.items():
        cores = fr.core.unique()
        if len(cores) != 1:
            W.stop(f"family {f} spans {len(cores)} cores")
        x = fr.sigma.to_numpy(float)
        for t in PRIMARY:
            e = fr[f"yhat_{t}"].to_numpy(float) - fr[f"y_{t}"].to_numpy(float)
            rows.append(dict(fam=f, core=cores[0], n_points=len(x) + 1, target=t, r=W.line_rms(*W.with_h(x, e)),
                             s_t=s_t[t]))
    return pd.DataFrame(rows)


def h_value(Ht, wmap=None):
    """H = mean_t mean_f r / s_t; with core weights wmap {core: weight} the family mean is weighted."""
    per = {}
    for t, q in Ht.groupby("target"):
        w = np.ones(len(q)) if wmap is None else q.core.map(wmap).to_numpy(float)
        per[t] = float((w * q.r.to_numpy()).sum() / w.sum() / q.s_t.iloc[0])
    return float(np.mean([per[t] for t in PRIMARY])), per


def h_boot(Ht, uc, wb):
    """bootstrap H [n_boot] and per target [n_boot, 9] on the core weights wb [n_boot, k] (uc order)."""
    ci = {c: i for i, c in enumerate(uc)}
    out = np.zeros((len(wb), len(PRIMARY)))
    for j, t in enumerate(PRIMARY):
        q = Ht[Ht.target == t]
        col = q.core.map(ci).to_numpy()
        Wf = wb[:, col].astype(float)                               # [n_boot, n_fam]
        out[:, j] = (Wf @ q.r.to_numpy()) / Wf.sum(1) / q.s_t.iloc[0]
    return out.mean(1), out


def interp_ppv(R):
    xs, ys = zip(*SIM_PPV)
    return float(np.interp(R, xs, ys)) if R is not None and np.isfinite(R) else None


# ================================================================ evaluation
def build_frames(source="lf"):
    """the six combinations as eval_tf frames + the graph tables + input sha256."""
    if not K_LF.is_file():
        W.not_ready(f"{K_LF} missing (DEPLOY_MODEL train_tf.py collect)")
    if not K_W.is_file():
        W.not_ready(f"{K_W} missing (train_w.py collect)")
    K = pd.read_parquet(K_LF)
    KW = pd.read_parquet(K_W)
    if set(K.rung) != {"lf"} or set(KW.rung) != {"lfW", "lfV"}:
        W.stop("K predictions: rung lf / lfW + lfV expected")
    preds = {"lf": K, "lfV": KW[KW.rung == "lfV"], "lfW": KW[KW.rung == "lfW"]}
    S = ET.s_folds()
    frames, graphs, shas, dchk = [], [], {}, {}
    for m in MODELS:
        base = preds[m][["pair_id", "fold", "target", "y", "yhat", "yhat_seed_sd", "pos", "ref_kind"]].copy()
        base["pair_id"] = base.pair_id.astype(str)
        frames.append(ET.frame(base, f"{m}-direct", S))
        gs = []
        for k in range(W.NF):
            g, sha = load_graph(m, k, source)
            shas[f"within_pred/D__{m}__S__TF__f{k}.parquet"] = sha
            gs.append(g.assign(model=m))
        G = pd.concat(gs, ignore_index=True)
        mg = base.merge(G[["pair_id", "target", "yhat", "yhat_seed_sd", "g_direct", "m"]], on=["pair_id", "target"],
                        how="left", suffixes=("_d", ""))
        if mg.yhat.isna().any():
            W.stop(f"{m}: {int(mg.yhat.isna().sum())} test rows without a graph value (spec 15 E: point sets differ)")
        dd = np.abs(mg.g_direct - mg.yhat_d).to_numpy(float)
        dchk[m] = dict(max_abs=float(np.nanmax(dd)), n=int(len(dd)), n_missing=int(np.isnan(dd).sum()))
        if not (dchk[m]["max_abs"] <= DIRECT_TOL_STOP) or dchk[m]["n_missing"]:
            W.stop(f"{m}: Phase P (a, ref) predictions differ from the K predictions by {dchk[m]['max_abs']:.2e} "
                   f"(missing {dchk[m]['n_missing']}): other model files?")
        gframe = mg.drop(columns=["yhat_d", "yhat_seed_sd_d", "g_direct", "m"])
        frames.append(ET.frame(gframe, f"{m}-graph", S))
        graphs.append(mg.assign(model=m))
    A = pd.concat(frames, ignore_index=True)
    for c in COMBOS[1:]:
        ET.check_y(A[A.model == "lf-direct"], A[A.model == c])
    shas.update(K_predictions=W.sha256(K_LF), K_predictions_w=W.sha256(K_W), S_folds=W.sha256(ET.S_FOLDS))
    return A, pd.concat(graphs, ignore_index=True), shas, dchk


def evaluate(A):
    out = {}
    own = {c: sorted(A[A.model == c].pair_id.unique()) for c in COMBOS}
    if len({tuple(v) for v in own.values()}) != 1:
        W.stop("the combinations have different test pair sets")
    ids = own["lf-direct"]
    # ---------------- families / points (spec 7 'evaluation families (fixed)', 15 E)
    fams = {c: family_points(A, c, "p") for c in COMBOS}
    ref = fams["lf-direct"]
    for c, fc in fams.items():
        if sorted(fc) != sorted(ref) or any(list(fc[f].index) != list(ref[f].index) for f in ref):
            W.stop(f"{c}: evaluation families / points differ from lf-direct (spec 15 E)")
    pts = pd.concat([fr.assign(fam=f) for f, fr in ref.items()])
    s_t = {t: float(pts[f"y_{t}"].std(ddof=0)) for t in PRIMARY}
    Ht = {c: h_table(fams[c], s_t) for c in COMBOS}
    cores = sorted(pts.core.unique())
    uc, wb = boot_weights(np.array(cores))
    Hv, Hb, Hper, Hbper = {}, {}, {}, {}
    for c in COMBOS:
        Hv[c], Hper[c] = h_value(Ht[c])
        Hb[c], Hbper[c] = h_boot(Ht[c], uc, wb)
    hrows = []
    for c in COMBOS:
        for _, r in Ht[c].iterrows():
            hrows.append(dict(combo=c, pos="p", level="family", target=r.target, fam=r.fam, core=r.core,
                              n_points=int(r.n_points), r=float(r.r), s_t=float(r.s_t)))
        for j, t in enumerate(PRIMARY):
            lo, hi = np.percentile(Hbper[c][:, j], [2.5, 97.5])
            hrows.append(dict(combo=c, pos="p", level="target", target=t, H=Hper[c][t], ci_lo=float(lo), ci_hi=float(hi),
                              s_t=s_t[t], n_fam=len(ref)))
        lo, hi = np.percentile(Hb[c], [2.5, 97.5])
        hrows.append(dict(combo=c, pos="p", level="H", target="mean9", H=Hv[c], ci_lo=float(lo), ci_hi=float(hi),
                          n_fam=len(ref), n_cores=len(uc)))
    out["E_H"] = pd.DataFrame(hrows)
    ratios = [("R", "lfW-direct", "lf-direct"), ("R_g_lf", "lf-graph", "lf-direct"),
              ("R_g_lfW", "lfW-graph", "lfW-direct"), ("lfW_over_lfV", "lfW-direct", "lfV-direct"),
              ("lfV_over_lf", "lfV-direct", "lf-direct"), ("R_g_lfV", "lfV-graph", "lfV-direct"),
              ("lfW_graph_over_lf", "lfW-graph", "lf-direct")]
    rr, RR = [], {}
    for name, a, b in ratios:
        v = Hv[a] / Hv[b]
        bs = Hb[a] / Hb[b]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        RR[name] = dict(num=a, den=b, R=float(v), ci_lo=float(lo), ci_hi=float(hi))
        rr.append(dict(name=name, num=a, den=b, R=float(v), ci_lo=float(lo), ci_hi=float(hi), H_num=Hv[a], H_den=Hv[b],
                       n_boot=W.N_BOOT, n_fam=len(ref), n_cores=len(uc)))
    out["E_ratio"] = pd.DataFrame(rr)
    # ---------------- MAE (A-2 and the graph non-inferiority)
    pm = A[(A.ref_kind == "reference") & A.in_P_main.astype(bool) & (A.pos == "p")]
    mrows = []
    comps = [(c, "lf-direct") for c in COMBOS if c != "lf-direct"] + [("lf-graph", "lf-direct"),
                                                                      ("lfW-graph", "lfW-direct"),
                                                                      ("lfV-graph", "lfV-direct")]
    seen = set()
    for c in COMBOS:
        for t in W.TARGETS:
            q = pm[(pm.model == c) & (pm.target == t)].set_index("pair_id").sort_index()
            e = (q.yhat - q.y).to_numpy(float)
            _, lo, hi, _ = EG.boot_diff(e, np.zeros(len(e)), q.core.to_numpy())
            mrows.append(dict(combo=c, base=None, target=t, primary=t in PRIMARY, n=int(len(q)),
                              n_cores=int(q.core.nunique()), mae=float(np.abs(e).mean()), mae_ci_lo=lo, mae_ci_hi=hi))
    for c, b in comps:
        if (c, b) in seen:
            continue
        seen.add((c, b))
        for t in W.TARGETS:
            qx = pm[(pm.model == c) & (pm.target == t)].set_index("pair_id").sort_index()
            qb = pm[(pm.model == b) & (pm.target == t)].set_index("pair_id").reindex(qx.index)
            ex, eb = (qx.yhat - qx.y).to_numpy(float), (qb.yhat - qb.y).to_numpy(float)
            cc = qx.core.to_numpy()
            d, lo, hi, p = EG.boot_diff(ex, eb, cc)
            r, rlo, rhi, (dlo, dhi) = ET.boot_rel(ex, eb, cc)
            if abs(dlo - lo) > 1e-12 or abs(dhi - hi) > 1e-12:
                W.stop(f"boot_rel resamples differ from eval_geo2.boot_diff ({c} vs {b}, {t})")
            mrows.append(dict(combo=c, base=b, target=t, primary=t in PRIMARY, n=int(len(qx)), n_cores=int(len(set(cc))),
                              mae=float(np.abs(ex).mean()), mae_base=float(np.abs(eb).mean()), d=d, d_ci_lo=lo,
                              d_ci_hi=hi, p=p, rel=r, rel_ci_lo=rlo, rel_ci_hi=rhi,
                              pass_rel=(bool(rhi <= REL_MAX) if t in PRIMARY else None)))
    out["E_mae"] = M = pd.DataFrame(mrows)

    def ni(c, b):
        q = M[(M.combo == c) & (M.base == b) & M.primary]
        per = {r.target: dict(rel=r.rel, rel_ci_hi=r.rel_ci_hi, pass_=bool(r.rel_ci_hi <= REL_MAX)) for r in q.itertuples()}
        return dict(pass_=bool(len(per) == len(PRIMARY) and all(v["pass_"] for v in per.values())), per_target=per,
                    failed=[t for t, v in per.items() if not v["pass_"]], n_pairs=int(q.n.iloc[0]) if len(q) else 0)
    # ---------------- bends (descriptive) + DFT calls identical
    Bs, rates = [], {}
    nf = None
    for c in COMBOS:
        B = ET.bends(A, c, ids, "p")
        Bs.append(B)
        if nf is None:
            nf = B.fam.nunique()
    Ball = pd.concat(Bs, ignore_index=True)
    dft = Ball.pivot_table(index=["fam", "target"], columns="model", values="bent_dft", aggfunc="first")
    if not (dft.nunique(axis=1) == 1).all():
        W.stop("DFT bend calls differ between the combinations")
    if sorted(Ball.fam.unique()) != sorted(ref):
        W.stop("eval_tf.bends families differ from the H families")
    idx = ET.boot_index(nf)
    hid = {}
    for c in COMBOS:
        B = Ball[Ball.model == c]
        r, famlist = ET.bend_rates(B, idx)
        Hd = ET.hidden(B).reindex(famlist)
        hd, hp = Hd.hidden_dft.to_numpy(bool), Hd.hidden_pred.to_numpy(bool)
        hid[c] = (hd, hp)
        hb = hidden_boot(hd, hp, idx)
        rates[c] = dict(n_fam=len(famlist), targets={t: {k: v for k, v in r[t].items() if k != "auc_boot"} for t in r},
                        hidden=ET.hidden_rates(Hd), hidden_boot_ci={k: list(ET.pct_ci(v)) for k, v in hb.items()},
                        _auc=r, _hb=hb)
    for c in COMBOS:
        if c == "lf-direct":
            continue
        diff = {}
        for k in ("sens", "fpr", "ppv"):
            a, b = rates[c]["_hb"][k], rates["lf-direct"]["_hb"][k]
            diff[k] = dict(d=_val(rates[c]["hidden"][k]) - _val(rates["lf-direct"]["hidden"][k]),
                           ci=list(ET.pct_ci(a - b)))
        adiff = {t: dict(d=rates[c]["targets"][t]["auc_g"] - rates["lf-direct"]["targets"][t]["auc_g"],
                         ci=list(ET.pct_ci(rates[c]["_auc"][t]["auc_boot"] - rates["lf-direct"]["_auc"][t]["auc_boot"])))
                 for t in rates[c]["targets"]}
        rates[c]["hidden_diff_vs_lf_direct"] = diff
        rates[c]["auc_diff_vs_lf_direct"] = adiff
    for c in COMBOS:
        rates[c].pop("_auc")
        rates[c].pop("_hb")
    out["E_bends_w"] = Ball
    # ---------------- verdicts
    A1 = dict(RR["R"], pass_=bool(RR["R"]["ci_hi"] < 1.0), rule="95 % upper bound of R = H_lfW / H_lf < 1.00")
    A2 = dict(ni("lfW-direct", "lf-direct"), rule=f"each primary target: 95 % upper bound of (MAE_lfW - MAE_lf) / MAE_lf "
                                                  f"<= +{REL_MAX * 100:g} % (para P_main)")
    grade = ("no improvement" if not A1["pass_"] else ("goal reached" if RR["R"]["R"] <= GOAL_R else
                                                         "partial improvement"))
    sec = {}
    for m in ("lf", "lfW", "lfV"):
        rg = RR[f"R_g_{m}"]
        nn = ni(f"{m}-graph", f"{m}-direct")
        sec[m] = dict(R_g=rg, non_inferiority=nn, pass_=bool(rg["ci_hi"] < 1.0 and nn["pass_"]),
                      readout="graph" if (rg["ci_hi"] < 1.0 and nn["pass_"]) else "direct")
    out["report"] = dict(A1=A1, A2=A2, adopt_lfW=bool(A1["pass_"] and A2["pass_"]), grade=grade, secondary=sec,
                         H={c: Hv[c] for c in COMBOS}, H_per_target={c: Hper[c] for c in COMBOS}, ratios=RR,
                         s_t=s_t, n_families=len(ref), n_cores=len(uc), n_points=int(len(pts)),
                         expected_ppv_sim={"R": interp_ppv(RR["R"]["R"]), "R_ci_hi": interp_ppv(RR["R"]["ci_hi"]),
                                           "table": SIM_PPV},
                         rates=rates)
    out["ids"] = ids
    return out


def _val(x):
    return x["value"] if isinstance(x, dict) else x


def hidden_boot(hd, hp, idx):
    """family-bootstrap sensitivity / FPR / PPV of the hidden-bend call ([n_boot] each, NaN when undefined)."""
    tp = (hd & hp)[idx].sum(1).astype(float)
    fp = (~hd & hp)[idx].sum(1).astype(float)
    fn = (hd & ~hp)[idx].sum(1).astype(float)
    tn = (~hd & ~hp)[idx].sum(1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(sens=tp / (tp + fn), fpr=fp / (fp + tn), ppv=tp / (tp + fp))


def summary(R):
    rep = R["report"]
    M = R["E_mae"]
    own = M[M.base.isna()]
    return dict(H=rep["H"], R=rep["ratios"]["R"], R_g_lf=rep["ratios"]["R_g_lf"], R_g_lfW=rep["ratios"]["R_g_lfW"],
                lfW_over_lfV=rep["ratios"]["lfW_over_lfV"], grade=rep["grade"],
                para_mae={c: {r.target: r.mae for r in own[own.combo == c].itertuples()} for c in COMBOS},
                hidden={c: {k: _val(v) for k, v in rep["rates"][c]["hidden"].items() if k in ("sens", "fpr", "ppv", "base")}
                        for c in COMBOS},
                auc_g={c: {t: v["auc_g"] for t, v in rep["rates"][c]["targets"].items()} for c in COMBOS},
                n_families=rep["n_families"])


def code_sha():
    return {**W.code_sha([HERE / "eval_w.py", HERE / "w_common.py"]), **ET.code_sha()}


def cmd_run(source="lf"):
    t0 = time.time()
    A, G, shas, dchk = build_frames(source)
    R = evaluate(A)
    W.write_csv(W.RESW / "E_H.csv", R["E_H"])
    W.write_csv(W.RESW / "E_ratio.csv", R["E_ratio"])
    W.write_csv(W.RESW / "E_mae.csv", R["E_mae"])
    W.write_csv(W.RESW / "E_bends_w.csv", R["E_bends_w"])
    W.write_json(W.RESW / "E_bend_rates_w.json", dict(rules=ET.RATE_RULES, rates=R["report"]["rates"]))
    ad = ET.ad_thresholds(A, "lfW-direct")
    W.write_json(W.RESW / "E_ad_w.json", dict(ad, model="lfW (direct)", lf_reference=ET.ad_thresholds(A, "lf-direct")))
    gp = W.EVAL_W / "E_graph_pred.parquet"
    W.write_parquet(gp, G[["model", "pair_id", "fold", "target", "y", "yhat", "yhat_seed_sd", "g_direct", "m"]])
    rep = {k: v for k, v in R["report"].items() if k != "rates"}
    rep.update(summary=summary(R), direct_consistency=dchk, graph_pred=str(gp), graph_pred_sha256=W.sha256(gp),
               inputs_sha256=dict(shas, prereg_w=(W.sha256(W.RESW / "prereg_w.json")
                                                  if (W.RESW / "prereg_w.json").is_file() else None)),
               outputs_sha256={f: W.sha256(W.RESW / f) for f in OUT}, code_sha256=code_sha(),
               rules=dict(rel_max=REL_MAX, goal_R=GOAL_R, n_boot=W.N_BOOT, seed=W.SEED, min_fam=W.MIN_FAM_EVAL,
                          primary=PRIMARY), seconds=time.time() - t0)
    W.write_json(W.RESW / "E_report_w.json", rep)
    print(json.dumps(W._jsonable(dict(A1=dict(R=rep["A1"]["R"], ci_hi=rep["A1"]["ci_hi"], pass_=rep["A1"]["pass_"]),
                                      A2=dict(pass_=rep["A2"]["pass_"], failed=rep["A2"]["failed"]), grade=rep["grade"],
                                      secondary={m: v["readout"] for m, v in rep["secondary"].items()},
                                      H=rep["H"])), indent=1))
    return 0


def cmd_decide():
    p = W.RESW / "E_report_w.json"
    if not p.is_file():
        W.not_ready(f"{p} missing (eval_w.py run)")
    e = W.read_json(p)
    if e.get("inputs_sha256", {}).get("K_predictions_w") != (W.sha256(K_W) if K_W.is_file() else None):
        W.stop("E_report_w.json was computed on another K_predictions_w.parquet")
    adopt = "lfW" if (e["A1"]["pass_"] and e["A2"]["pass_"]) else "lf"
    sec = e["secondary"][adopt]
    readout = sec["readout"]
    d = dict(phase="D", adopted=adopt, readout=readout, grade=e["grade"],
             A1=dict(R=e["A1"]["R"], ci=[e["A1"]["ci_lo"], e["A1"]["ci_hi"]], pass_=e["A1"]["pass_"]),
             A2=dict(pass_=e["A2"]["pass_"], failed=e["A2"]["failed"],
                     rel_ci_hi={t: v["rel_ci_hi"] for t, v in e["A2"]["per_target"].items()}),
             secondary=dict(model=adopt, R_g=[sec["R_g"]["R"], sec["R_g"]["ci_lo"], sec["R_g"]["ci_hi"]],
                            non_inferiority_pass=sec["non_inferiority"]["pass_"], readout=readout),
             w2_needed=adopt == "lfW",
             next=("W2 (train_w.py final) -> I2" if adopt == "lfW" else
                   ("no W2; I2 adds the graph readout to the DEPLOY_MODEL lf final model" if readout == "graph" else
                    "no W2; I2 checks only; deployed model = DEPLOY_MODEL lf final, direct readout")),
             deploy=dict(model_dir=str(W.MODEL_W if adopt == "lfW" else C.MODEL_DIR), readout=readout,
                         group_id_required=readout == "graph", max_group_size=W.GRAPH_MAX),
             e_report_sha256=W.sha256(p), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    W.write_json(W.RESW / "D_decision.json", d)
    print(json.dumps(W._jsonable(d), indent=1))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    src = "lf"
    if "--source" in a:
        j = a.index("--source")
        src = a[j + 1]
        a = a[:j] + a[j + 2:]
    if a == ["run"]:
        sys.exit(cmd_run(src))
    if a == ["decide"]:
        sys.exit(cmd_decide())
    print(__doc__)
    sys.exit(2)
