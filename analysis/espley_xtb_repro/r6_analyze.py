#!/usr/bin/env python3
"""r6_analyze.py — rev 6 Phase B / A outputs (spec §7, §8-2, §8-3, §9) from the unit cache ($R6_SCRATCH/units/).

  python r6_analyze.py B        # B_per_seed, B_summary, B_representative.json, B_primary_tests, B_secondary_tests,
                                # B_bias_check, B_predictions.parquet, B.json
  python r6_analyze.py A        # A_per_seed, A_summary, A_representative.json, A_predictions.parquet,
                                # A_appendix_per_seed / _summary / _predictions

Representative model (§3-3, pre-registered): per arm and target, the model with the lowest validation MAE averaged
over the 5 seeds (our pipeline: inner-CV best_score_; Espley protocol: their GridSearchCV best_score_ = hps.pkl
train_mae, constant over seeds; NNs: the 5-fold CV of the tuned NN). Test scores never choose anything.
Primary test (§7): per target d_s = MAE_s(ESP-C rep) - MAE_s(OUR-O rep) over the 5 seeds, Nadeau-Bengio corrected t
(rev5_common.nadeau_bengio, J = 5, n_test / n_train = seed means), two-sided p, 95 % CI; Holm over the 4 targets;
seeds won by OUR-O (d_s > 0). Everything else is descriptive (no correction).
All metrics are computed here from the stored per-reaction predictions (B_predictions / A_predictions).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rev6_common as R6  # noqa: E402

UNITS = R6.SCRATCH / "units"
OUT = R6.RES6


def load_units(groups):
    recs, preds = [], []
    for g in groups:
        for f in sorted((UNITS / g).glob("*.json")):
            r = json.loads(f.read_text())
            p = f.with_suffix(".parquet")
            if not p.is_file():
                sys.exit(f"STOP: {p} missing")
            recs.append(r)
            preds.append(pd.read_parquet(p))
    if not recs:
        sys.exit(f"no units in {groups}")
    return pd.DataFrame(recs), pd.concat(preds, ignore_index=True)


def metrics(df):
    y, p = df.y_true.to_numpy(float), df.y_pred.to_numpy(float)
    e = np.abs(p - y)
    ss = float(np.sum((y - y.mean()) ** 2))
    return pd.Series(dict(mae=float(e.mean()), nmae=R6.nmae(y, p), r2=1 - float(np.sum((p - y) ** 2)) / ss if ss else np.nan,
                          within1=float(np.mean(e <= R6.WITHIN_KCAL)), n_test=len(df)))


def per_seed(rec, pred, key=("arm", "model", "target", "seed", "rowset")):
    m = pred.groupby(list(key)).apply(metrics).reset_index()
    info = rec[list(key) + ["n_train", "val_mae", "val_source"]].copy()
    info["best_params"] = rec["best_params"].map(lambda d: json.dumps(d, sort_keys=True, default=str))
    info["edge_hits"] = rec["edge_hits"].map(lambda d: int(sum(bool(v) for v in (d or {}).values())))
    return m.merge(info, on=list(key), how="left")


def summary(ps, key=("arm", "model", "target", "rowset")):
    g = ps.groupby(list(key))
    out = g.agg(mae=("mae", "mean"), mae_sd=("mae", "std"), nmae=("nmae", "mean"), r2=("r2", "mean"),
                within1=("within1", "mean"), val_mae=("val_mae", "mean"), n_seeds=("seed", "nunique"),
                n_test=("n_test", "mean"), n_train=("n_train", "mean"), edge_hits=("edge_hits", "sum")).reset_index()
    out["mae_se"] = out.mae_sd / np.sqrt(out.n_seeds)
    return out


def representative(sm, rowset):
    rep = {}
    for (arm, t), g in sm[(sm.rowset == rowset) & sm.model.isin(R6.MODELS)].groupby(["arm", "target"]):
        g = g.sort_values(["val_mae", "model"])
        rep.setdefault(arm, {})[t] = dict(model=str(g.model.iloc[0]),
                                          val_mae_by_model={m: float(v) for m, v in zip(g.model, g.val_mae)})
    return rep


def paired(ps, a, b, t, rowset, ma, mb, label, filt=None, pred=None):
    """d_s = MAE_s(a, ma) - MAE_s(b, mb) on the same rows; NB t-test. filt: rxn_ids to drop (rescoring)."""
    if filt is not None:
        q = pred[(pred.target == t) & (pred.rowset == rowset) & ~pred.rxn_id.isin(filt)]
        A = q[(q.arm == a) & (q.model == ma)].groupby("seed").apply(lambda d: np.mean(np.abs(d.y_pred - d.y_true)))
        B = q[(q.arm == b) & (q.model == mb)].groupby("seed").apply(lambda d: np.mean(np.abs(d.y_pred - d.y_true)))
        ntest = q[(q.arm == b) & (q.model == mb)].groupby("seed").size()
    else:
        x = ps[(ps.target == t) & (ps.rowset == rowset)]
        A = x[(x.arm == a) & (x.model == ma)].set_index("seed").mae
        B = x[(x.arm == b) & (x.model == mb)].set_index("seed").mae
        ntest = x[(x.arm == b) & (x.model == mb)].set_index("seed").n_test
    ntr = ps[(ps.target == t) & (ps.rowset == rowset) & (ps.arm == b) & (ps.model == mb)].set_index("seed").n_train
    seeds = sorted(set(A.index) & set(B.index))
    if len(seeds) != len(R6.SEEDS):
        return dict(comparison=label, target=t, rowset=rowset, a=f"{a}:{ma}", b=f"{b}:{mb}", status="missing_seeds")
    d = np.array([A[s] - B[s] for s in seeds])
    nb = R6.nadeau_bengio(d, float(np.mean([ntest[s] for s in seeds])), float(np.mean([ntr[s] for s in seeds])))
    return dict(comparison=label, target=t, rowset=rowset, a=f"{a}:{ma}", b=f"{b}:{mb}", mae_a=float(A[seeds].mean()),
                mae_b=float(B[seeds].mean()), d_mean=nb["mean"], ci_lo=nb["ci95"][0], ci_hi=nb["ci95"][1], t=nb["t"],
                p=nb["p"], seeds_b_lower=int((d > 0).sum()), d_per_seed=json.dumps([float(v) for v in d]),
                n_test=nb["n_test"], n_train=nb["n_train"], status="ok")


def sigma_channel(predA, repA, predB):
    """Σ-channel arm (§8-3): OUR-O channel representatives -> Int^ = Σ6^ + c_ghost^, ΔE‡^ = d1^ + d2^ + Int^."""
    q = predA[(predA.arm == "OUR-O") & (predA.rowset == "C")]
    rep = repA["OUR-O"]
    parts = {}
    for ch in list(R6.SIX) + ["c_ghost", "d1", "d2"]:
        m = rep[ch]["model"]
        parts[ch] = q[(q.target == ch) & (q.model == m)].set_index(["seed", "rxn_id"]).y_pred
    s6 = sum(parts[c] for c in R6.SIX)
    rows = []
    for t, yhat in (("Int", s6 + parts["c_ghost"]), ("dEa", parts["d1"] + parts["d2"] + s6 + parts["c_ghost"])):
        yt = predB[(predB.arm == "OUR-O") & (predB.target == t) & (predB.rowset == "C")].drop_duplicates(
            ["seed", "rxn_id"]).set_index(["seed", "rxn_id"]).y_true
        df = pd.DataFrame(dict(y_pred=yhat, y_true=yt)).dropna().reset_index()
        rows.append(df.assign(arm="OUR-Σ", model="sum_of_channel_reps", target=t, rowset="C", group="SIGMA"))
    return pd.concat(rows, ignore_index=True)


def bias_check(repB):
    f = UNITS / "bias" / "ESP-C.parquet"
    if not f.is_file():
        return pd.DataFrame()
    b = pd.read_parquet(f)
    S = pd.read_csv(OUT / "S_rows.csv").set_index("rxn_id")
    b["D"] = b.rxn_id.map(S.D)
    rng = np.random.default_rng(R6.BOOT_SEED)
    rows = []
    for t in R6.TARGETS_B:
        for m in R6.MODELS:
            x = b[(b.target == t) & (b.model == m)]
            eD = np.abs(x[x.D].y_pred - x[x.D].y_true).to_numpy()
            eC = np.abs(x[x.inC].y_pred - x[x.inC].y_true).to_numpy()
            if not len(eD) or not len(eC):
                continue
            bs = np.array([rng.choice(eD, len(eD)).mean() - rng.choice(eC, len(eC)).mean() for _ in range(R6.N_BOOT)])
            rows.append(dict(target=t, model=m, representative=m == repB["ESP-C"][t]["model"], n_D=len(eD),
                             n_C=len(eC), mae_D=float(eD.mean()), mae_C=float(eC.mean()),
                             diff_D_minus_C=float(eD.mean() - eC.mean()), ci_lo=float(np.quantile(bs, .025)),
                             ci_hi=float(np.quantile(bs, .975))))
    return pd.DataFrame(rows)


def cmd_B():
    rec, pred = load_units(["B", "Call"])
    recA, predA = load_units(["A"])
    ps = per_seed(rec, pred)
    sm = summary(ps)
    R6.write_atomic(OUT / "B_per_seed.csv", lambda t: ps.to_csv(t, index=False))
    R6.write_atomic(OUT / "B_summary.csv", lambda t: sm.to_csv(t, index=False))
    R6.write_atomic(OUT / "B_predictions.parquet", lambda t: pred.to_parquet(t, index=False))
    rep = representative(sm, "C")
    rep_all = representative(sm, "C_all")
    # primary
    prim = []
    for t in R6.TARGETS_B:
        prim.append(paired(ps, "ESP-C", "OUR-O", t, "C", rep["ESP-C"][t]["model"], rep["OUR-O"][t]["model"],
                           "primary"))
    prim = pd.DataFrame(prim)
    prim["p_holm"] = R6.holm(prim.p.to_numpy())
    R6.write_atomic(OUT / "B_primary_tests.csv", lambda t: prim.to_csv(t, index=False))
    # secondary (descriptive)
    sec = []
    cref = pd.read_csv(OUT / "S_rows.csv")
    drop_cref = cref.loc[cref.recovered_by.fillna("").str.contains("constrained_ref"), "rxn_id"].tolist()
    test_best = {t: sm[(sm.arm == "ESP-C") & (sm.target == t) & (sm.rowset == "C") & sm.model.isin(R6.MODELS)]
                 .sort_values("mae").model.iloc[0] for t in R6.TARGETS_B}
    for t in R6.TARGETS_B:
        r = {a: rep[a][t]["model"] for a in rep}
        sec.append(paired(ps, "ESP-O", "OUR-O", t, "C", r["ESP-O"], r["OUR-O"], "ESP-O_vs_OUR-O (rep)"))
        sec.append(paired(ps, "ESP-C", "OUR-C", t, "C", r["ESP-C"], r["OUR-C"], "ESP-C_vs_OUR-C (rep)"))
        for m in R6.MODELS:
            sec.append(paired(ps, "ESP-C", "OUR-O", t, "C", m, m, f"same_model {m}: ESP-C_vs_OUR-O"))
            sec.append(paired(ps, "ESP-O", "OUR-O", t, "C", m, m, f"same_model {m}: ESP-O_vs_OUR-O"))
            sec.append(paired(ps, "ESP-C", "OUR-C", t, "C", m, m, f"same_model {m}: ESP-C_vs_OUR-C"))
        sec.append(paired(ps, "ESP-C", "OUR-O", t, "C", test_best[t], r["OUR-O"], "ESP-C test-best-of_vs_OUR-O rep"))
        if "ESP-C" in rep_all and "OUR-O" in rep_all:
            sec.append(paired(ps, "ESP-C", "OUR-O", t, "C_all", rep_all["ESP-C"][t]["model"],
                              rep_all["OUR-O"][t]["model"], "C_all sensitivity (rep on C_all)"))
        sec.append(paired(ps, "ESP-C", "OUR-O", t, "C", r["ESP-C"], r["OUR-O"], "without constrained_ref test rows",
                          filt=drop_cref, pred=pred))
    # Σ-channel arm
    psA = per_seed(recA, predA)
    smA = summary(pd.concat([psA, ps[ps.target.isin(["d1", "d2"]) & ps.arm.isin(["OUR-O", "ESP-O"])]]))
    repA = representative(smA, "C")
    sig = sigma_channel(pd.concat([predA, pred[pred.target.isin(["d1", "d2"])]]), repA, pred)
    sig_ps = sig.groupby(["arm", "model", "target", "seed", "rowset"]).apply(metrics).reset_index()
    ntr = ps[(ps.arm == "OUR-O") & (ps.rowset == "C")].groupby("seed").n_train.first()
    sig_ps["n_train"] = sig_ps.seed.map(ntr)
    ps_sig = pd.concat([ps, sig_ps.assign(val_mae=np.nan, val_source="none", best_params="{}", edge_hits=0)])
    for t in ("Int", "dEa"):
        sec.append(paired(ps_sig, "OUR-Σ", "OUR-O", t, "C", "sum_of_channel_reps", rep["OUR-O"][t]["model"],
                          "sigma_channel_vs_OUR-O direct rep"))
        sec.append(paired(ps_sig, "ESP-C", "OUR-Σ", t, "C", rep["ESP-C"][t]["model"], "sum_of_channel_reps",
                          "ESP-C rep_vs_sigma_channel"))
    sec = pd.DataFrame(sec)
    R6.write_atomic(OUT / "B_secondary_tests.csv", lambda t: sec.to_csv(t, index=False))
    R6.write_atomic(OUT / "B_sigma_per_seed.csv", lambda t: sig_ps.to_csv(t, index=False))
    R6.write_atomic(OUT / "B_sigma_predictions.parquet", lambda t: sig.to_parquet(t, index=False))
    R6.write_json(OUT / "B_representative.json", dict(C=rep, C_all=rep_all, espley_test_best_of=test_best,
                                                      rule="lowest 5-seed mean validation MAE (§3-3)"))
    bias = bias_check(rep)
    R6.write_atomic(OUT / "B_bias_check.csv", lambda t: bias.to_csv(t, index=False))
    pre = R6.gate_prereg()
    side = {k: json.loads((OUT / f).read_text()) if (OUT / f).is_file() else None
            for k, f in (("P1", "P1_reproduction.json"), ("P2", "P2_protocol_check.json"),
                         ("P3", "P3_role_check.json"), ("S", "S.json"), ("R5", "R5_report.json"))}
    R6.write_json(OUT / "B.json", dict(prereg=pre, units=dict(n=len(rec), by_arm=rec.arm.value_counts().to_dict()),
                                       representative=rep, primary=prim.to_dict("records"),
                                       predictions_sha256=R6.sha256(OUT / "B_predictions.parquet"),
                                       versions=rec.versions.iloc[0] if "versions" in rec else None, **side))
    print(prim[["target", "a", "b", "mae_a", "mae_b", "d_mean", "ci_lo", "ci_hi", "p", "p_holm",
                "seeds_b_lower"]].to_string())
    return 0


def cmd_A():
    recA, predA = load_units(["A"])
    recB, predB = load_units(["B"])
    keepB = recB.arm.isin(["OUR-O", "ESP-O"]) & recB.target.isin(["d1", "d2"])
    rec = pd.concat([recA, recB[keepB]], ignore_index=True)
    pred = pd.concat([predA, predB[predB.arm.isin(["OUR-O", "ESP-O"]) & predB.target.isin(["d1", "d2"])]])
    ps = per_seed(rec, pred)
    sm = summary(ps)
    R6.write_atomic(OUT / "A_per_seed.csv", lambda t: ps.to_csv(t, index=False))
    R6.write_atomic(OUT / "A_summary.csv", lambda t: sm.to_csv(t, index=False))
    R6.write_atomic(OUT / "A_predictions.parquet", lambda t: pred.to_parquet(t, index=False))
    R6.write_json(OUT / "A_representative.json", dict(C=representative(sm, "C"), rule="§3-3"))
    if (UNITS / "AppA").is_dir() and any((UNITS / "AppA").glob("*.json")):
        rA, pA = load_units(["AppA"])
        psA = per_seed(rA, pA)
        R6.write_atomic(OUT / "A_appendix_per_seed.csv", lambda t: psA.to_csv(t, index=False))
        R6.write_atomic(OUT / "A_appendix_summary.csv", lambda t: summary(psA).to_csv(t, index=False))
        R6.write_atomic(OUT / "A_appendix_predictions.parquet", lambda t: pA.to_parquet(t, index=False))
    print(sm[sm.arm == "OUR-O"].pivot_table(index="target", columns="model", values="mae").round(3).to_string())
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "B":
        sys.exit(cmd_B())
    if cmd == "A":
        sys.exit(cmd_A())
    raise SystemExit(__doc__)
