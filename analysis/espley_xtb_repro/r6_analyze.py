#!/usr/bin/env python3
"""r6_analyze.py — rev 6 Phase B / A outputs (spec §7, §8-2, §8-3, §9) from the unit cache ($R6_SCRATCH/units/).

  python r6_analyze.py B        # B_per_seed, B_summary, B_representative.json, B_primary_tests, B_secondary_tests,
                                # B_bias_check, B_predictions.parquet, B_sigma_*, B.json
  python r6_analyze.py A        # A_per_seed, A_summary, A_representative.json, A_predictions.parquet,
                                # A_appendix_per_seed / _summary / _predictions

Representative model (§3-3, PREREG_REV6 §3): per arm, target and row set, the model with the lowest validation MAE
averaged over the 5 seeds (our pipeline: inner-CV best_score_; Espley protocol: their GridSearchCV best_score_ = hps.pkl
train_mae, constant over seeds; NNs: the 5-fold CV of the tuned NN). Test scores never choose anything.
Primary test (§7): per target d_s = MAE_s(ESP-C rep) - MAE_s(OUR-O rep) over the 5 seeds, Nadeau-Bengio corrected t
(rev5_common.nadeau_bengio, J = 5, n_test / n_train = seed means), two-sided p, 95 % CI; Holm over the 4 targets;
seeds won by OUR-O (d_s > 0). Everything else is descriptive (no correction).
Row identity (§2-3): every unit of a (target, row set, seed) must carry the same test_sha256, equal to S.json's
split_sha256["<seed>/test/<row set>"] — else STOP (exit 3). All metrics come from the stored per-reaction predictions.
"""
from __future__ import annotations

import itertools
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
ARMS4 = ["ESP-C", "ESP-O", "OUR-C", "OUR-O"]


def load_units(groups, required=True):
    recs, preds = [], []
    for g in groups:
        for f in sorted((UNITS / g).glob("*.json")):
            p = f.with_suffix(".parquet")
            if not p.is_file():
                sys.exit(f"STOP: {p} missing")
            recs.append(json.loads(f.read_text()))
            preds.append(pd.read_parquet(p))
    if not recs:
        if required:
            sys.exit(f"no units in {groups}")
        return None, None
    return pd.DataFrame(recs), pd.concat(preds, ignore_index=True)


def check_rows(rec):
    """§2-3: one test row set per (target, rowset, seed), equal to the Phase S split."""
    S = json.loads((OUT / "S.json").read_text())["split_sha256"]
    bad = []
    for (t, rs, s), g in rec.groupby(["target", "rowset", "seed"]):
        shas = set(g.test_sha256)
        want = S.get(f"{s}/test/{rs}", {}).get("sha256") if rs != "appA" else S.get(f"A/{s}/test", {}).get("sha256")
        if len(shas) != 1 or (want is not None and shas != {want}):
            bad.append(f"{t}/{rs}/seed {s}: {len(shas)} test sets, Phase S {str(want)[:12]}")
    if bad:
        print("STOP (§2-3 rows differ between arms / from Phase S):\n  " + "\n  ".join(bad[:20]), file=sys.stderr)
        sys.exit(3)
    return dict(n_groups=int(rec.groupby(["target", "rowset", "seed"]).ngroups), identical=True)


def metrics(df):
    y, p = df.y_true.to_numpy(float), df.y_pred.to_numpy(float)
    e = np.abs(p - y)
    ss = float(np.sum((y - y.mean()) ** 2))
    return pd.Series(dict(mae=float(e.mean()), nmae=R6.nmae(y, p), r2=1 - float(np.sum((p - y) ** 2)) / ss if ss else np.nan,
                          within1=float(np.mean(e <= R6.WITHIN_KCAL)), n_test=len(df)))


def per_seed(rec, pred, key=("arm", "model", "target", "seed", "rowset")):
    m = pred.groupby(list(key)).apply(metrics).reset_index()
    cols = list(key) + ["n_train", "val_mae", "val_source"] + (["their_y_train_cv"] if "their_y_train_cv" in rec else [])
    info = rec[cols].copy()
    info["best_params"] = rec["best_params"].map(lambda d: json.dumps(d, sort_keys=True, default=str))
    info["edge_hits"] = rec["edge_hits"].map(lambda d: np.nan if d is None or (isinstance(d, float) and np.isnan(d))
                                             else int(sum(bool(v) for v in d.values())))
    return m.merge(info, on=list(key), how="left")


def summary(ps, key=("arm", "model", "target", "rowset")):
    g = ps.groupby(list(key))
    out = g.agg(mae=("mae", "mean"), mae_sd=("mae", "std"), nmae=("nmae", "mean"), r2=("r2", "mean"),
                within1=("within1", "mean"), val_mae=("val_mae", "mean"), n_seeds=("seed", "nunique"),
                n_test=("n_test", "mean"), n_train=("n_train", "mean")).reset_index()
    out = out.merge(g.edge_hits.sum(min_count=1).rename("edge_hits").reset_index(), on=list(key))
    out["mae_se"] = out.mae_sd / np.sqrt(out.n_seeds)
    return out


def representative(sm, rowset, col="val_mae"):
    rep = {}
    for (arm, t), g in sm[(sm.rowset == rowset) & sm.model.isin(R6.MODELS)].groupby(["arm", "target"]):
        g = g.sort_values([col, "model"])
        rep.setdefault(arm, {})[t] = dict(model=str(g.model.iloc[0]),
                                          val_mae_by_model={m: float(v) for m, v in zip(g.model, g[col])})
    return rep


def paired(ps, a, b, t, rowset, ma, mb, label, filt=None, pred=None):
    """d_s = MAE_s(a, ma) - MAE_s(b, mb) on the same rows; NB t-test. filt: rxn_ids dropped before scoring."""
    if filt is not None:
        q = pred[(pred.target == t) & (pred.rowset == rowset) & ~pred.rxn_id.isin(filt)]
        f = lambda d: float(np.mean(np.abs(d.y_pred - d.y_true)))           # noqa: E731
        A = q[(q.arm == a) & (q.model == ma)].groupby("seed").apply(f)
        B = q[(q.arm == b) & (q.model == mb)].groupby("seed").apply(f)
        ntest = q[(q.arm == b) & (q.model == mb)].groupby("seed").size()
    else:
        x = ps[(ps.target == t) & (ps.rowset == rowset)]
        A = x[(x.arm == a) & (x.model == ma)].set_index("seed").mae
        B = x[(x.arm == b) & (x.model == mb)].set_index("seed").mae
        ntest = x[(x.arm == b) & (x.model == mb)].set_index("seed").n_test
    ntr = ps[(ps.target == t) & (ps.rowset == rowset) & (ps.arm == b) & (ps.model == mb)].set_index("seed").n_train
    seeds = sorted(set(A.index) & set(B.index))
    base = dict(comparison=label, target=t, rowset=rowset, a=f"{a}:{ma}", b=f"{b}:{mb}")
    if len(seeds) != len(R6.SEEDS):
        return dict(base, status=f"missing_seeds ({len(seeds)})")
    d = np.array([A[s] - B[s] for s in seeds])
    nb = R6.nadeau_bengio(d, float(np.mean([ntest[s] for s in seeds])), float(np.mean([ntr[s] for s in seeds])))
    return dict(base, mae_a=float(A[seeds].mean()), mae_b=float(B[seeds].mean()), d_mean=nb["mean"],
                ci_lo=nb["ci95"][0], ci_hi=nb["ci95"][1], t=nb["t"], p=nb["p"], seeds_b_lower=int((d > 0).sum()),
                d_per_seed=json.dumps([float(v) for v in d]), n_test=nb["n_test"], n_train=nb["n_train"], status="ok")


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
        df = pd.DataFrame(dict(y_pred=yhat, y_true=yt))
        assert not df.isna().any().any(), f"sigma channel {t}: rows do not line up"
        rows.append(df.reset_index().assign(arm="OUR-Σ", model="sum_of_channel_reps", target=t, rowset="C",
                                            group="SIGMA"))
    return pd.concat(rows, ignore_index=True), {ch: rep[ch]["model"] for ch in parts}


def bias_check(repB):
    """§2-2: ESP-C |error| on the test rows our side lost (D) vs the C rows, 5 seeds pooled; per reaction the |error|
    is averaged over the seeds in which it was a test row, then reactions are resampled (10,000, rng 20261001)."""
    f = UNITS / "bias" / "ESP-C.parquet"
    if not f.is_file():
        return pd.DataFrame()
    b = pd.read_parquet(f)
    S = pd.read_csv(OUT / "S_rows.csv").set_index("rxn_id")
    b["D"] = b.rxn_id.map(S.D).astype(bool)
    rows = []
    for t in R6.TARGETS_B:
        for m in R6.MODELS:
            x = b[(b.target == t) & (b.model == m)].assign(ae=lambda d: np.abs(d.y_pred - d.y_true))
            per = x.groupby(["rxn_id", "D", "inC"]).ae.mean().reset_index()
            eD, eC = per[per.D].ae.to_numpy(), per[per.inC].ae.to_numpy()
            if not len(eD) or not len(eC):
                continue
            rng = np.random.default_rng(R6.BOOT_SEED)
            bs = np.array([rng.choice(eD, len(eD)).mean() - rng.choice(eC, len(eC)).mean() for _ in range(R6.N_BOOT)])
            rows.append(dict(target=t, model=m, representative=m == repB["ESP-C"][t]["model"], n_D=len(eD),
                             n_C=len(eC), mae_D=float(eD.mean()), mae_C=float(eC.mean()),
                             diff_D_minus_C=float(eD.mean() - eC.mean()), ci_lo=float(np.quantile(bs, .025)),
                             ci_hi=float(np.quantile(bs, .975)), unit="reaction (seed-averaged |error|)"))
    return pd.DataFrame(rows)


def cmd_B():
    pre = R6.gate_prereg()
    rec, pred = load_units(["B", "Call", "Nocref"])
    rows_ok = check_rows(rec)
    ps = per_seed(rec, pred)
    sm = summary(ps)
    R6.write_atomic(OUT / "B_per_seed.csv", lambda t: ps.to_csv(t, index=False))
    R6.write_atomic(OUT / "B_summary.csv", lambda t: sm.to_csv(t, index=False))
    R6.write_atomic(OUT / "B_predictions.parquet", lambda t: pred.to_parquet(t, index=False))
    rep = representative(sm, "C")
    rep_all, rep_nc = representative(sm, "C_all"), representative(sm, "C_nocref")
    # sensitivity of the Espley-protocol rule: their per-seed y_train_CV (5-fold CV of the run model) for the sklearn
    # models / XGB, the NN 5-fold CV as before
    sm2 = sm.merge(ps.groupby(["arm", "model", "target", "rowset"]).their_y_train_cv.mean().rename("ytcv").reset_index(),
                   on=["arm", "model", "target", "rowset"], how="left") if "their_y_train_cv" in ps else sm.assign(ytcv=np.nan)
    sm2["val_ytcv"] = sm2.ytcv.where(sm2.ytcv.notna(), sm2.val_mae)
    rep_ytcv = representative(sm2, "C", col="val_ytcv")
    # primary
    prim = pd.DataFrame([paired(ps, "ESP-C", "OUR-O", t, "C", rep["ESP-C"][t]["model"], rep["OUR-O"][t]["model"],
                                "primary") for t in R6.TARGETS_B])
    prim["p_holm"] = R6.holm(prim.p.to_numpy())
    R6.write_atomic(OUT / "B_primary_tests.csv", lambda t: prim.to_csv(t, index=False))
    # secondary (descriptive)
    sec = []
    S = pd.read_csv(OUT / "S_rows.csv")
    drop_cref = S.loc[S.recovered_by.fillna("").astype(str).str.contains("constrained_ref"), "rxn_id"].tolist()
    test_best = {t: str(sm[(sm.arm == "ESP-C") & (sm.target == t) & (sm.rowset == "C") & sm.model.isin(R6.MODELS)]
                        .sort_values("mae").model.iloc[0]) for t in R6.TARGETS_B}
    for t in R6.TARGETS_B:
        r = {a: rep[a][t]["model"] for a in rep}
        sec.append(paired(ps, "ESP-O", "OUR-O", t, "C", r["ESP-O"], r["OUR-O"], "ESP-O_vs_OUR-O (rep)"))
        sec.append(paired(ps, "ESP-C", "OUR-C", t, "C", r["ESP-C"], r["OUR-C"], "ESP-C_vs_OUR-C (rep)"))
        for m in R6.MODELS:
            for a, b in itertools.combinations(ARMS4, 2):
                sec.append(paired(ps, a, b, t, "C", m, m, f"same_model {m}: {a}_vs_{b}"))
        sec.append(paired(ps, "ESP-C", "OUR-O", t, "C", test_best[t], r["OUR-O"], "ESP-C test-best-of_vs_OUR-O rep"))
        sec.append(paired(ps, "ESP-C", "OUR-O", t, "C", rep_ytcv["ESP-C"][t]["model"], r["OUR-O"],
                          "ESP-C rep by their y_train_CV_vs_OUR-O rep"))
        if "ESP-C" in rep_all and "OUR-O" in rep_all:
            sec.append(paired(ps, "ESP-C", "OUR-O", t, "C_all", rep_all["ESP-C"][t]["model"],
                              rep_all["OUR-O"][t]["model"], "C_all sensitivity (retrained, rep on C_all)"))
        if "ESP-C" in rep_nc and "OUR-O" in rep_nc:
            sec.append(paired(ps, "ESP-C", "OUR-O", t, "C_nocref", rep_nc["ESP-C"][t]["model"],
                              rep_nc["OUR-O"][t]["model"], "without constrained_ref rows (retrained, rep on C_nocref)"))
        sec.append(paired(ps, "ESP-C", "OUR-O", t, "C", r["ESP-C"], r["OUR-O"],
                          "without constrained_ref test rows (rescored only)", filt=drop_cref, pred=pred))
    # Σ-channel arm (needs Phase A)
    recA, predA = load_units(["A"], required=False)
    sig_ps, sig_models = None, None
    if recA is not None:
        psA = per_seed(recA, predA)
        smA = summary(pd.concat([psA, ps[ps.target.isin(["d1", "d2"]) & ps.arm.isin(["OUR-O", "ESP-O"])]]))
        repA = representative(smA, "C")
        sig, sig_models = sigma_channel(pd.concat([predA, pred[pred.target.isin(["d1", "d2"])]]), repA, pred)
        sig_ps = sig.groupby(["arm", "model", "target", "seed", "rowset"]).apply(metrics).reset_index()
        ntr = ps[(ps.arm == "OUR-O") & (ps.rowset == "C")].groupby("seed").n_train.first()
        sig_ps["n_train"] = sig_ps.seed.map(ntr)
        ps_sig = pd.concat([ps, sig_ps.assign(val_mae=np.nan, val_source="none", best_params="{}", edge_hits=np.nan)])
        for t in ("Int", "dEa"):
            sec.append(paired(ps_sig, "OUR-Σ", "OUR-O", t, "C", "sum_of_channel_reps", rep["OUR-O"][t]["model"],
                              "sigma_channel_vs_OUR-O direct rep"))
            sec.append(paired(ps_sig, "ESP-C", "OUR-Σ", t, "C", rep["ESP-C"][t]["model"], "sum_of_channel_reps",
                              "ESP-C rep_vs_sigma_channel"))
        R6.write_atomic(OUT / "B_sigma_per_seed.csv", lambda t: sig_ps.to_csv(t, index=False))
        R6.write_atomic(OUT / "B_sigma_predictions.parquet", lambda t: sig.to_parquet(t, index=False))
    sec = pd.DataFrame(sec)
    R6.write_atomic(OUT / "B_secondary_tests.csv", lambda t: sec.to_csv(t, index=False))
    R6.write_json(OUT / "B_representative.json", dict(C=rep, C_all=rep_all, C_nocref=rep_nc,
                                                      espley_rule_by_their_y_train_cv=rep_ytcv,
                                                      espley_test_best_of=test_best, sigma_channel_models=sig_models,
                                                      rule="lowest 5-seed mean validation MAE (§3-3)"))
    bias = bias_check(rep)
    R6.write_atomic(OUT / "B_bias_check.csv", lambda t: bias.to_csv(t, index=False))
    side = {k: json.loads((OUT / f).read_text()) if (OUT / f).is_file() else None
            for k, f in (("P1", "P1_reproduction.json"), ("P2", "P2_protocol_check.json"),
                         ("P3", "P3_role_check.json"), ("S", "S.json"), ("R5", "R5_report.json"))}
    R6.write_json(OUT / "B.json", dict(prereg=pre, rows_identical=rows_ok,
                                       units=dict(n=len(rec), by_group_arm=rec.groupby(["group", "arm"]).size()
                                                  .rename("n").reset_index().to_dict("records")),
                                       representative=rep, primary=prim.to_dict("records"),
                                       predictions_sha256=R6.sha256(OUT / "B_predictions.parquet"),
                                       versions=rec.versions.dropna().iloc[0] if "versions" in rec else None, **side))
    print(prim[["target", "a", "b", "mae_a", "mae_b", "d_mean", "ci_lo", "ci_hi", "p", "p_holm",
                "seeds_b_lower"]].to_string())
    return 0


def cmd_A():
    R6.gate_prereg()
    recA, predA = load_units(["A"])
    recB, predB = load_units(["B"])
    keepB = recB.arm.isin(["OUR-O", "ESP-O"]) & recB.target.isin(["d1", "d2"])
    rec = pd.concat([recA, recB[keepB]], ignore_index=True)
    pred = pd.concat([predA, predB[predB.arm.isin(["OUR-O", "ESP-O"]) & predB.target.isin(["d1", "d2"])]])
    check_rows(rec)
    ps = per_seed(rec, pred)
    sm = summary(ps)
    R6.write_atomic(OUT / "A_per_seed.csv", lambda t: ps.to_csv(t, index=False))
    R6.write_atomic(OUT / "A_summary.csv", lambda t: sm.to_csv(t, index=False))
    R6.write_atomic(OUT / "A_predictions.parquet", lambda t: pred.to_parquet(t, index=False))
    R6.write_json(OUT / "A_representative.json", dict(C=representative(sm, "C"), rule="§3-3"))
    rA, pA = load_units(["AppA"], required=False)
    if rA is not None:
        check_rows(rA)
        psA = per_seed(rA, pA)
        smA = summary(psA)
        R6.write_atomic(OUT / "A_appendix_per_seed.csv", lambda t: psA.to_csv(t, index=False))
        R6.write_atomic(OUT / "A_appendix_summary.csv", lambda t: smA.to_csv(t, index=False))
        R6.write_atomic(OUT / "A_appendix_predictions.parquet", lambda t: pA.to_parquet(t, index=False))
        R6.write_json(OUT / "A_appendix_representative.json", dict(appA=representative(smA, "appA"), rule="§3-3"))
    print(sm[sm.arm == "OUR-O"].pivot_table(index="target", columns="model", values="mae").round(3).to_string())
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "B":
        sys.exit(cmd_B())
    if cmd == "A":
        sys.exit(cmd_A())
    raise SystemExit(__doc__)
