#!/usr/bin/env python3
r"""eval_abs.py — ABS3 Phase E (docs/specs/ABS3_MODELS.md 8). CPU, env_espley (jobs/cpu_abs.slurm ABS3ENV=espley); no torch.

  python eval_abs.py selftest   before K (8-6): the same code reproduces the CH3M EM_metrics.csv P_main para MAE rows and
                                the EM_rank.csv para rows of GNN3 · TF3 · XGB3 at 4 decimals; a difference is reported (not
                                a STOP) -> results/EA_selftest.json
  python eval_abs.py run        6 models (A-TF, A-GNN, A-XGB, TF3, GNN3, XGB3) -> EA_abs.csv (8-2), ED_compare.csv (8-3),
                                EC_cancel.csv (8-4), EA_metrics.csv, EA_rank.csv, EA_cause.csv, EA_identity.csv (8-5),
                                EA_report.json. Exit 0 written, 2 inputs missing, 3 STOP (y differs from the labels).

The ch3 evaluation code is used unchanged (analysis/ch3/eval_ch3.py: labels, s_folds, ch3_frame, assemble, set_mask,
wide, compare, metrics, rank_table, cause, cause_rows, identity_residual, pearson, _clean; analysis/geo2/eval_geo2.py
boot_diff through it). Pair frames: A-x = results/KA_pairs.parquet (yhat(a) - yhat(b)), TF3 = ch3 K3_predictions, GNN3 /
XGB3 = CH3M KM_predictions rung lf3 (= ch3m eval_m.direct4).
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
import abs_common as C  # noqa: E402
import eval_ch3 as EV  # noqa: E402                   (analysis/ch3, on sys.path through abs_common)

EG = EV.EG
T3 = list(C.TARGETS3)
KA = C.RES / "KA_predictions.parquet"
KP = C.RES / "KA_pairs.parquet"
KR = C.RES / "KA_report.json"
NOISE_FLOOR = 0.02
ABS = [C.ABS_LABEL[m] for m in C.MODELS]
PAIR = [C.PAIR_LABEL[m] for m in C.MODELS]
MODELS6 = list(C.LABEL_MODELS)
BEST_REF = "GNN3"
MARKS = ("절대값 나음", "짝 나음", "같음")
GROUP_BY = ("all", "dipole", "attach", "role")


def _defs():
    return {
        "models": "A-TF / A-GNN / A-XGB = 절대값 모델(KA_predictions, 반응 하나의 strain · inter · solv · barrier); 짝 ΔΔ = "
                  "yhat(a) − yhat(b) (같은 fold 모델, KA_pairs); TF3 = ch3 K3_predictions; GNN3 / XGB3 = CH3M "
                  "KM_predictions rung lf3",
        "truth": "반응 정답 = labels_d1_cr2.json의 10 키 @ M; 짝 정답 = F_pairs_cr2 dd_<10> @ M (eval_ch3.labels). 다르면 STOP",
        "pairs": "6개 모델이 모두 예측한 공통 시험 짝; 주 집합 = P_main(reference & in_P_main) · para, 보조 = P_main all",
        "q_a": "EA_abs: 시험 반응 전부(fold마다 그 fold의 시험 반응, KA_predictions), 모델 × 타깃 × 묶음(all, dipole, attach, "
               "role H / sub): n, MAE, MAE의 core 부트스트랩 95 % CI(eval_geo2.boot_diff(e, 0, core), 2,000회, seed "
               "20261019; core = 반응이 속한 짝들의 S_folds core), RMSE, Pearson r, R² = 1 − SSE / SST(묶음 안 y 평균 기준), "
               "기준값: 학습 반응 평균 예측의 MAE(fold마다 그 fold 학습 반응의 타깃 평균), 정답 표준편차(ddof 0)",
        "q_d": "ED_compare: 모델 종류마다 A-x 대 x3 (TF: A-TF 대 TF3, GNN: A-GNN 대 GNN3, XGB: A-XGB 대 XGB3), eval_ch3.compare "
               "(P_main 위치 p · m · all, 상대 차 r = (MAE(A-x) − MAE(x3)) / MAE(x3), core 부트스트랩 2,000회 seed 20261019 "
               "95 % CI). 표시(p · all 행): 문턱 = max(noise_t, 0.02), noise_t = CH3M EM_noise.csv 같은 위치 · 타깃의 "
               "|r(TF10 대 dep)|; 절대값 나음 = 상한 < 0 & r < −문턱; 짝 나음 = 하한 > 0 & r > +문턱; 그 밖 같음. 추가 줄: "
               "4타깃 P_main para MAE 평균이 가장 낮은 절대값 모델 대 GNN3, 같은 형식",
        "q_c": "EC_cancel: 절대값 모델마다 P_main para · all 공통 시험 짝, 타깃마다 e(r) = yhat(r) − y(r)(반응 정답); "
               "corr(e(a), e(b)) (Pearson); 상쇄 비율 = 평균 |e(a) − e(b)| / 평균(|e(a)| + |e(b)|); MAE(a), MAE(b), "
               "MAE(ΔΔ) = 평균 |e(a) − e(b)|; ED_compare의 상대 차를 함께 적음",
        "aux": "EA_metrics / EA_rank / EA_cause: eval_ch3 정의 그대로(공통 짝, 9-3 주 집합 P_sens_ref, 예측 자신의 barrier "
               "부호); EA_identity: 반응 단위 |yhat_barrier − (yhat_strain + yhat_inter + yhat_solv)|(절대값 모델, 시험 "
               "반응)와 짝 단위 같은 값(6개 모델, 공통 짝)",
    }


DEFINITIONS = _defs()


# ================================================================ inputs
def load_pack():
    if not (C.PACK_NPZ.is_file() and C.PACK_JSON.is_file()):
        C.not_ready(f"{C.PACK_NPZ} missing")
    info = C.read_json(C.PACK_JSON)
    if not info.get("ok") or C.sha256(C.PACK_NPZ) != info.get("npz_sha256"):
        C.stop(f"{C.PACK_NPZ}: pack not ok or changed after its json")
    return np.load(C.PACK_NPZ, allow_pickle=False), info


def direct4(K, name, Y):
    """= analysis/ch3m/eval_m.py direct4: a 4-target long frame of a pair model, y checked against F_pairs_cr2 @ M."""
    K = K.copy()
    K["pair_id"] = K.pair_id.astype(str)
    if set(K.target) != set(T3) or K.duplicated(["pair_id", "target"]).any():
        C.stop(f"{name}: targets {sorted(set(K.target))} / duplicated rows")
    dy = {t: float(np.abs(K[K.target == t].y.to_numpy(float) - Y.loc[K[K.target == t].pair_id, t].to_numpy(float)).max())
          for t in T3}
    if max(dy.values()) > 1e-9:
        C.stop(f"spec 12 stage 8: {name} y differs from F_pairs_cr2 @ M: {dy}")
    return K[["pair_id", "fold", "target", "yhat", "yhat_seed_sd"]].copy(), dict(max_abs_dy_vs_labels=dy,
                                                                               n_pairs=int(K.pair_id.nunique()))


def pair_frames(Y, with_abs=True):
    frames, info = {}, {}
    frames["TF3"], info["TF3"] = EV.ch3_frame(Y)
    KMp = pd.read_parquet(C.KM_PRED)
    for m in ("GNN", "XGB"):
        frames[f"{m}3"], info[f"{m}3"] = direct4(KMp[(KMp.rung == "lf3") & (KMp.model == m)], f"{m}3", Y)
    if with_abs:
        P = pd.read_parquet(KP)
        for m in ABS:
            frames[m], info[m] = direct4(P[P.model == m], m, Y)
    return frames, info


def common_ids(A, models):
    s = None
    for m in models:
        ids = set(A[A.model == m].pair_id)
        s = ids if s is None else s & ids
    return sorted(s)


def reaction_truth(ids):
    """labels_d1_cr2.json @ M for the reaction ids (spec 12 stage 8)."""
    by = {int(r["rxn_id"]): r for r in json.loads(C.LABELS_JSON.read_text())}
    y10 = np.array([[float(by[int(i)][C.LABEL_KEYS[t]]) for t in C.TARGETS10] for i in ids])
    return pd.DataFrame(y10 @ C.M(), index=[int(i) for i in ids], columns=T3)


# ================================================================ Q-A (8-2)
def mark(r, lo, hi, noise):
    th = max(noise, NOISE_FLOOR)
    if hi < 0 and r < -th:
        return MARKS[0]
    if lo > 0 and r > th:
        return MARKS[1]
    return MARKS[2]


def q_a(KAf, z):
    rid = z["rids"].astype(np.int64)
    row = {int(r): k for k, r in enumerate(rid)}
    y4 = z["y4"].astype(float)
    tmean = {k: y4[z[f"f{k}_train"].astype(int)].mean(axis=0) for k in range(C.NF)}
    core = z["core"].astype(str)
    role = z["role"].astype(str)
    rows = []
    for m in ABS:
        q = KAf[KAf.model == m].copy()
        ri = np.array([row[int(r)] for r in q.rxn_id])
        q["core"] = core[ri]
        q["role"] = role[ri]
        q["base"] = [tmean[int(f)][T3.index(t)] for f, t in zip(q.fold, q.target)]
        for t in T3:
            d = q[q.target == t]
            for gb in GROUP_BY:
                groups = [("all", d)] if gb == "all" else list(d.groupby(gb, sort=True))
                for g, x in groups:
                    y, yh = x.y.to_numpy(float), x.yhat.to_numpy(float)
                    e = yh - y
                    cores = x.core.to_numpy()
                    mae, lo, hi, _ = EG.boot_diff(e, np.zeros(len(e)), cores)
                    sst = float(((y - y.mean()) ** 2).sum())
                    rows.append(dict(model=m, target=t, group_by=gb, group=str(g), n=int(len(x)),
                                     n_cores=int(len(set(cores))), mae=float(np.abs(e).mean()), mae_ci_lo=lo, mae_ci_hi=hi,
                                     rmse=float(np.sqrt((e ** 2).mean())), r=EV.pearson(y, yh),
                                     r2=float(1.0 - (e ** 2).sum() / sst) if sst > 0 else np.nan,
                                     base_mae_train_mean=float(np.abs(x.base.to_numpy(float) - y).mean()),
                                     y_sd=float(y.std()), mae_over_y_sd=float(np.abs(e).mean() / y.std()) if y.std() > 0
                                     else np.nan))
    return pd.DataFrame(rows)


# ================================================================ Q-D (8-3)
def q_d(A, ids):
    nz = pd.read_csv(C.EM_NOISE)
    nz = nz[(nz.x == "TF10") & (nz.b == "dep")]
    noise = {(r.pos, r.target): abs(float(r.rel)) for r in nz.itertuples()}
    out = []

    def one(x, b, label, model):
        c = EV.compare(A, ids, x=x, b=b).drop(columns=["rule_pass", "decision"], errors="ignore")
        c["comparison"], c["model"] = label, model
        c["noise_t"] = [noise.get((r.pos, r.target)) for r in c.itertuples()]
        c["threshold"] = [max(v, NOISE_FLOOR) if v is not None else None for v in c.noise_t]
        c["mark"] = [mark(r.rel, r.rel_ci_lo, r.rel_ci_hi, noise[(r.pos, r.target)]) if r.pos in ("p", "all") else None
                     for r in c.itertuples()]
        c["main"] = c.pos == "p"
        return c
    for m in C.MODELS:
        out.append(one(C.ABS_LABEL[m], C.PAIR_LABEL[m], f"{C.ABS_LABEL[m]} 대 {C.PAIR_LABEL[m]}", m))
    pm = EV.metrics(A, ids, ABS, sets=("P_main",))
    pm = pm[pm.pos == "p"]
    mean_mae = {m: float(pm[pm.model == m].mae.mean()) for m in ABS}
    best = min(ABS, key=lambda m: (mean_mae[m], ABS.index(m)))
    out.append(one(best, BEST_REF, f"{best} 대 {BEST_REF} (가장 좋은 절대값 모델 대 GNN3)", "best"))
    return pd.concat(out, ignore_index=True), best, mean_mae, noise


# ================================================================ Q-C (8-4)
def q_c(A, ids, KAf, L, QD):
    rel = {(r.comparison, r.pos, r.target): float(r.rel) for r in QD.itertuples()}
    rows = []
    ab = L[["a_rxn", "b_rxn"]].astype(int)
    for mk in C.MODELS:
        m = C.ABS_LABEL[mk]
        lab = f"{m} 대 {C.PAIR_LABEL[mk]}"
        W = EV.wide(A, m, ids)
        W = W.join(ab, how="left")
        R = KAf[KAf.model == m].set_index(["fold", "rxn_id", "target"])
        for pos in ("p", "all"):
            msk = EV.set_mask(W, "P_main") & ((W.pos == pos).to_numpy() if pos != "all" else True)
            w = W[msk]
            for t in T3:
                ka = pd.MultiIndex.from_arrays([w.fold.astype(int), w.a_rxn, [t] * len(w)])
                kb = pd.MultiIndex.from_arrays([w.fold.astype(int), w.b_rxn, [t] * len(w)])
                qa, qb = R.reindex(ka), R.reindex(kb)
                if qa.yhat.isna().any() or qb.yhat.isna().any():
                    C.stop(f"Q-C: reactions of common pairs without predictions ({m}, {t})")
                ea = (qa.yhat - qa.y).to_numpy(float)
                eb = (qb.yhat - qb.y).to_numpy(float)
                den = float((np.abs(ea) + np.abs(eb)).mean())
                rows.append(dict(model=m, eval_set="P_main", pos=pos, target=t, n=int(len(w)),
                                 corr_ea_eb=EV.pearson(ea, eb), cancel_ratio=float(np.abs(ea - eb).mean()) / den if den > 0
                                 else np.nan, mae_a=float(np.abs(ea).mean()), mae_b=float(np.abs(eb).mean()),
                                 mae_dd_from_rxn=float(np.abs(ea - eb).mean()),
                                 mae_dd_pair=float(np.abs(w[f"yhat_{t}"] - w[f"y_{t}"]).mean()),
                                 rel_vs_pair=rel.get((lab, pos, t))))
    return pd.DataFrame(rows)


# ================================================================ 8-5
def identity_rows(A, ids, KAf):
    rows = []
    for m in ABS:
        q = KAf[KAf.model == m]
        Wh = q.pivot_table(index=["fold", "rxn_id"], columns="target", values="yhat")[T3]
        Wy = q.pivot_table(index=["fold", "rxn_id"], columns="target", values="y")[T3]
        r = (Wh.barrier - (Wh.strain + Wh.inter + Wh.solv)).abs()
        ry = (Wy.barrier - (Wy.strain + Wy.inter + Wy.solv)).abs()
        rows.append(dict(level="reaction", model=m, yhat_mean_abs=float(r.mean()), yhat_p95_abs=float(np.percentile(r, 95)),
                         yhat_max_abs=float(r.max()), y_max_abs=float(ry.max()), n=int(len(r))))
    for m in MODELS6:
        rows.append(dict(level="pair", **EV.identity_residual(A, ids, m)))
    return pd.DataFrame(rows)


def code_sha():
    return {"analysis/abs3/eval_abs.py": C.sha256(Path(__file__).resolve()),
            "analysis/abs3/abs_common.py": C.sha256(HERE / "abs_common.py"),
            "analysis/ch3/eval_ch3.py": C.sha256(C.CH3_DIR / "eval_ch3.py"),
            "analysis/geo2/eval_geo2.py": C.sha256(Path(EG.__file__).resolve())}


# ================================================================ commands
def cmd_run():
    for p in (KA, KP, KR):
        if not p.is_file():
            C.not_ready(f"{p} missing (xgb_abs.py collect)")
    k = C.read_json(KR)
    if k.get("predictions_sha256") != C.sha256(KA) or k.get("pairs_sha256") != C.sha256(KP):
        C.stop(f"{KA.name} / {KP.name} changed after {KR.name}")
    if not all(v.get("ok") for v in (k.get("C_P") or {}).values()) or not k.get("C_P"):
        C.stop(f"C-P not ok in {KR.name}")
    import prereg_abs as PR
    z, _ = load_pack()
    L, Y, attr = EV.labels()
    KAf = pd.read_parquet(KA)
    ytrue = reaction_truth(sorted(KAf.rxn_id.unique()))
    dyr = {t: float(np.abs(KAf[KAf.target == t].y.to_numpy(float)
                           - ytrue.loc[KAf[KAf.target == t].rxn_id, t].to_numpy(float)).max()) for t in T3}
    if max(dyr.values()) > 1e-9:
        C.stop(f"spec 12 stage 8: KA_predictions y differs from labels_d1_cr2.json @ M: {dyr}")
    frames, info = pair_frames(Y)
    A = EV.assemble(frames, L, Y, attr)
    ids = common_ids(A, MODELS6)
    QA = q_a(KAf, z)
    QD, best, mean_mae, noise = q_d(A, ids)
    QC = q_c(A, ids, KAf, L, QD)
    M_ = EV.metrics(A, ids, MODELS6)
    R_ = EV.rank_table(A, ids, MODELS6)
    CZ = EV.cause_rows([EV.cause(A, ids, m) for m in MODELS6])
    IDN = identity_rows(A, ids, KAf)
    for name, df in (("EA_abs.csv", QA), ("ED_compare.csv", QD), ("EC_cancel.csv", QC), ("EA_metrics.csv", M_),
                     ("EA_rank.csv", R_), ("EA_cause.csv", CZ), ("EA_identity.csv", IDN)):
        C.write_csv(C.RES / name, df)
    qa_all = QA[QA.group_by == "all"]
    pm = M_[(M_.eval_set == "P_main") & (M_.pos == "p")]
    summ = dict(
        q_a={m: {t: {c: float(g[g.target == t][c].iloc[0]) for c in ("mae", "mae_ci_lo", "mae_ci_hi", "r2", "rmse",
                                                                       "base_mae_train_mean", "y_sd", "n")}
                 for t in T3} for m, g in qa_all.groupby("model")},
        q_d={lab: {f"{r.pos}/{r.target}": dict(rel=r.rel, lo=r.rel_ci_lo, hi=r.rel_ci_hi, mark=r.mark, noise_t=r.noise_t,
                                                mae_x=r.mae_x, mae_b=r.mae_b)
                   for r in g.itertuples() if r.pos in ("p", "all")} for lab, g in QD.groupby("comparison", sort=False)},
        best_abs_model=best, best_rule="4타깃 P_main para MAE 평균 최소", para_mae_mean_abs=mean_mae,
        q_c={m: {f"{r.pos}/{r.target}": dict(corr=r.corr_ea_eb, cancel_ratio=r.cancel_ratio, rel=r.rel_vs_pair)
                 for r in g.itertuples()} for m, g in QC.groupby("model")},
        para_mae={m: {t: float(g[g.target == t].mae.iloc[0]) for t in T3} for m, g in pm.groupby("model")},
        noise_t={f"{p}/{t}": v for (p, t), v in noise.items()}, n_common=len(ids))
    rep = dict(spec="docs/specs/ABS3_MODELS.md 8", written=C.now(), definitions=DEFINITIONS, summary=summ,
               checks=dict(pairs=info, reactions_y_vs_labels_json=dyr),
               n_pairs=dict(common=len(ids), **{m: int(A[A.model == m].pair_id.nunique()) for m in MODELS6}),
               n_reactions={m: int(KAf[(KAf.model == m) & (KAf.target == "barrier")].shape[0]) for m in ABS},
               inputs_sha256=dict(KA_predictions=C.sha256(KA), KA_pairs=C.sha256(KP), KA_report=C.sha256(KR),
                                  K3_predictions=C.sha256(C.K3_PRED), KM_predictions=C.sha256(C.KM_PRED),
                                  EM_noise=C.sha256(C.EM_NOISE), labels=C.sha256(C.LABELS),
                                  labels_json=C.sha256(C.LABELS_JSON), S_folds=C.sha256(EV.S_FOLDS),
                                  pack_npz=C.sha256(C.PACK_NPZ), prereg_md=C.sha256(PR.MD) if PR.MD.is_file() else None),
               outputs_sha256={f: C.sha256(C.RES / f) for f in ("EA_abs.csv", "ED_compare.csv", "EC_cancel.csv",
                                                                 "EA_metrics.csv", "EA_rank.csv", "EA_cause.csv",
                                                                 "EA_identity.csv")},
               code_sha256=code_sha())
    C.write_json(C.RES / "EA_report.json", rep)
    print(json.dumps(C.clean(dict(best=best, q_d={lab: {k: v["mark"] for k, v in d.items()} for lab, d in summ["q_d"].items()},
                                  q_a={m: {t: round(v["mae"], 4) for t, v in d.items()} for m, d in summ["q_a"].items()},
                                  n_common=len(ids))), indent=1, ensure_ascii=False))
    return 0


def cmd_selftest():
    L, Y, attr = EV.labels()
    frames, info = pair_frames(Y, with_abs=False)
    A = EV.assemble(frames, L, Y, attr)
    models = list(frames)
    ids = common_ids(A, models)
    M_ = EV.metrics(A, ids, models, sets=("P_main",))
    pm = M_[M_.pos == "p"]
    em = pd.read_csv(C.EM_METRICS)
    em = em[(em.eval_set == "P_main") & (em.pos == "p") & em.model.isin(models)]
    cols = ["n", "mae", "rmse", "r", "sign_acc", "n_sign", "mae_ci_lo", "mae_ci_hi"]
    met = []
    for m in models:
        for t in T3:
            a = pm[(pm.model == m) & (pm.target == t)].iloc[0]
            b = em[(em.model == m) & (em.target == t)].iloc[0]
            met.append(dict(model=m, target=t, mae=float(a.mae), mae_ch3m=float(b.mae),
                            match=round(float(a.mae), 4) == round(float(b.mae), 4),
                            other_cols_match={c: round(float(a[c]), 4) == round(float(b[c]), 4) for c in cols}))
    R_ = EV.rank_table(A, ids, models)
    er = pd.read_csv(C.EM_RANK)
    rcols = [c for c in R_.columns if c not in ("model", "scope", "rank_set")]
    rk = []
    for m in models:
        a = R_[(R_.model == m) & (R_.scope == "para")].iloc[0]
        b = er[(er.model == m) & (er.scope == "para")].iloc[0]
        dev = {c: (round(float(a[c]), 4), round(float(b[c]), 4)) for c in rcols}
        rk.append(dict(model=m, scope="para", match=all(x == y for x, y in dev.values()),
                       mismatches={c: v for c, v in dev.items() if v[0] != v[1]}))
    checks = {"EM_metrics P_main para MAE rows (GNN3 · TF3 · XGB3, 4 dp)": all(r["match"] for r in met),
              "EM_rank para rows (GNN3 · TF3 · XGB3, 4 dp)": all(r["match"] for r in rk)}
    rep = dict(spec="docs/specs/ABS3_MODELS.md 8-6 (a mismatch is a definition difference, not a STOP)", written=C.now(),
               all_pass=bool(all(checks.values())), checks=checks, metrics=met, rank=rk, n_common=len(ids),
               definitions=DEFINITIONS, code_sha256=code_sha(), pair_checks=info,
               inputs_sha256=dict(K3_predictions=C.sha256(C.K3_PRED), KM_predictions=C.sha256(C.KM_PRED),
                                  EM_metrics=C.sha256(C.EM_METRICS), EM_rank=C.sha256(C.EM_RANK), labels=C.sha256(C.LABELS),
                                  S_folds=C.sha256(EV.S_FOLDS)))
    C.write_json(C.RES / "EA_selftest.json", rep)
    print(json.dumps(C.clean(dict(all_pass=rep["all_pass"], checks=checks, n_common=len(ids),
                                  metrics=[(r["model"], r["target"], round(r["mae"], 4), r["match"]) for r in met],
                                  rank=[(r["model"], r["match"], r["mismatches"]) for r in rk])), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["selftest"]:
        sys.exit(cmd_selftest())
    if a == ["run"]:
        sys.exit(cmd_run())
    print(__doc__)
    sys.exit(2)
