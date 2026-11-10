#!/usr/bin/env python3
r"""eval_m.py — CH3M Phase E (docs/specs/CH3M_MODELS.md 8). CPU, env_espley (jobs/cpu_m.slurm CH3MENV=espley); no torch.

  python eval_m.py selftest   before K: (i) TF3 vs dep reproduces the 4 P_main para rows of ch3 E3_compare.csv (4 dp);
                              (ii) LADDER L_predictions g2 . S TF / GNN / XGB / AW summed to 3 channels reproduce the
                              spec 0-3 table (3 dp); a difference is reported (not a STOP) -> results/EM_selftest.json
  python eval_m.py run        7 models (dep, TF10, TF3, GNN10, GNN3, XGB10, XGB3) on their common test pairs ->
                              EM_3v10.csv (8-2), EM_noise.csv, EM_choice.csv (8-3), EM_cancel.csv (8-4), EM_metrics.csv,
                              EM_rank.csv, EM_cause.csv, EM_ad.json, EM_identity.csv (8-5), EM_report.json
                              (incl. choice.final_model for spec 9). Exit 0 written, 2 inputs missing, 3 STOP (y differs
                              from F_pairs_cr2 @ M, XGB candidate chosen = spec 13 stage 9).

The ch3 evaluation code is used unchanged (analysis/ch3/eval_ch3.py: labels, s_folds, lf_sum, ch3_frame, assemble,
set_mask, wide, boot_rel, compare, metrics, rank_table, cause, cause_rows, identity_residual, _clean); 10-target models are
summed with M (barrier = its direct prediction) exactly like lf_sum.
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
import ch3m_common as C  # noqa: E402
import eval_ch3 as EV  # noqa: E402                   (analysis/ch3, on sys.path through ch3m_common)

EG = EV.EG
T3, T10 = list(C.TARGETS3), list(C.TARGETS10)
KM = C.RES / "KM_predictions.parquet"
DEP_K = EV.DEP_K_PRED
K3 = EV.K3_PRED
L_PRED = C.LAD_RES / "L_predictions.parquet"
E3_COMPARE = C.CH3_RES / "E3_compare.csv"
NOISE_FLOOR = 0.02
REL_A, REL_B = 0.05, -0.05
TIE_ORDER = list(C.CANDIDATES)
GROUPS = dict(strain=["d1", "d2"], inter=["elst", "pauli", "oi", "disp", "c_ghost"], solv=["cpcm", "cds"],
              epo=["elst", "pauli", "oi"])
SPEC03 = {"g2 TF 10합": (0.360, 0.373, 0.294, 0.327), "g2 GNN 10합": (0.362, 0.368, 0.288, 0.325),
          "g2 AW 10합": (0.427, 0.404, 0.317, 0.385), "g2 XGB 10합": (0.436, 0.772, 0.309, 0.392),
          "dep (lf TF 10합)": (0.361, 0.383, 0.310, 0.330), "ch3 (lf TF 3직접)": (0.390, 0.382, 0.306, 0.349)}


def _defs():
    return {
        "models": "dep = deploy K_predictions (lf TF 10타깃, 원래 입력) @ M; TF10 / GNN10 / XGB10 = KM rung lf10 @ M; "
                  "TF3 = ch3 K3_predictions; GNN3 / XGB3 = KM rung lf3. barrier는 각 모델의 직접 예측",
        "pairs": "7개 모델이 모두 예측한 공통 시험 짝; 주 집합 = P_main(reference & in_P_main) · para",
        "q1": "EM_3v10: 모델마다 x3 대 x10합, 상대 차 r = (MAE(x3) − MAE(x10합)) / MAE(x10합), core 부트스트랩 2,000회 "
              "seed 20261019 95 % CI (eval_geo2.boot_diff 재표본, eval_ch3.boot_rel); P_main para(주) · all(보조) · m. "
              f"표시: 3 나음 = 상한 < 0 & r < −max(noise_t, {NOISE_FLOOR:g}); 10합 나음 = 하한 > 0 & r > +max(noise_t, "
              f"{NOISE_FLOOR:g}); 그 밖 같음. noise_t = |r(TF10 대 dep)|, 같은 위치(para 행은 P_main para, all 행은 "
              "P_main all)",
        "q2": f"EM_choice: 후보 {TIE_ORDER} 대 dep, P_main para r_t. 교체 = (a) 4타깃 모두 95 % 상한 ≤ +{REL_A * 100:g} % "
              f"그리고 (b) 2타깃 이상 r ≤ {REL_B * 100:g} % & 상한 < 0. 여럿이면 r_t 평균 최소, 동률 {TIE_ORDER} 순서. "
              "없으면 배포 모델 그대로",
        "cancel": "EM_cancel: 10타깃 모델(dep, TF10, GNN10, XGB10)의 오차 e_t = yhat_t − y_t; corr(e_pauli, e_elst + e_oi), "
                  "corr(e_d1, e_d2), corr(e_cpcm, e_cds); 묶음 상쇄 비율 = 평균 |Σ e| / 평균 Σ |e| (strain, inter, solv, "
                  "epo = elst + pauli + oi); Σ MAE(묶음 타깃) → MAE(묶음 합)",
        "aux": "EM_metrics / EM_rank / EM_cause / EM_identity: eval_ch3 정의 그대로(9-3 주 집합 P_sens_ref, 예측 자신의 "
               "barrier 부호); EM_ad: TF · GNN · dep 모델의 자기 타깃별 시험 짝 yhat_seed_sd 95 백분위(XGB 없음)",
    }


DEFINITIONS = _defs()


# ================================================================ inputs
def sum4(K, name, L, need_targets=T10):
    """a 10-target long prediction frame of one model -> 4-target frame (pair_id, fold, target, yhat, yhat_seed_sd NaN);
    y checked against the label file (STOP)."""
    K = K.copy()
    K["pair_id"] = K.pair_id.astype(str)
    if set(K.target) != set(need_targets) or K.duplicated(["pair_id", "target"]).any():
        C.stop(f"{name}: targets {sorted(set(K.target))} / duplicated rows")
    dy = {t: float(np.abs(K[K.target == t].y.to_numpy(float)
                          - L.loc[K[K.target == t].pair_id, f"dd_{t}"].to_numpy(float)).max()) for t in T10}
    if max(dy.values()) > 1e-12:
        C.stop(f"spec 13 (8): {name} y differs from F_pairs_cr2.csv: {dy}")
    W = K.pivot(index="pair_id", columns="target", values="yhat")[T10]
    fold = K.groupby("pair_id").fold.first()
    yh = W.to_numpy(float) @ C.M()
    rows = [pd.DataFrame(dict(pair_id=W.index, fold=fold.reindex(W.index).to_numpy(), target=t, yhat=yh[:, j],
                              yhat_seed_sd=np.nan)) for j, t in enumerate(T3)]
    return pd.concat(rows, ignore_index=True), dict(max_abs_dy_vs_labels=dy, n_pairs=int(len(W)))


def direct4(K, name, Y):
    K = K.copy()
    K["pair_id"] = K.pair_id.astype(str)
    if set(K.target) != set(T3) or K.duplicated(["pair_id", "target"]).any():
        C.stop(f"{name}: targets {sorted(set(K.target))} / duplicated rows")
    dy = {t: float(np.abs(K[K.target == t].y.to_numpy(float) - Y.loc[K[K.target == t].pair_id, t].to_numpy(float)).max())
          for t in T3}
    if max(dy.values()) > 1e-9:
        C.stop(f"spec 13 (8): {name} y differs from F_pairs_cr2 @ M: {dy}")
    return K[["pair_id", "fold", "target", "yhat", "yhat_seed_sd"]].copy(), dict(max_abs_dy_vs_labels=dy,
                                                                               n_pairs=int(K.pair_id.nunique()))


def load_all(L, Y):
    KMp = pd.read_parquet(KM)
    frames, info, raw10 = {}, {}, {}
    frames["dep"], info["dep"] = EV.lf_sum(L)
    raw10["dep"] = pd.read_parquet(DEP_K)
    for model in ("TF", "GNN", "XGB"):
        k10 = KMp[(KMp.rung == "lf10") & (KMp.model == model)]
        frames[f"{model}10"], info[f"{model}10"] = sum4(k10, f"{model}10", L)
        raw10[f"{model}10"] = k10
    frames["TF3"], info["TF3"] = EV.ch3_frame(Y)
    for model in ("GNN", "XGB"):
        frames[f"{model}3"], info[f"{model}3"] = direct4(KMp[(KMp.rung == "lf3") & (KMp.model == model)], f"{model}3", Y)
    return frames, info, raw10, KMp


def common_ids(A, models):
    s = None
    for m in models:
        ids = set(A[A.model == m].pair_id)
        s = ids if s is None else s & ids
    return sorted(s)


# ================================================================ tables
def cmp_rows(A, ids, x, b):
    """eval_ch3.compare (pos p / m / all; d, rel and CIs) without its ch3 adoption column."""
    Cm = EV.compare(A, ids, x=x, b=b)
    return Cm.drop(columns=["rule_pass", "decision"], errors="ignore")


def mark(r, lo, hi, noise):
    th = max(noise, NOISE_FLOOR)
    if hi < 0 and r < -th:
        return "3 나음"
    if lo > 0 and r > th:
        return "10합 나음"
    return "같음"


def q1(A, ids):
    """8-2; noise_t per position (para rows: TF10 vs dep on para; all rows: on all) — review 11-10, before K."""
    nz = cmp_rows(A, ids, "TF10", "dep")
    noise_pos = {(r.pos, r.target): abs(float(r.rel)) for r in nz.itertuples()}
    rows = []
    for model in ("TF", "GNN", "XGB"):
        c = cmp_rows(A, ids, f"{model}3", f"{model}10").assign(model=model)
        c["noise_t"] = [noise_pos.get((r.pos, r.target)) for r in c.itertuples()]
        c["mark"] = [mark(r.rel, r.rel_ci_lo, r.rel_ci_hi, noise_pos[(r.pos, r.target)]) if r.pos in ("p", "all") else None
                     for r in c.itertuples()]
        c["main"] = c.pos == "p"
        rows.append(c)
    noise = {t: noise_pos[("p", t)] for t in T3}
    return pd.concat(rows, ignore_index=True), nz, noise


def q2(A, ids):
    rows, per = [], {}
    for cand in TIE_ORDER:
        c = cmp_rows(A, ids, cand, "dep")
        p = c[c.pos == "p"].set_index("target")
        a_ok = bool(all(p.at[t, "rel_ci_hi"] <= REL_A for t in T3))
        nb = int(sum(1 for t in T3 if p.at[t, "rel"] <= REL_B and p.at[t, "rel_ci_hi"] < 0))
        per[cand] = dict(rel={t: float(p.at[t, "rel"]) for t in T3}, rel_ci_hi={t: float(p.at[t, "rel_ci_hi"]) for t in T3},
                         a_all_upper_le_5pct=a_ok, b_n_targets_le_minus5_and_upper_lt0=nb, passes=bool(a_ok and nb >= 2),
                         mean_rel=float(np.mean([p.at[t, "rel"] for t in T3])))
        rows.append(c.assign(candidate=cand, passes=per[cand]["passes"]))
    passed = [c for c in TIE_ORDER if per[c]["passes"]]
    chosen = min(passed, key=lambda c: (per[c]["mean_rel"], TIE_ORDER.index(c))) if passed else None
    return pd.concat(rows, ignore_index=True), per, chosen


def cancel(A_raw, models, P, sets=("P_main",), ids=None):
    """EM_cancel on the 10-target predictions (raw frames; ids = the common pairs when given)."""
    rows = []
    for m in models:
        K = A_raw[m].copy()
        K["pair_id"] = K.pair_id.astype(str)
        if ids is not None:
            K = K[K.pair_id.isin(set(ids))]
        W = K.pivot(index="pair_id", columns="target", values="yhat")[T10]
        Y = K.pivot(index="pair_id", columns="target", values="y")[T10]
        E = (W - Y)
        at = P.reindex(E.index)
        for es in sets:
            msk = EV.set_mask(at.assign(ref_kind=at.ref_kind, in_P_main=at.in_P_main, in_P_sens=at.in_P_sens), es)
            for pos in ("p", "all"):
                mm = msk & ((at.pos == pos).to_numpy() if pos != "all" else True)
                e = E[mm]
                r = dict(model=m, eval_set=es, pos=pos, n=int(len(e)),
                         corr_pauli_vs_elst_oi=float(np.corrcoef(e.pauli, e.elst + e.oi)[0, 1]),
                         corr_d1_d2=float(np.corrcoef(e.d1, e.d2)[0, 1]),
                         corr_cpcm_cds=float(np.corrcoef(e.cpcm, e.cds)[0, 1]))
                for g, ts in GROUPS.items():
                    s_abs = e[ts].abs().sum(1).mean()
                    abs_s = e[ts].sum(1).abs().mean()
                    r[f"{g}_sum_mae"] = float(s_abs)
                    r[f"{g}_mae_of_sum"] = float(abs_s)
                    r[f"{g}_cancel_ratio"] = float(abs_s / s_abs) if s_abs > 0 else np.nan
                rows.append(r)
    return pd.DataFrame(rows)


def ad(raws, P):
    out = {}
    for m, K in raws.items():
        if m.startswith("XGB"):
            continue
        out[m] = {t: float(np.percentile(g.yhat_seed_sd.to_numpy(float), EV.AD_PCT)) for t, g in K.groupby("target")}
    return dict(rule=f"{EV.AD_PCT:.0f}th percentile (numpy linear) of each model's test-pair yhat_seed_sd per own target "
                     "(SD over the 5 final seeds of each fold); XGB has one fit (none)", thresholds=out)


def code_sha():
    return {"analysis/ch3m/eval_m.py": C.sha256(Path(__file__).resolve()),
            "analysis/ch3m/ch3m_common.py": C.sha256(HERE / "ch3m_common.py"),
            "analysis/ch3/eval_ch3.py": C.sha256(C.CH3_DIR / "eval_ch3.py"),
            "analysis/geo2/eval_geo2.py": C.sha256(Path(EG.__file__).resolve())}


# ================================================================ commands
def cmd_run():
    if not KM.is_file():
        C.not_ready(f"{KM} missing (xgb_m.py collect)")
    kr = C.RES / "KM_report.json"
    if not kr.is_file():
        C.not_ready(f"{kr} missing (xgb_m.py collect)")
    k = C.read_json(kr)
    if not (k.get("C_XB") or {}).get("ok"):
        C.stop(f"spec 13 stage 7: C-XB failed in {kr.name}: {k.get('C_XB')}")
    if k.get("predictions_sha256") != C.sha256(KM):
        C.stop(f"{KM.name} changed after {kr.name}")
    import prereg_m as PR
    L, Y, attr = EV.labels()
    frames, info, raw10, KMp = load_all(L, Y)
    A = EV.assemble(frames, L, Y, attr)
    models = list(C.LABEL_MODELS)
    ids = common_ids(A, models)
    Q1, NZ, noise = q1(A, ids)
    Q2, per, chosen = q2(A, ids)
    S = EV.s_folds()
    raw_all = {**raw10, "TF3": pd.read_parquet(K3), **{f"{m}3": KMp[(KMp.rung == "lf3") & (KMp.model == m)] for m in ("GNN", "XGB")}}
    CA = cancel(raw10, ["dep", "TF10", "GNN10", "XGB10"], S[["ref_kind", "in_P_main", "in_P_sens", "pos"]].assign(
        in_P_main=S.in_P_main.astype(bool), in_P_sens=S.in_P_sens.astype(bool)), sets=("P_main",), ids=ids)
    rel_q1 = {m: {t: float(Q1[(Q1.model == m) & (Q1.pos == "p") & (Q1.target == t)].rel.iloc[0]) for t in T3}
              for m in ("TF", "GNN", "XGB")}
    CA["rel_3v10_para"] = CA.model.map(lambda m: json.dumps(rel_q1.get(m.replace("10", ""), {})) if m != "dep" else None)
    M_ = EV.metrics(A, ids, models)
    R_ = EV.rank_table(A, ids, models)
    CZ = EV.cause_rows([EV.cause(A, ids, m) for m in models])
    IDN = pd.DataFrame([EV.identity_residual(A, ids, m) for m in models])
    AD = ad(raw_all, S)
    for name, df in (("EM_3v10.csv", Q1), ("EM_noise.csv", NZ), ("EM_choice.csv", Q2), ("EM_cancel.csv", CA),
                     ("EM_metrics.csv", M_), ("EM_rank.csv", R_), ("EM_cause.csv", CZ), ("EM_identity.csv", IDN)):
        C.write_csv(C.RES / name, df)
    C.write_json(C.RES / "EM_ad.json", EV._clean(AD))
    s0 = C.read_json(C.RES / "S0_start.json")
    dep_present = bool(((s0.get("checks") or {}).get("1-8") or {}).get("all_present"))
    if chosen is None:
        final_model = None if dep_present else "TF10"
        action = "keep deploy lf TF 10-target sum" + ("" if dep_present else " (rebuild TF10 final model: deploy files absent)")
    elif chosen == "TF3":
        final_model, action = None, "replace by TF3 = ch3 final model (exists; record only)"
    elif chosen.startswith("GNN"):
        final_model, action = chosen, f"replace by {chosen}: train_m.py final {chosen} + I-1 / I-2"
    else:
        final_model, action = None, f"STOP: XGB candidate {chosen} passed (spec 9 / 13 stage 9)"
    pm = M_[(M_.eval_set == "P_main") & (M_.pos == "p")]
    summ = dict(para_mae={m: {t: float(g[g.target == t].mae.iloc[0]) for t in T3} for m, g in pm.groupby("model")},
                q1_marks={m: {t: Q1[(Q1.model == m) & (Q1.pos == "p") & (Q1.target == t)]["mark"].iloc[0] for t in T3}
                          for m in ("TF", "GNN", "XGB")},
                noise=noise, n_common=len(ids))
    rep = dict(spec="docs/specs/CH3M_MODELS.md 8", written=C.now(), definitions=DEFINITIONS, summary=summ,
               choice=dict(candidates=per, chosen=chosen, final_model=final_model, action=action,
                           deploy_model_files_present=dep_present, rule=DEFINITIONS["q2"]),
               checks=info, n_pairs=dict(common=len(ids), **{m: int(A[A.model == m].pair_id.nunique()) for m in models}),
               inputs_sha256=dict(KM_predictions=C.sha256(KM), deploy_K_predictions=C.sha256(DEP_K), K3_predictions=C.sha256(K3),
                                  labels=C.sha256(C.LABELS), S_folds=C.sha256(EV.S_FOLDS),
                                  KM_report=C.sha256(C.RES / "KM_report.json") if (C.RES / "KM_report.json").is_file() else None,
                                  prereg_md=C.sha256(PR.MD) if PR.MD.is_file() else None),
               outputs_sha256={f: C.sha256(C.RES / f) for f in ("EM_3v10.csv", "EM_noise.csv", "EM_choice.csv",
                                                                 "EM_cancel.csv", "EM_metrics.csv", "EM_rank.csv",
                                                                 "EM_cause.csv", "EM_identity.csv", "EM_ad.json")},
               code_sha256=code_sha())
    C.write_json(C.RES / "EM_report.json", EV._clean(rep))
    print(json.dumps(EV._clean(dict(q1=summ["q1_marks"], noise=noise, chosen=chosen, final_model=final_model,
                                    action=action, n_common=len(ids))), indent=1, ensure_ascii=False))
    if chosen is not None and chosen.startswith("XGB"):
        C.stop(f"XGB candidate {chosen} passed the replacement rule (spec 13 stage 9): report")
    return 0


def cmd_selftest():
    L, Y, attr = EV.labels()
    checks, detail = {}, {}
    dep, _ = EV.lf_sum(L)
    tf3, _ = EV.ch3_frame(Y)
    A = EV.assemble({"dep": dep, "TF3": tf3}, L, Y, attr)
    ids = common_ids(A, ["dep", "TF3"])
    c = cmp_rows(A, ids, "TF3", "dep")
    c = c[c.pos == "p"].set_index("target")
    e3 = pd.read_csv(E3_COMPARE)
    e3 = e3[e3.pos == "p"].set_index("target")
    cols = ["n", "mae_x", "mae_b", "d", "d_ci_lo", "d_ci_hi", "rel", "rel_ci_lo", "rel_ci_hi"]
    dev = {t: {k: abs(float(c.at[t, k]) - float(e3.at[t, k])) for k in cols} for t in T3}
    detail["i_E3_compare_max_abs_diff"] = {t: max(v.values()) for t, v in dev.items()}
    checks["(i) TF3 vs dep = ch3 E3_compare P_main para (4 dp)"] = all(
        round(float(c.at[t, k]), 4) == round(float(e3.at[t, k]), 4) for t in T3 for k in cols)
    Lp = pd.read_parquet(L_PRED)
    frames = {"dep": dep, "TF3": tf3}
    for m in ("TF", "GNN", "XGB", "AW"):
        q = Lp[(Lp.rung == "g2") & (Lp.arm == "S") & (Lp.model == m)]
        frames[f"g2{m}"], _ = sum4(q[["pair_id", "fold", "target", "y", "yhat"]], f"g2 {m}", L)
    A2 = EV.assemble(frames, L, Y, attr)
    ids2 = common_ids(A2, list(frames))
    M2 = EV.metrics(A2, ids2, list(frames), sets=("P_main",))
    pm = M2[(M2.pos == "p")]
    got = {m: tuple(round(float(pm[(pm.model == m) & (pm.target == t)].mae.iloc[0]), 3) for t in T3) for m in frames}
    names = {"g2TF": "g2 TF 10합", "g2GNN": "g2 GNN 10합", "g2AW": "g2 AW 10합", "g2XGB": "g2 XGB 10합",
             "dep": "dep (lf TF 10합)", "TF3": "ch3 (lf TF 3직접)"}
    table = [dict(row=names[m], spec=SPEC03[names[m]], value=got[m], match=got[m] == SPEC03[names[m]]) for m in frames]
    detail["ii_table"] = table
    detail["ii_n_pairs"] = int(pm[pm.model == "dep"].n.iloc[0]) if len(pm) else None
    checks["(ii) LADDER g2 sums + dep + ch3 = spec 0-3 table (3 dp)"] = all(r["match"] for r in table)
    raw = {f"g2{m}": Lp[(Lp.rung == "g2") & (Lp.arm == "S") & (Lp.model == m)] for m in ("TF", "GNN", "XGB")}
    S = EV.s_folds()
    CA = cancel(raw, list(raw), S[["ref_kind", "in_P_main", "in_P_sens", "pos"]].assign(
        in_P_main=S.in_P_main.astype(bool), in_P_sens=S.in_P_sens.astype(bool)))
    detail["cancel_g2_para"] = CA[CA.pos == "p"][["model", "n", "corr_pauli_vs_elst_oi", "corr_d1_d2", "epo_sum_mae",
                                                 "epo_mae_of_sum"]].to_dict("records")
    rep = dict(spec="docs/specs/CH3M_MODELS.md 8-6 (a mismatch is a definition difference, not a STOP)", written=C.now(),
               all_pass=bool(all(checks.values())), checks=checks, detail=detail, definitions=DEFINITIONS,
               code_sha256=code_sha(), inputs_sha256=dict(deploy_K_predictions=C.sha256(DEP_K), K3_predictions=C.sha256(K3),
                                                         L_predictions=C.sha256(L_PRED), E3_compare=C.sha256(E3_COMPARE),
                                                         labels=C.sha256(C.LABELS)))
    C.write_json(C.RES / "EM_selftest.json", EV._clean(rep))
    print(json.dumps(EV._clean(dict(all_pass=rep["all_pass"], checks=checks, table=table,
                                    cancel=detail["cancel_g2_para"])), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["selftest"]:
        sys.exit(cmd_selftest())
    if a == ["run"]:
        sys.exit(cmd_run())
    print(__doc__)
    sys.exit(2)
