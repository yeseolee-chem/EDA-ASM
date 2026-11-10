#!/usr/bin/env python3
r"""eval_ch3.py — CH3-MODEL Phase E: ch3 (K3 cross-validation predictions) vs lf_sum (deploy lf 8-channel predictions
summed to the 3 channels; barrier = the lf direct prediction) on the same pairs (docs/specs/CH3_MODEL.md 9).
CPU, env_espley (numpy, pandas, scipy) through jobs/cpu_ch3.slurm CH3ENV=espley. No torch import.

  python eval_ch3.py selftest   lf_sum alone: the spec 9-3 reference values (pair MAE, sign accuracy, dominant channel /
                                type agreement, single pair and cell mean) at 2 decimals, the compare machinery on
                                identical inputs (d = 0, CI [0, 0]); alternatives tried when a value differs (definition
                                difference: reported, not a STOP) -> results/E3_selftest.json
  python eval_ch3.py run        results/K3_predictions.parquet vs lf_sum -> results/E3_compare.csv (9-2 adoption rule),
                                E3_metrics.csv, E3_rank.csv, E3_cause.csv, E3_ad.json, E3_report.json.
                                Exit 0 = written (also when the adoption rule fails: not a STOP), 2 = inputs missing,
                                3 = STOP (deploy K_predictions y != F_pairs_cr2.csv, inconsistent inputs).

The same code evaluates both models (spec 5). Definitions: DEFINITIONS below (quoted in PREREG_CH3.md).
Reused: analysis/geo2/eval_geo2.py boot_diff; boot_rel = analysis/deploy/eval_tf.py boot_rel (copied: eval_tf imports
the deploy paths, this module must stay light).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3_common as C3  # noqa: E402

sys.path.insert(0, str(C3.REPO / "analysis" / "geo2"))
import eval_geo2 as EG  # noqa: E402

T3 = list(C3.TARGETS3)
T10 = list(C3.TARGETS10)
CH = ["strain", "inter", "solv"]
POSITIONS = ("all", "p", "m", "o", "direct")
DEP_K_PRED = C3.DEPLOY_RES / "K_predictions.parquet"
K3_PRED = C3.RES / "K3_predictions.parquet"
S_FOLDS = C3.REPO / "analysis" / "d1_pair" / "results" / "S_folds.csv"
SIGN_MIN = 1.0                                      # |y| >= 1 kcal/mol (sign accuracy, cause analysis)
CELL_MIN = 4                                        # pairs per cell
RANK_MIN = 8                                        # points per family incl. H
RANK_BINS = ((0.41, 1.0, "0.41-1"), (1.0, 2.0, "1-2"), (2.0, np.inf, ">2"))
AD_PCT = 95.0
REL_MAX = 0.05
# primary definitions (fixed in PREREG_CH3.md before any ch3 result). The 9-2 decision always uses P_main para.
# For the 9-3 auxiliary tables the selftest on lf_sum (job 1070605) reproduced 13 / 15 reference values with the reference
# test pairs (P_sens_ref = ref_all on the test pairs) and the model's own barrier sign, 11 / 15 with P_main
# (DECISIONS.md 9-3): P_sens_ref / own are the primary auxiliary definitions; P_main rows are written too.
METRIC_SET = "P_sens_ref"                           # E3_metrics primary auxiliary eval set (pair MAE all, sign accuracy)
METRIC_SETS = ("P_main", "P_sens_ref", "test_all")
CAUSE_SET = "P_sens_ref"                            # pairs of E3_cause (primary); P_main also written
CAUSE_SIGN = "own"                                  # predicted dominant channel / type: sign of the model's own barrier
RANK_SET = "P_sens_ref"                             # E3_rank families (spec 9-3: P_sens reference pairs)
SETS = {"P_main": "ref_kind == reference & in_P_main", "P_sens_ref": "ref_kind == reference & in_P_sens",
        "ref_all": "ref_kind == reference", "test_all": "every common test pair (reference + anchor)"}
# spec 9-3 reference values (lf_sum, 'this conversation')
REF = {"mae_all": dict(strain=0.81, inter=0.65, solv=0.52, barrier=0.75),
       "sign_acc": dict(strain=0.90, inter=0.93, solv=0.84),
       "dominant_single": dict(value=0.72, base=0.63), "dominant_cell": dict(value=0.84, base=0.71),
       "type_single": dict(value=0.69, base=0.45), "type_cell": dict(value=0.83, base=0.53)}


def _defs():
    return {
        "pairs": "공통 집합 = ch3(K3_predictions)와 lf_sum(deploy K_predictions)이 모두 예측한 시험 짝. 짝 속성 = "
                 "S_folds.csv(core, fam, sigma, in_P_main, in_P_sens, ref_kind, pos) + F_pairs_cr2.csv(dipole, cfg(빈 값 "
                 "'-'), series, pos, Xb); pos · ref_kind가 서로 다르면 STOP",
        "targets": "y(두 모델 공통) = F_pairs_cr2 dd_<10 targets> @ M; lf_sum yhat = deploy K_predictions yhat 10열 @ M "
                   "(strain = d1+d2, inter = elst+pauli+oi+disp+c_ghost, solv = cpcm+cds, barrier = lf 직접 예측)",
        "sets": "; ".join(f"{k}: {v}" for k, v in SETS.items()),
        "compare": "E3_compare: P_main · 위치 p(판정) 공통 짝, 타깃마다 d = MAE(ch3) - MAE(lf_sum) (eval_geo2.boot_diff, "
                   "core 2,000회, seed 20261019), 상대 차 r = d / MAE(lf_sum)와 같은 재표본의 95 % CI (eval_tf.boot_rel); "
                   "위치 m · all은 설명용",
        "adoption": f"strain · inter · solv: r ≤ 0 그리고 r 95 % 상한 ≤ +{REL_MAX * 100:.0f} %; barrier: r 95 % 상한 ≤ "
                    f"+{REL_MAX * 100:.0f} %. 모두 만족 → ch3 채택",
        "metrics": f"E3_metrics: 모델 × eval_set ({', '.join(METRIC_SETS)}; 9-3 짝 MAE · 부호의 주 집합 = {METRIC_SET}, "
                   "para MAE CI = P_main) × 타깃 × 위치(all, p, m, o, direct): n, MAE, RMSE, Pearson r, 부호 정답률"
                   "(|y| ≥ 1 짝에서 sign(yhat) = sign(y); yhat = 0은 오답), P_main · p의 MAE core 부트스트랩 95 % CI "
                   "(boot_diff(e, 0, core))",
        "aux_set_choice": "9-3 보조 지표(짝 MAE all, 부호, 원인)의 주 집합은 selftest(lf_sum)가 참고값을 가장 많이 재현한 "
                          "정의(P_sens_ref = reference & in_P_sens 공통 시험 짝, 예측 자신의 barrier 부호: 15개 중 13개)로 "
                          "고정했다. P_main 정의는 11/15. 유형 일치율 두 값(반응 하나 0.68 대 0.69, 칸 평균 0.82 대 0.83)은 "
                          "어느 후보로도 함께 재현되지 않아 정의 차이로 보고한다. 9-2 채택 판정은 명령서대로 P_main para.",
        "rank": f"E3_rank: 계열 = fam × pos, {RANK_SET} 짝({SETS[RANK_SET]}) 중 공통 짝, Xb 중복은 첫 행, H(y = yhat = 0) "
                f"추가, 점 ≥ {RANK_MIN}. 대상 = barrier. 두 작용기 순서: 계열 안 모든 점 쌍, 실제 |Δy| 구간 [0.41, 1), "
                "[1, 2), [2, ∞)에서 sign(Δyhat) = sign(Δy) 비율(쌍 합산). 최저: 실제 최저 점(argmin y, 첫째)의 예측 순위 "
                "(= 1 + 그보다 yhat이 작은 점 수)가 1 / ≤ 3인 계열 비율, 무작위 기대값 = 계열 평균 1/n, min(3, n)/n. "
                "Spearman(y, yhat) 계열별 중앙값 · 25 % 분위. 범위 para(pos p 계열)와 all(모든 위치 계열)",
        "cause_single": f"E3_cause 반응 하나: {CAUSE_SET} 공통 짝, |y_barrier| ≥ 1. 주도 채널 = argmax_c (v_c × "
                        "sign(v_barrier)), c ∈ (strain, inter, solv); 유형: s = sign(v_barrier), S = strain·s > 0 & "
                        "(inter+solv)·s ≤ 0, SI = 둘 다 > 0, I = strain·s ≤ 0 & (inter+solv)·s > 0, 그 밖 none. 실제는 y, "
                        f"예측은 yhat({'예측 자신의 barrier 부호' if CAUSE_SIGN == 'own' else '실제 barrier 부호'}). "
                        "일치율과 기준값(실제 범주 중 가장 흔한 범주의 비율)",
        "cause_cell": f"E3_cause 칸 평균: {CAUSE_SET} 공통 짝, 칸 = dipole × cfg × series × pos × Xb, 짝 ≥ {CELL_MIN}; "
                      "칸 평균 y · yhat로 |평균 y_barrier| ≥ 1인 칸에서 같은 두 일치율과 기준값; 모든 칸에서 타깃별 칸 "
                      "평균의 Pearson r · MAE",
        "ad": "E3_ad: 타깃별 ch3 시험 짝(K3 전부) yhat_seed_sd의 95 백분위(numpy linear)",
    }


DEFINITIONS = _defs()


# ================================================================ inputs
def labels():
    L = pd.read_csv(C3.LABELS)
    L["pair_id"] = L.pair_id.astype(str)
    L = L.set_index("pair_id")
    y4 = L[[f"dd_{t}" for t in T10]].to_numpy(float) @ C3.M()
    Y = pd.DataFrame(y4, index=L.index, columns=T3)
    attr = pd.DataFrame(dict(dipole=L.dipole.astype(str), cfg=L.cfg.fillna("").astype(str).replace("", "-"),
                             series=L.series.fillna("").astype(str), pos_l=L.pos.fillna("").astype(str),
                             Xb=L.Xb.fillna("").astype(str), ref_kind_l=L.ref_kind.fillna("").astype(str)),
                        index=L.index)
    return L, Y, attr


def s_folds():
    S = pd.read_csv(S_FOLDS)
    S["pair_id"] = S.pair_id.astype(str)
    S["pos"] = S.pos.fillna("").astype(str)
    S["ref_kind"] = S.ref_kind.fillna("").astype(str)
    return S[["pair_id", "core", "fam", "sigma", "in_P_main", "in_P_sens", "ref_kind", "pos"]].set_index("pair_id")


def lf_sum(L):
    """deploy K_predictions -> 4-target lf_sum frame (pair_id, fold, target, yhat); y checked against F_pairs_cr2."""
    K = pd.read_parquet(DEP_K_PRED)
    K["pair_id"] = K.pair_id.astype(str)
    if set(K.target) != set(T10) or K.duplicated(["pair_id", "target"]).any():
        C3.stop(f"{DEP_K_PRED}: targets {sorted(set(K.target))} / duplicated rows")
    miss = sorted(set(K.pair_id) - set(L.index))
    if miss:
        C3.stop(f"{DEP_K_PRED}: {len(miss)} pairs not in {C3.LABELS}")
    dy = {}
    for t in T10:
        q = K[K.target == t]
        dy[t] = float(np.abs(q.y.to_numpy(float) - L.loc[q.pair_id, f"dd_{t}"].to_numpy(float)).max())
    if max(dy.values()) > 1e-12:
        C3.stop(f"spec 15 (9): deploy K_predictions y differs from F_pairs_cr2.csv: {dy}")
    W = K.pivot(index="pair_id", columns="target", values="yhat")[T10]
    fold = K.groupby("pair_id").fold.first()
    yh = W.to_numpy(float) @ C3.M()
    rows = [pd.DataFrame(dict(pair_id=W.index, fold=fold.reindex(W.index).to_numpy(), target=t, yhat=yh[:, j],
                              yhat_seed_sd=np.nan)) for j, t in enumerate(T3)]
    return pd.concat(rows, ignore_index=True), dict(max_abs_dy_vs_labels=dy, n_pairs=int(len(W)))


def ch3_frame(Y):
    K = pd.read_parquet(K3_PRED)
    K["pair_id"] = K.pair_id.astype(str)
    if set(K.target) != set(T3) or K.duplicated(["pair_id", "target"]).any():
        C3.stop(f"{K3_PRED}: targets {sorted(set(K.target))} / duplicated rows")
    if set(K.rung) != {C3.RUNG3}:
        C3.stop(f"{K3_PRED}: rung {sorted(set(K.rung))}")
    dy = {}
    for t in T3:
        q = K[K.target == t]
        dy[t] = float(np.abs(q.y.to_numpy(float) - Y.loc[q.pair_id, t].to_numpy(float)).max())
    if max(dy.values()) > 1e-9:
        C3.stop(f"K3 y differs from F_pairs_cr2 @ M: {dy}")
    return K[["pair_id", "fold", "target", "yhat", "yhat_seed_sd"]].copy(), dict(max_abs_dy_vs_labels=dy,
                                                                               n_pairs=int(K.pair_id.nunique()))


def assemble(frames, L, Y, attr):
    """long frame: model, pair_id, fold, target, y, yhat, yhat_seed_sd + attributes (STOP on inconsistencies)."""
    S = s_folds()
    out = []
    for name, d in frames.items():
        d = d.copy()
        miss = sorted(set(d.pair_id) - set(S.index))
        if miss:
            C3.stop(f"{name}: {len(miss)} pairs not in S_folds.csv: {miss[:5]}")
        ri = Y.index.get_indexer(d.pair_id)
        if (ri < 0).any():
            C3.stop(f"{name}: pairs without labels")
        d["y"] = Y.to_numpy()[ri, np.array([T3.index(t) for t in d.target])]
        d["model"] = name
        out.append(d)
    A = pd.concat(out, ignore_index=True)
    A = A.join(S, on="pair_id").join(attr, on="pair_id")
    if (A.pos != A.pos_l).any() or (A.ref_kind != A.ref_kind_l).any():
        C3.stop("pos / ref_kind differ between S_folds.csv and F_pairs_cr2.csv")
    A = A.drop(columns=["pos_l", "ref_kind_l"])
    A["in_P_main"] = A.in_P_main.astype(bool)
    A["in_P_sens"] = A.in_P_sens.astype(bool)
    return A


def set_mask(d, es):
    ref = (d.ref_kind == "reference").to_numpy()
    if es == "P_main":
        return ref & d.in_P_main.to_numpy(bool)
    if es == "P_sens_ref":
        return ref & d.in_P_sens.to_numpy(bool)
    if es == "ref_all":
        return ref
    if es == "test_all":
        return np.ones(len(d), dtype=bool)
    C3.stop(f"unknown set {es}")


def wide(A, model, ids=None):
    """one row per pair: y_<t>, yhat_<t> + attributes."""
    d = A[A.model == model]
    if ids is not None:
        d = d[d.pair_id.isin(ids)]
    y = d.pivot(index="pair_id", columns="target", values="y")[T3].add_prefix("y_")
    yh = d.pivot(index="pair_id", columns="target", values="yhat")[T3].add_prefix("yhat_")
    at = d.drop_duplicates("pair_id").set_index("pair_id")[["core", "fam", "sigma", "in_P_main", "in_P_sens", "ref_kind",
                                                             "pos", "dipole", "cfg", "series", "Xb", "fold"]]
    return pd.concat([y, yh], axis=1).join(at)


# ================================================================ statistics
def boot_rel(dx, db, cores, n_boot=C3.N_BOOT, seed=C3.SEED):
    """= analysis/deploy/eval_tf.py boot_rel: r = (MAE(x) - MAE(b)) / MAE(b) with the eval_geo2.boot_diff resamples."""
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


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def sign_acc(y, yh, thr=SIGN_MIN):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    m = np.abs(y) >= thr
    return (float((np.sign(yh[m]) == np.sign(y[m])).mean()) if m.any() else np.nan), int(m.sum())


# ================================================================ tables
def compare(A, ids, x="ch3", b="lf_sum", thr=REL_MAX):
    rows = []
    for pos in ("p", "m", "all"):
        for t in T3:
            qx = A[(A.model == x) & (A.target == t) & A.pair_id.isin(ids)]
            qx = qx[set_mask(qx, "P_main") & ((qx.pos == pos).to_numpy() if pos != "all" else True)].set_index("pair_id")
            qb = A[(A.model == b) & (A.target == t)].set_index("pair_id").reindex(qx.index)
            if not len(qx):
                continue
            if qb.yhat.isna().any():
                C3.stop(f"compare: {b} lacks predictions on common pairs ({t}, {pos})")
            if np.abs(qx.y.to_numpy() - qb.y.to_numpy()).max() > 1e-12:
                C3.stop(f"compare: y differs between {x} and {b} ({t}, {pos})")
            ex, eb = (qx.yhat - qx.y).to_numpy(float), (qb.yhat - qb.y).to_numpy(float)
            cores = qx.core.to_numpy()
            d, lo, hi, p = EG.boot_diff(ex, eb, cores)
            r, rlo, rhi, (dlo, dhi) = boot_rel(ex, eb, cores)
            if abs(dlo - lo) > 1e-12 or abs(dhi - hi) > 1e-12:
                C3.stop(f"boot_rel resamples differ from eval_geo2.boot_diff ({t}, {pos})")
            dec = pos == "p"
            ok = None
            if dec:
                ok = bool(rhi <= thr) if t == "barrier" else bool(r <= 0 and rhi <= thr)
            rows.append(dict(x=x, b=b, pos=pos, target=t, decision=dec, n=int(len(qx)), n_cores=int(len(set(cores))),
                             mae_x=float(np.abs(ex).mean()), mae_b=float(np.abs(eb).mean()), d=d, d_ci_lo=lo,
                             d_ci_hi=hi, p=p, rel=r, rel_ci_lo=rlo, rel_ci_hi=rhi, rule_pass=ok))
    return pd.DataFrame(rows)


def adoption(Cm, thr=REL_MAX):
    c = Cm[(Cm.pos == "p")].set_index("target")
    per = {t: dict(rel=float(c.at[t, "rel"]), rel_ci_lo=float(c.at[t, "rel_ci_lo"]), rel_ci_hi=float(c.at[t, "rel_ci_hi"]),
                   mae_ch3=float(c.at[t, "mae_x"]), mae_lf_sum=float(c.at[t, "mae_b"]),
                   pass_=bool(c.at[t, "rule_pass"])) for t in T3 if t in c.index}
    miss = [t for t in T3 if t not in per]
    ok = not miss and all(v["pass_"] for v in per.values())
    return {"adopted": bool(ok), "threshold": thr, "missing": miss,
            "rule": f"strain/inter/solv: rel <= 0 and rel 95 % upper <= +{thr * 100:g} %; barrier: rel 95 % upper <= "
                    f"+{thr * 100:g} % (common P_main para pairs, core bootstrap 2,000, seed 20261019)",
            "n_pairs": int(c.n.iloc[0]) if len(c) else 0,
            "per_target": {t: {("pass" if k == "pass_" else k): v for k, v in r.items()} for t, r in per.items()},
            "failed": [t for t, v in per.items() if not v["pass_"]]}


def metrics(A, ids, models, sets=METRIC_SETS):
    rows = []
    for model in models:
        d0 = A[(A.model == model) & A.pair_id.isin(ids)]
        for es in dict.fromkeys(sets):
            d1 = d0[set_mask(d0, es)]
            for t in T3:
                q = d1[d1.target == t]
                e = (q.yhat - q.y).to_numpy(float)
                for p in POSITIONS:
                    m = np.ones(len(q), dtype=bool) if p == "all" else (q.pos == p).to_numpy()
                    if not m.any():
                        continue
                    sa, ns = sign_acc(q.y.to_numpy()[m], q.yhat.to_numpy()[m])
                    r = dict(model=model, eval_set=es, target=t, pos=p, n=int(m.sum()), mae=float(np.abs(e[m]).mean()),
                             rmse=float(np.sqrt((e[m] ** 2).mean())), r=pearson(q.y.to_numpy()[m], q.yhat.to_numpy()[m]),
                             sign_acc=sa, n_sign=ns)
                    if p == "p" and es == "P_main":
                        cores = q.core.to_numpy()[m]
                        _, lo, hi, _ = EG.boot_diff(e[m], np.zeros(int(m.sum())), cores)
                        r.update(n_cores=int(len(set(cores))), mae_ci_lo=lo, mae_ci_hi=hi)
                    rows.append(r)
    return pd.DataFrame(rows)


def rank_families(Wd, scope):
    """barrier points per family (fam x pos) + H(0, 0); >= RANK_MIN points."""
    out = []
    d = Wd if scope == "all" else Wd[Wd.pos == "p"]
    for (fam, pos), g in d.groupby(["fam", "pos"], sort=True):
        g = g.drop_duplicates("Xb")
        y = np.r_[g.y_barrier.to_numpy(float), 0.0]
        yh = np.r_[g.yhat_barrier.to_numpy(float), 0.0]
        if len(y) >= RANK_MIN:
            out.append((fam, pos, y, yh))
    return out


def rank_table(A, ids, models, rank_set=RANK_SET):
    from scipy.stats import spearmanr
    rows = []
    for model in models:
        Wd = wide(A, model, ids)
        Wd = Wd[set_mask(Wd, rank_set)]
        for scope in ("para", "all"):
            fams = rank_families(Wd, scope)
            cnt = {b[2]: [0, 0] for b in RANK_BINS}
            top1, top3, e1, e3, rho = [], [], [], [], []
            for fam, pos, y, yh in fams:
                n = len(y)
                for i, j in combinations(range(n), 2):
                    dy, dh = y[i] - y[j], yh[i] - yh[j]
                    a = abs(dy)
                    for lo, hi, lab in RANK_BINS:
                        if lo <= a < hi:
                            cnt[lab][0] += int(np.sign(dh) == np.sign(dy))
                            cnt[lab][1] += 1
                i0 = int(np.argmin(y))
                rk = 1 + int((yh < yh[i0]).sum())
                top1.append(rk == 1)
                top3.append(rk <= 3)
                e1.append(1.0 / n)
                e3.append(min(3, n) / n)
                rho.append(float(spearmanr(y, yh).correlation))
            r = dict(model=model, scope=scope, rank_set=rank_set, n_families=len(fams),
                     n_points=int(sum(len(f[2]) for f in fams)))
            for lab, (k, n) in cnt.items():
                r[f"order_acc_{lab}"] = k / n if n else np.nan
                r[f"order_n_{lab}"] = n
            rho = np.asarray(rho, float)
            r.update(lowest_top1=float(np.mean(top1)) if fams else np.nan,
                     lowest_top1_random=float(np.mean(e1)) if fams else np.nan,
                     lowest_top3=float(np.mean(top3)) if fams else np.nan,
                     lowest_top3_random=float(np.mean(e3)) if fams else np.nan,
                     spearman_median=float(np.nanmedian(rho)) if len(rho) else np.nan,
                     spearman_q25=float(np.nanpercentile(rho, 25)) if len(rho) else np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def dominant(v, s):
    """argmax_c (v_c x s) over CH (v: [n, 3] strain, inter, solv; s: [n] barrier sign) -> channel names."""
    return np.array(CH, dtype=object)[np.argmax(v * s[:, None], axis=1)]


def ctype(v, s):
    st = v[:, 0] * s > 0
    rt = (v[:, 1] + v[:, 2]) * s > 0
    out = np.full(len(s), "none", dtype=object)
    out[st & ~rt] = "S"
    out[st & rt] = "SI"
    out[~st & rt] = "I"
    return out


def agree(y, yh, sign_mode=CAUSE_SIGN):
    """dominant channel / type agreement on rows with |y_barrier| >= 1: y, yh = [n, 4] (strain, inter, solv, barrier)."""
    m = np.abs(y[:, 3]) >= SIGN_MIN
    y, yh = y[m], yh[m]
    s_true = np.sign(y[:, 3])
    if sign_mode == "own":
        s_pred = np.sign(yh[:, 3])
    elif sign_mode == "sum":
        s_pred = np.sign(yh[:, 0] + yh[:, 1] + yh[:, 2])
    elif sign_mode == "true":
        s_pred = s_true
    else:
        C3.stop(f"unknown sign mode {sign_mode}")
    out = dict(n=int(m.sum()))
    for key, f in (("dominant", dominant), ("type", ctype)):
        a, b = f(y[:, :3], s_true), f(yh[:, :3], s_pred)
        vc = pd.Series(a).value_counts(normalize=True)
        out[key] = float((a == b).mean()) if len(a) else np.nan
        out[f"{key}_base"] = float(vc.iloc[0]) if len(vc) else np.nan
        out[f"{key}_base_class"] = str(vc.index[0]) if len(vc) else None
        out[f"{key}_true_dist"] = {str(k): float(v) for k, v in vc.items()}
    return out


def cause(A, ids, model, cause_set=CAUSE_SET, sign_mode=CAUSE_SIGN):
    Wd = wide(A, model, ids)
    Wd = Wd[set_mask(Wd, cause_set)]
    yc, hc = [f"y_{t}" for t in T3], [f"yhat_{t}" for t in T3]
    single = agree(Wd[yc].to_numpy(float), Wd[hc].to_numpy(float), sign_mode)
    keys = ["dipole", "cfg", "series", "pos", "Xb"]
    g = Wd.groupby(keys)
    n = g.size()
    cm = g[yc + hc].mean()[n >= CELL_MIN]
    cell = agree(cm[yc].to_numpy(float), cm[hc].to_numpy(float), sign_mode)
    cell["n_cells_all"] = int(len(cm))
    cellfit = {t: dict(r=pearson(cm[f"y_{t}"], cm[f"yhat_{t}"]),
                       mae=float(np.abs(cm[f"yhat_{t}"] - cm[f"y_{t}"]).mean()) if len(cm) else np.nan) for t in T3}
    return dict(model=model, cause_set=cause_set, sign_mode=sign_mode, n_pairs=int(len(Wd)), single=single, cell=cell,
                cell_fit=cellfit)


def cause_rows(res):
    rows = []
    for r in res:
        for level in ("single", "cell"):
            q = r[level]
            for key in ("dominant", "type"):
                rows.append(dict(model=r["model"], level=level, measure=key, cause_set=r["cause_set"],
                                 sign_mode=r["sign_mode"], n=q["n"], agreement=q[key], baseline=q[f"{key}_base"],
                                 baseline_class=q[f"{key}_base_class"], true_dist=json.dumps(q[f"{key}_true_dist"])))
        for t, v in r["cell_fit"].items():
            rows.append(dict(model=r["model"], level="cell_all", measure=f"cell_mean_{t}", cause_set=r["cause_set"],
                             sign_mode=r["sign_mode"], n=r["cell"]["n_cells_all"], r=v["r"], mae=v["mae"]))
    return pd.DataFrame(rows)


def ad_thresholds(A, model="ch3"):
    d = A[A.model == model]
    th = {t: float(np.percentile(d[d.target == t].yhat_seed_sd.to_numpy(float), AD_PCT)) for t in T3}
    return dict(rule=f"{AD_PCT:.0f}th percentile (numpy linear) of the {model} test-pair yhat_seed_sd per target "
                     "(SD over the 5 final seeds of each fold); out_of_range = yhat_sd > threshold",
                thresholds=th, n_pairs=int(d.pair_id.nunique()), model=model)


def identity_residual(A, ids, model):
    Wd = wide(A, model, ids)
    r = Wd.yhat_barrier - (Wd.yhat_strain + Wd.yhat_inter + Wd.yhat_solv)
    ry = Wd.y_barrier - (Wd.y_strain + Wd.y_inter + Wd.y_solv)
    return dict(model=model, yhat_mean_abs=float(r.abs().mean()), yhat_p95_abs=float(np.percentile(r.abs(), 95)),
                yhat_max_abs=float(r.abs().max()), y_max_abs=float(ry.abs().max()), n=int(len(Wd)))


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def code_sha():
    return {"analysis/ch3/eval_ch3.py": C3.sha256(Path(__file__).resolve()),
            "analysis/ch3/ch3_common.py": C3.sha256(HERE / "ch3_common.py"),
            "analysis/geo2/eval_geo2.py": C3.sha256(Path(EG.__file__).resolve())}


# ================================================================ commands
def cmd_run():
    if not K3_PRED.is_file():
        C3.not_ready(f"{K3_PRED} missing (train_ch3.py collect)")
    import prereg_ch3 as PR
    thr = PR.rel_threshold()
    L, Y, attr = labels()
    lfs, lf_info = lf_sum(L)
    k3, k3_info = ch3_frame(Y)
    A = assemble({"ch3": k3, "lf_sum": lfs}, L, Y, attr)
    own = {m: set(A[A.model == m].pair_id) for m in ("ch3", "lf_sum")}
    common = sorted(own["ch3"] & own["lf_sum"])
    Cm = compare(A, common, thr=thr)
    ad_ = adoption(Cm, thr)
    M = metrics(A, common, ("ch3", "lf_sum"))
    R = rank_table(A, common, ("ch3", "lf_sum"))
    cz = [cause(A, common, m) for m in ("ch3", "lf_sum")]
    cz_main = [cause(A, common, m, "P_main") for m in ("ch3", "lf_sum")] if CAUSE_SET != "P_main" else []
    CZ = cause_rows(cz + cz_main).assign(primary=lambda d: d.cause_set == CAUSE_SET)
    ad = ad_thresholds(A, "ch3")
    C3.write_csv(C3.RES / "E3_compare.csv", Cm)
    C3.write_csv(C3.RES / "E3_metrics.csv", M)
    C3.write_csv(C3.RES / "E3_rank.csv", R)
    C3.write_csv(C3.RES / "E3_cause.csv", CZ)
    C3.write_json(C3.RES / "E3_ad.json", _clean(ad))
    pm = M[(M.eval_set == METRIC_SET)]
    pp = M[(M.eval_set == "P_main")]
    summ = dict(
        aux_set=METRIC_SET,
        para_mae_P_main={m: {t: float(g[(g.target == t) & (g.pos == "p")].mae.iloc[0]) for t in T3}
                         for m, g in pp.groupby("model")},
        all_mae={m: {t: float(g[(g.target == t) & (g.pos == "all")].mae.iloc[0]) for t in T3} for m, g in pm.groupby("model")},
        sign_acc_all={m: {t: g[(g.target == t) & (g.pos == "all")].sign_acc.iloc[0] for t in T3}
                      for m, g in pm.groupby("model")},
        rank={f"{r.model}/{r.scope}": {k: v for k, v in r.items() if k not in ("model", "scope")} for _, r in R.iterrows()},
        cause={c["model"]: dict(single=dict((k, c["single"][k]) for k in ("n", "dominant", "dominant_base", "type", "type_base")),
                                cell=dict((k, c["cell"][k]) for k in ("n", "dominant", "dominant_base", "type", "type_base")),
                                cell_fit=c["cell_fit"]) for c in cz},
        identity_residual={m: identity_residual(A, common, m) for m in ("ch3", "lf_sum")})
    rep = dict(spec="docs/specs/CH3_MODEL.md 9", written=C3.now(), adoption=ad_,
               compare_para=Cm[Cm.pos == "p"].to_dict("records"), summary=summ, ad=ad, definitions=DEFINITIONS,
               n_pairs=dict(ch3=len(own["ch3"]), lf_sum=len(own["lf_sum"]), common=len(common)),
               checks=dict(lf_sum=lf_info, ch3=k3_info),
               inputs_sha256=dict(K3_predictions=C3.sha256(K3_PRED), deploy_K_predictions=C3.sha256(DEP_K_PRED),
                                  labels=C3.sha256(C3.LABELS), S_folds=C3.sha256(S_FOLDS),
                                  K3_report=C3.sha256(C3.RES / "K3_report.json") if (C3.RES / "K3_report.json").is_file()
                                  else None),
               outputs_sha256={f: C3.sha256(C3.RES / f) for f in ("E3_compare.csv", "E3_metrics.csv", "E3_rank.csv",
                                                                   "E3_cause.csv", "E3_ad.json")},
               code_sha256=code_sha())
    C3.write_json(C3.RES / "E3_report.json", _clean(rep))
    print(json.dumps(_clean(dict(adopted=ad_["adopted"], failed=ad_["failed"], per_target=ad_["per_target"],
                                 n_pairs=rep["n_pairs"])), indent=1))
    return 0


def _match(v, ref):
    return v is not None and np.isfinite(v) and abs(round(float(v), 2) - ref) < 1e-9


def cmd_selftest():
    L, Y, attr = labels()
    lfs, lf_info = lf_sum(L)
    A = assemble({"lf_sum": lfs}, L, Y, attr)
    ids = sorted(A.pair_id.unique())
    checks, notes, table = {}, [], []
    # identical inputs through the compare machinery
    B = pd.concat([A, A.assign(model="ch3")], ignore_index=True)
    Cz = compare(B, ids)
    ad0 = adoption(Cz)
    checks["identical inputs: d = 0, CI [0, 0], r = 0, adopted"] = bool(
        (Cz.d.abs() == 0).all() and (Cz.d_ci_lo == 0).all() and (Cz.d_ci_hi == 0).all() and (Cz.rel.abs() == 0).all()
        and ad0["adopted"])
    # reference values with the primary definitions; alternatives when a value differs
    alt_sets = ("P_main", "P_sens_ref", "ref_all", "test_all")
    Mall = metrics(A, ids, ("lf_sum",), sets=alt_sets)
    for t, ref in REF["mae_all"].items():
        v = float(Mall[(Mall.eval_set == METRIC_SET) & (Mall.target == t) & (Mall.pos == "all")].mae.iloc[0])
        alts = {es: float(Mall[(Mall.eval_set == es) & (Mall.target == t) & (Mall.pos == "all")].mae.iloc[0])
                for es in alt_sets}
        table.append(dict(metric=f"짝 MAE all {t}", ref=ref, value=round(v, 4), match=_match(v, ref),
                          alternatives={k: round(x, 4) for k, x in alts.items()},
                          matching=[k for k, x in alts.items() if _match(x, ref)]))
    for t, ref in REF["sign_acc"].items():
        v = float(Mall[(Mall.eval_set == METRIC_SET) & (Mall.target == t) & (Mall.pos == "all")].sign_acc.iloc[0])
        alts = {es: float(Mall[(Mall.eval_set == es) & (Mall.target == t) & (Mall.pos == "all")].sign_acc.iloc[0])
                for es in alt_sets}
        table.append(dict(metric=f"부호 정답률 {t}", ref=ref, value=round(v, 4), match=_match(v, ref),
                          alternatives={k: round(x, 4) for k, x in alts.items()},
                          matching=[k for k, x in alts.items() if _match(x, ref)]))
    variants = {(cs, sm): cause(A, ids, "lf_sum", cs, sm) for cs in alt_sets for sm in ("own", "true", "sum")}
    prim = variants[(CAUSE_SET, CAUSE_SIGN)]
    for key, lev, name in (("dominant", "single", "주도 채널 반응 하나"), ("dominant", "cell", "주도 채널 칸 평균"),
                           ("type", "single", "유형 반응 하나"), ("type", "cell", "유형 칸 평균")):
        ref = REF[f"{key}_{lev}"]
        for part, rv in (("", ref["value"]), ("_base", ref["base"])):
            v = prim[lev][f"{key}{part}"]
            alts = {f"{cs}/{sm}": c[lev][f"{key}{part}"] for (cs, sm), c in variants.items()}
            table.append(dict(metric=f"{name}{' 기준값' if part else ''}", ref=rv, value=round(float(v), 4),
                              match=_match(v, rv), alternatives={k: round(float(x), 4) for k, x in alts.items()},
                              matching=[k for k, x in alts.items() if _match(x, rv)]))
    combo = {}
    for cs in alt_sets:
        for sm in ("own", "true", "sum"):
            hit = [r["metric"] for r in table
                   if (f"{cs}/{sm}" in r["matching"]) or (cs in r["matching"] and "/" not in "".join(r["alternatives"]))]
            combo[f"{cs}/{sm}"] = dict(n_match=len(hit), of=len(table),
                                       missed=[r["metric"] for r in table if r["metric"] not in hit])
    n_match = sum(r["match"] for r in table)
    checks[f"9-3 reference values reproduced at 2 dp with the primary definitions ({n_match}/{len(table)})"] = \
        n_match == len(table)
    for r in table:
        if not r["match"]:
            notes.append(f"{r['metric']}: 참고 {r['ref']}, 주 정의 {r['value']}; 일치하는 다른 정의 {r['matching'] or '없음'}.")
    rep = dict(spec="docs/specs/CH3_MODEL.md 9-3 (selftest; a mismatch is a definition difference, not a STOP)",
               written=C3.now(), all_pass=bool(all(checks.values())), checks=checks, reference_table=table, notes=notes,
               combo_matches=combo,
               definitions=DEFINITIONS, primary=dict(metric_set=METRIC_SET, cause_set=CAUSE_SET, cause_sign=CAUSE_SIGN,
                                                     rank_set=RANK_SET),
               lf_sum_rank=rank_table(A, ids, ("lf_sum",)).to_dict("records"),
               lf_sum_cause_primary=prim, lf_info=lf_info, n_pairs=len(ids),
               identity_residual_lf_sum=identity_residual(A, ids, "lf_sum"),
               code_sha256=code_sha(), inputs_sha256=dict(deploy_K_predictions=C3.sha256(DEP_K_PRED),
                                                         labels=C3.sha256(C3.LABELS), S_folds=C3.sha256(S_FOLDS)))
    C3.write_json(C3.RES / "E3_selftest.json", _clean(rep))
    print(json.dumps(_clean(dict(all_pass=rep["all_pass"], checks=checks, combo_matches=combo,
                                 table=[{k: r[k] for k in ("metric", "ref", "value", "match", "matching")} for r in table])),
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["selftest"]:
        sys.exit(cmd_selftest())
    if a == ["run"]:
        sys.exit(cmd_run())
    print(__doc__)
    sys.exit(2)
