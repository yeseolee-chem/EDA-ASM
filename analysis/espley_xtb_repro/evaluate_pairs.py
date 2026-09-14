"""evaluate_pairs.py — f 모델의 MMP-Δ 성능과 margin gate 보정을 분할 시나리오별로 계산.

python evaluate_pairs.py <repo_results_dir> <coley_full_dataset.csv>
출력: split_metrics.csv, margin_calibration.csv
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mmp_utils import annotate, build_pairs, splits

RES, CSV = sys.argv[1], sys.argv[2]
feat = annotate(pd.read_parquet(RES + "/xtb_features.parquet"), CSV)
pairs = build_pairs(feat)
print(f"pairs {len(pairs)}  {pairs.kind.value_counts().to_dict()}")
src = open(RES + "/../train_ml_single.py").read(); ns = {}
exec(src[src.index("DIST11"):src.index("TARGETS =")], ns)
X = feat[ns["FEATURE_SETS"]["ESPLEY72"]].values
CH = {"d1": "dft_d1_kcal", "d2": "dft_d2_kcal", "elst": "dft_elst_dft", "pauli": "dft_pauli_dft",
      "oi": "dft_oi_dft", "cpcm": "dft_cpcm_dft"}
EXACT = {"disp": "dft_disp_dft"}                      # b_disp == target, 예측 대상 아님
pos = {r: i for i, r in enumerate(feat.rxn_id)}
i1, i2 = pairs.r1.map(pos).values, pairs.r2.map(pos).values


def krr():
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", alpha=1e-3, gamma=1e-3)),
        transformer=StandardScaler())


def oof_predict(folds, y):
    p = np.full_like(y, np.nan); seen = np.zeros(len(y), bool)
    for tr, te in folds:
        m = krr().fit(X[tr], y[tr]); p[te] = m.predict(X[te]); seen[te] = True
        if len(folds) == 1:                         # loso: train 멤버는 in-sample 예측(낙관적, 하한 명시)
            p[tr] = m.predict(X[tr])
    return p, seen


metrics, calib = [], []
for name, folds in splits(feat, pairs):
    P = {}
    for c, tg in CH.items():
        P[c], seen = oof_predict(folds, feat[tg].values)
    P["disp"] = feat[EXACT["disp"]].values.copy()
    # 평가 쌍: 두 멤버 모두 OOF(random/class) 또는 to-멤버가 test(loso)
    if name.startswith("loso"):
        sub = name.split(":", 1)[1]
        te_set = set(np.where(seen)[0])
        ok = np.array([(a in te_set) ^ (b in te_set) for a, b in zip(i1, i2)])
    else:
        fold_of = np.zeros(len(feat), int)
        for k, (_, te) in enumerate(folds): fold_of[te] = k
        ok = fold_of[i1] == fold_of[i2]
    if ok.sum() < 30:
        continue
    a, b = i1[ok], i2[ok]
    chans = list(CH) + ["disp"]
    Dt = np.stack([feat[(CH | EXACT)[c]].values[b] - feat[(CH | EXACT)[c]].values[a] for c in chans], 1)
    Dp = np.stack([P[c][b] - P[c][a] for c in chans], 1)
    row = dict(split=name, n_pairs=int(ok.sum()))
    for j, c in enumerate(chans):
        if c == "disp": continue
        e = P[c] - feat[CH[c]].values
        row[f"mae_{c}"] = np.nanmean(np.abs(e[seen])); row[f"dmae_{c}"] = np.mean(np.abs(Dp[:, j] - Dt[:, j]))
    dom_t, dom_p = np.argmax(np.abs(Dt), 1), np.argmax(np.abs(Dp), 1)
    row["dom_agree_all"] = np.mean(dom_t == dom_p)
    # margin gate — 배포 상황엔 예측 margin만 있다
    sp = np.sort(np.abs(Dp), 1); m_pred = sp[:, -1] - sp[:, -2]
    st = np.sort(np.abs(Dt), 1); m_true = st[:, -1] - st[:, -2]
    for tau in np.arange(0, 8.01, 0.5):
        g = m_pred >= tau
        calib.append(dict(split=name, tau=tau, coverage=g.mean(), agree=np.mean(dom_t[g] == dom_p[g]) if g.any() else np.nan,
                          agree_true_margin=np.mean(dom_t[m_true >= tau] == dom_p[m_true >= tau]) if (m_true >= tau).any() else np.nan))
    for target_acc in (0.95, 0.99):
        taus = [r["tau"] for r in calib if r["split"] == name and r["agree"] >= target_acc]
        row[f"tau_{int(target_acc*100)}"] = min(taus) if taus else np.nan
        row[f"cov_{int(target_acc*100)}"] = np.mean(m_pred >= min(taus)) if taus else np.nan
    metrics.append(row)
    print(f"{name:22s} n={row['n_pairs']:5d} dMAE pauli {row['dmae_pauli']:.2f} elst {row['dmae_elst']:.2f} "
          f"oi {row['dmae_oi']:.2f} | dom {row['dom_agree_all']:.3f} | tau95 {row['tau_95']} (cov {row['cov_95']:.2f}) "
          f"tau99 {row['tau_99']} (cov {row['cov_99']:.2f})", flush=True)

pd.DataFrame(metrics).round(3).to_csv("split_metrics.csv", index=False)
pd.DataFrame(calib).round(3).to_csv("margin_calibration.csv", index=False)
pairs.to_csv("mmp_pairs_v2.csv", index=False)
print("wrote split_metrics.csv, margin_calibration.csv, mmp_pairs_v2.csv")
