"""evaluate_pairs.py — f 모델의 MMP-Δ 성능과 margin gate 보정을 분할 시나리오별로 계산 (기하 하나씩, rev 4).

python evaluate_pairs.py [--geom g1] [--feat PARQUET] [--rows CSV] [--csv COLEY_CSV] [--out DIR]
기본값: $ESPLEY_GEOM, $ESPLEY_FEAT, $ESPLEY_ROWS (results_rev4/rows_rev4.csv), $ESPLEY_COLEY_CSV, results_rev4/downstream_<geom>/
출력: split_metrics_<geom>.csv, margin_calibration_<geom>.csv, mmp_pairs_v2_<geom>.csv (모두 있으면 건너뜀)
행 = --rows. disp: G0에서는 b_disp == 라벨이라 참값을 그대로 쓰고(예측 대상 아님), G1에서는 다른 채널처럼 OOF 예측
(데이터로 확인, 어긋나면 중단).
"""
import argparse, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mmp_utils import annotate, build_pairs, splits
from train_ml_single import FEATURE_SETS

HERE = Path(__file__).resolve().parent
CSV = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")
FEAT_TAG = {"dft": "g0", "g1": "g1", "g2": "g2"}      # parquet `geom` column -> geometry label
DISP_IDENTITY_TOL = 1e-3                              # kcal/mol; G0 MAE(b_disp − dft_disp_dft) = 2.4e-6

env = os.environ.get
ap = argparse.ArgumentParser()
ap.add_argument("--geom", default=env("ESPLEY_GEOM"), help="g0 | g1 | g2 (output label)")
ap.add_argument("--feat", type=Path, default=env("ESPLEY_FEAT"))
ap.add_argument("--rows", type=Path, default=env("ESPLEY_ROWS", HERE / "results_rev4" / "rows_rev4.csv"))
ap.add_argument("--csv", type=Path, default=env("ESPLEY_COLEY_CSV", CSV))
ap.add_argument("--out", type=Path, default=None, help="default results_rev4/downstream_<geom>/")
A = ap.parse_args()
if A.geom not in FEAT_TAG.values() or A.feat is None:
    ap.error("need --geom g0|g1|g2 and --feat (or ESPLEY_GEOM, ESPLEY_FEAT)")
OUT = A.out or HERE / "results_rev4" / f"downstream_{A.geom}"
OUT.mkdir(parents=True, exist_ok=True)
OUTS = {k: OUT / f"{k}_{A.geom}.csv" for k in ("split_metrics", "margin_calibration", "mmp_pairs_v2")}
if all(p.exists() for p in OUTS.values()):
    print(f"{', '.join(p.name for p in OUTS.values())} exist in {OUT} — skip")
    sys.exit(0)

F = FEATURE_SETS["ESPLEY73"]
CH = {"d1": "dft_d1_kcal", "d2": "dft_d2_kcal", "elst": "dft_elst_dft", "pauli": "dft_pauli_dft",
      "oi": "dft_oi_dft", "cpcm": "dft_cpcm_dft"}
rows = pd.read_csv(A.rows)[["rxn_id"]]
feat = rows.merge(pd.read_parquet(A.feat), on="rxn_id", how="left", validate="1:1")
bad = (feat.xtb_status != "ok") | feat[list(dict.fromkeys(F + list(CH.values()) + ["dft_disp_dft"]))].isna().any(axis=1)
if bad.any():
    sys.exit(f"{int(bad.sum())} of {len(feat)} rows of {A.rows} are missing / not ok / NaN in {A.feat}: "
             f"{feat.rxn_id[bad].tolist()[:10]}")
tags = set(feat.geom) if "geom" in feat else {"<no geom column: rev 3 parquet?>"}
if {FEAT_TAG.get(t) for t in tags} != {A.geom}:
    sys.exit(f"{A.feat} has geom {sorted(tags)}, not --geom {A.geom}")
disp_mae = float((feat.b_disp - feat.dft_disp_dft).abs().mean())
if (disp_mae < DISP_IDENTITY_TOL) != (A.geom == "g0"):  # pre-registered: identity at G0 only (REV4 §3-1)
    sys.exit(f"MAE(b_disp − dft_disp_dft) {disp_mae:.2e} inconsistent with --geom {A.geom}")
if A.geom == "g0":
    EXACT = {"disp": "dft_disp_dft"}                  # b_disp == target (G0), 예측 대상 아님
else:
    EXACT = {}
    CH["disp"] = "dft_disp_dft"                       # G1: disp도 OOF 예측
print(f"geom {A.geom}: {len(feat)} rows ({A.rows}), feat {A.feat}\n"
      f"MAE(b_disp − dft_disp_dft) {disp_mae:.2e} -> disp {'= label (exact)' if EXACT else 'predicted'}", flush=True)

feat = annotate(feat, A.csv)
pairs = build_pairs(feat)
print(f"pairs {len(pairs)}  {pairs.kind.value_counts().to_dict()}")
X = feat[F].values
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
TG = CH | EXACT
chans = list(TG)
for name, folds in splits(feat, pairs):
    P = {}
    for c, tg in CH.items():
        P[c], seen = oof_predict(folds, feat[tg].values)
    for c, tg in EXACT.items():
        P[c] = feat[tg].values.copy()
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
    Dt = np.stack([feat[TG[c]].values[b] - feat[TG[c]].values[a] for c in chans], 1)
    Dp = np.stack([P[c][b] - P[c][a] for c in chans], 1)
    row = dict(split=name, n_pairs=int(ok.sum()))
    for j, c in enumerate(chans):
        if c in EXACT: continue
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

for key, df in (("split_metrics", pd.DataFrame(metrics).round(3)), ("margin_calibration", pd.DataFrame(calib).round(3)),
                ("mmp_pairs_v2", pairs)):
    tmp = OUTS[key].with_name(OUTS[key].name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, OUTS[key])
print(f"wrote {', '.join(str(p) for p in OUTS.values())}")
