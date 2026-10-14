"""evaluate_pairs.py — f 모델의 MMP-Δ 성능과 margin gate 보정을 분할 시나리오별로 계산 (기하 하나, arm 하나).

python evaluate_pairs.py [--geom g1] [--feat PARQUET] [--rows CSV] [--csv COLEY_CSV] [--out DIR]
기본값: $ESPLEY_GEOM, $ESPLEY_FEAT, $ESPLEY_ROWS, $ESPLEY_COLEY_CSV;
  ESPLEY_RES5=1: rows results_rev5/rows_rev5.csv, 출력 results_rev5/downstream_<arm>/
  그 외:         rows results_rev4/rows_rev4.csv, 출력 results_rev4/downstream_<geom>/ (arm ESPLEY73만; 다른 arm은 --out 필요)
arm = $ESPLEY_DOWNSTREAM_ARM (기본 ESPLEY73), 열 = train_ml_single.resolve_arm(arm). 확장 블록(B1..B6)을 쓰는 arm이면
rev5_common.FEAT_EXT를 rxn_id로 합친다 (모든 행 ext_status ok). EXT_SEL(또는 prereg_rev5b.json이 생긴 뒤의 확장 arm)이면
--feat·블록 파일·(rows_rev5.csv인) --rows의 sha256이 prereg_rev5b.json `inputs`(Phase C 선별 입력)와 같아야 한다
(train_ml_single.check_prereg_inputs, 건너뛰기 전에 확인). 모델 = KRR_rbf (rev 4와 같은 고정 HP alpha 1e-3,
gamma 1e-3, OOF). 행 = --rows. 채널 7개(d1, d2, elst, pauli, oi, cpcm, disp)를 모두 OOF 예측한다.
출력: split_metrics_<geom>.csv, margin_calibration_<geom>.csv, mmp_pairs_v2_<geom>.csv (모두 있으면 건너뜀)
      + evaluate_pairs_<geom>.json (입력 기록)
dom_agree_all (쌍별 |Δ|가 가장 큰 채널의 참/예측 일치율)의 분할별 null 기준선 (split_metrics 열):
  dom_null_majority    그 분할의 평가 쌍에서 가장 흔한 참 지배 채널(dom_majority_channel)의 비율 = "항상 그 채널" 예측
  dom_null_perm        예측 지배 채널을 그 분할의 쌍 사이에서 무작위 순열했을 때의 기대 일치율
                       = Σ_c p_true(c) p_pred(c) (정확한 기댓값)
  dom_null_perm_mc(_sd) 같은 값의 Monte-Carlo 확인: 1,000 순열, rng = np.random.default_rng(rev5_common.SEED)를
                       분할마다 새로 만든다. 정확값과 5 SE 넘게 다르면 아무것도 쓰지 않고 중단.
  dom_perm_p           순열 일치율 >= 관측 dom_agree_all 인 비율, (1 + k) / (1 + 1000)
  dom_true_share_<c>, dom_pred_share_<c>   채널별 참 / 예측 지배 비율
"""
import argparse, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev5_common as R5
from mmp_utils import annotate, build_pairs, splits
from train_ml_single import GEOM_TAG, arm_needs_ext, check_prereg_inputs, load_ext, merge_ext, resolve_arm, sha256

CSV = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")
MODEL = "KRR_rbf"
N_PERM = 1000
ARM = os.environ.get("ESPLEY_DOWNSTREAM_ARM") or "ESPLEY73"
RES5_ENV = os.environ.get("ESPLEY_RES5") or "0"
RES5 = RES5_ENV == "1"

env = os.environ.get
ap = argparse.ArgumentParser()
ap.add_argument("--geom", default=env("ESPLEY_GEOM"), help="g1 | g2 (output label)")
ap.add_argument("--feat", type=Path, default=env("ESPLEY_FEAT"))
ap.add_argument("--rows", type=Path, default=env("ESPLEY_ROWS") or (R5.ROWS5 if RES5 else R5.ROWS4))
ap.add_argument("--csv", type=Path, default=env("ESPLEY_COLEY_CSV", CSV))
ap.add_argument("--out", type=Path, default=None,
                help="default results_rev5/downstream_<arm>/ (ESPLEY_RES5=1) or results_rev4/downstream_<geom>/")
A = ap.parse_args()
if A.geom not in GEOM_TAG or A.feat is None:
    ap.error(f"need --geom {'|'.join(GEOM_TAG)} and --feat (or ESPLEY_GEOM, ESPLEY_FEAT)")
if RES5_ENV not in ("0", "1"):
    ap.error(f"ESPLEY_RES5 must be 0 or 1, got {RES5_ENV!r}")
F = resolve_arm(ARM)
EXT = arm_needs_ext(ARM)
if EXT and A.geom != "g1":
    sys.exit(f"arm {ARM} uses the blocks B1..B6, which exist for G1 only (--geom {A.geom})")
CHECKED = check_prereg_inputs([ARM], A.feat, R5.FEAT_EXT if EXT else None, A.rows)   # EXT_SEL: files C selected on
if CHECKED:
    print(f"prereg_rev5b.json inputs match: {', '.join(CHECKED)}", flush=True)
if A.out is not None:
    OUT = A.out
elif RES5:
    OUT = R5.RES5 / f"downstream_{ARM}"
elif ARM != "ESPLEY73":                               # results_rev4/downstream_<geom>/ holds the rev 4 ESPLEY73 outputs
    sys.exit(f"ESPLEY_DOWNSTREAM_ARM={ARM} needs ESPLEY_RES5=1 or --out")
else:
    OUT = R5.RES4 / f"downstream_{A.geom}"
OUT.mkdir(parents=True, exist_ok=True)
OUTS = {k: OUT / f"{k}_{A.geom}.csv" for k in ("split_metrics", "margin_calibration", "mmp_pairs_v2")}
if all(p.exists() for p in OUTS.values()):
    print(f"{', '.join(p.name for p in OUTS.values())} exist in {OUT} — skip")
    sys.exit(0)

CH = {"d1": "dft_d1_kcal", "d2": "dft_d2_kcal", "elst": "dft_elst_dft", "pauli": "dft_pauli_dft",
      "oi": "dft_oi_dft", "cpcm": "dft_cpcm_dft", "disp": "dft_disp_dft"}     # rev 4 order (argmax ties: first)
rows = pd.read_csv(A.rows)[["rxn_id"]]
if rows.rxn_id.duplicated().any():
    sys.exit(f"{int(rows.rxn_id.duplicated().sum())} duplicate rxn_id in {A.rows}")
feat = rows.merge(pd.read_parquet(A.feat), on="rxn_id", how="left", validate="1:1")
ext_info = None
if EXT:
    blocks, ext_info = load_ext()
    feat = merge_ext(feat, blocks, how="left")
bad = (feat.xtb_status != "ok") | feat[list(dict.fromkeys(F + list(CH.values())))].isna().any(axis=1)
if EXT:
    bad |= feat.ext_status != "ok"
if bad.any():
    sys.exit(f"{int(bad.sum())} of {len(feat)} rows of {A.rows} are missing / not ok / NaN in {A.feat}"
             f"{' + ' + ext_info['ext_features_file'] if EXT else ''}: {feat.rxn_id[bad].tolist()[:10]}")
tags = set(feat.geom.astype(str)) if "geom" in feat else {"<no geom column: rev 3 parquet?>"}
if tags != {GEOM_TAG[A.geom]}:
    sys.exit(f"{A.feat} has geom {sorted(tags)}, not --geom {A.geom}")
print(f"geom {A.geom}, arm {ARM} ({len(F)} columns), {MODEL} fixed HP: {len(feat)} rows ({A.rows}), feat {A.feat}"
      f"{' + ' + ext_info['ext_features_file'] if EXT else ''} -> {OUT}", flush=True)

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


def dom_null(dom_t, dom_p, name):
    """Null baselines of dom_agree on one split's evaluated pairs (see the module docstring)."""
    n, k = len(dom_t), len(chans)
    pt, pp = np.bincount(dom_t, minlength=k) / n, np.bincount(dom_p, minlength=k) / n
    maj = int(np.argmax(pt))
    exact = float(pt @ pp)
    rng = np.random.default_rng(R5.SEED)
    perm = np.array([np.mean(dom_t == rng.permutation(dom_p)) for _ in range(N_PERM)])
    mc, sd = float(perm.mean()), float(perm.std(ddof=1))
    tol = 5 * sd / np.sqrt(N_PERM) + 1e-12
    if abs(mc - exact) > tol:
        sys.exit(f"{name}: permutation baseline, Monte-Carlo {mc:.6f} vs exact {exact:.6f} differ by more than 5 SE "
                 f"({tol:.2e}) — nothing written")
    obs = float(np.mean(dom_t == dom_p))
    out = dict(dom_majority_channel=chans[maj], dom_null_majority=float(pt[maj]), dom_null_perm=exact,
               dom_null_perm_mc=mc, dom_null_perm_mc_sd=sd, dom_perm_p=(1 + int((perm >= obs).sum())) / (1 + N_PERM))
    out.update({f"dom_true_share_{c}": float(pt[j]) for j, c in enumerate(chans)})
    out.update({f"dom_pred_share_{c}": float(pp[j]) for j, c in enumerate(chans)})
    return out


metrics, calib = [], []
chans = list(CH)
for name, folds in splits(feat, pairs):
    P = {}
    for c, tg in CH.items():
        P[c], seen = oof_predict(folds, feat[tg].values)
    # 평가 쌍: 두 멤버 모두 OOF(random/class) 또는 to-멤버가 test(loso)
    if name.startswith("loso"):
        te_set = set(np.where(seen)[0])
        ok = np.array([(a in te_set) ^ (b in te_set) for a, b in zip(i1, i2)])
    else:
        fold_of = np.zeros(len(feat), int)
        for k, (_, te) in enumerate(folds): fold_of[te] = k
        ok = fold_of[i1] == fold_of[i2]
    if ok.sum() < 30:
        continue
    a, b = i1[ok], i2[ok]
    Dt = np.stack([feat[CH[c]].values[b] - feat[CH[c]].values[a] for c in chans], 1)
    Dp = np.stack([P[c][b] - P[c][a] for c in chans], 1)
    row = dict(split=name, n_pairs=int(ok.sum()))
    for j, c in enumerate(chans):
        e = P[c] - feat[CH[c]].values
        row[f"mae_{c}"] = np.nanmean(np.abs(e[seen])); row[f"dmae_{c}"] = np.mean(np.abs(Dp[:, j] - Dt[:, j]))
    dom_t, dom_p = np.argmax(np.abs(Dt), 1), np.argmax(np.abs(Dp), 1)
    row["dom_agree_all"] = np.mean(dom_t == dom_p)
    row.update(dom_null(dom_t, dom_p, name))
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
          f"oi {row['dmae_oi']:.2f} | dom {row['dom_agree_all']:.3f} (null: majority {row['dom_majority_channel']} "
          f"{row['dom_null_majority']:.3f}, perm {row['dom_null_perm']:.3f}, p {row['dom_perm_p']:.3f}) | "
          f"tau95 {row['tau_95']} (cov {row['cov_95']:.2f}) tau99 {row['tau_99']} (cov {row['cov_99']:.2f})", flush=True)

for key, df in (("split_metrics", pd.DataFrame(metrics).round(3)), ("margin_calibration", pd.DataFrame(calib).round(3)),
                ("mmp_pairs_v2", pairs)):
    R5.write_atomic(OUTS[key], lambda f, df=df: df.to_csv(f, index=False))
R5.write_json(OUT / f"evaluate_pairs_{A.geom}.json",
              dict(arm=ARM, n_features=len(F), model=f"{MODEL} (fixed alpha 1e-3, gamma 1e-3; OOF)", geom=A.geom,
                   n_rows=len(feat), n_pairs=len(pairs), channels=CH, n_perm=N_PERM, perm_seed=R5.SEED,
                   rows_file=str(Path(A.rows).resolve()), rows_sha256=sha256(A.rows),
                   features_file=str(A.feat), features_sha256=sha256(A.feat),
                   ext_features_file=ext_info["ext_features_file"] if EXT else None,
                   ext_features_sha256=ext_info["ext_features_sha256"] if EXT else None,
                   prereg_inputs_checked=CHECKED))
print(f"wrote {', '.join(str(p) for p in OUTS.values())}")
