# SPEC — A/B/C Baseline Ablation (in-distribution, n=787)

**버전** 1.0
**목적** Δ-learning의 baseline으로 `ridge` vs `xgb` 중 무엇이 나은지를,
**OOD 없이 in-distribution CV만으로** 결정한다.
**환경** `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction`, conda env `reactot`,
RTX 3090, SLURM.

---

## 0. 범위 (Scope)

- 데이터: 기존 in-distribution **787 반응만**. **Claisen/OOD는 이 실험에서
  완전히 제외**한다(OOD 비교는 별도 spec).
- 채널: `strain, Pauli, elst, oi, disp` (kcal/mol). `barrier = Σ_c ŷ_c`
  (ASM 항등식).
- descriptor: 기본 `m3` (24-d, `d1..d24` — 다각 xgb를 돌린 그 set). 플래그
  `--descriptor-set {m1,m2,m3}` 로 교체 가능.
- 목표: A/B/C를 **동일 split · 동일 metric**으로 비교하고, baseline 선택을
  데이터로 확정한다.

---

## 1. 세 팔 (Arms)

| arm | 이름 | baseline `b` | residual `δ` | 최종 `ŷ` |
|---|---|---|---|---|
| A | `xgb_direct` | — | — | `XGB(X)` (5채널 직접 예측) |
| B | `ridge_delta` (현재 모델) | per-channel `Ridge(α=1)` on `z(X)` | `MACE→CA→MLP` | `b + δ` |
| C | `xgb_delta` (검증 대상) | `XGB(X)` | `MACE→CA→MLP` (**B와 동일**) | `b + δ` |

- **불변 조건**: δ 아키텍처·학습 설정은 B와 C에서 **완전히 동일**하다. 오직
  baseline `b`만 다르다 → baseline 효과만 격리한다.

---

## 2. 데이터 & Split

- outer CV: **K = 5 fold**, **reaction-level**(conformer/atom-map 중복 없음;
  기존 dedup 재사용), **family-stratified**
  (`dipolar / qmrxn20_e2 / qmrxn20_sn2 / rgd1`).
- `seed = 42` 고정. fold index를 `splits/outer_folds.json` 에 **1회 저장**
  → 모든 arm이 동일 split 사용(assert, §8-1).
- 모든 적합(scaler, baseline, δ)은 **train fold만** 사용. 모든 metric은
  787 전체의 **pooled out-of-fold (OOF)** 예측으로 계산한다.

---

## 3. 누설 방지 — Nested Cross-Fitting (이 spec의 핵심)

**문제.** δ는 잔차 `r_train = y_train − b(X_train)` 을 학습한다. 만약 `b`를
train 전체로 적합한 뒤 **같은 train을 예측**하면, xgb는 train을 외워
`r_train ≈ 0` 이 되고 → δ가 배울 신호가 사라지며, val에서 `b`가 훨씬
떠져 `ŷ`가 miscalibrate 된다. (ridge는 자치적으로 이 문제가 약하지만,
C의 xgb에는 치명적.)

**규칙 (B·C 공통 적용).** residual arm의 **δ 학습 타깃**은 baseline의
**out-of-fold 예측**으로 만든다:

1. 각 outer-train fold 안에서 inner **K′ = 5** fold를 나눈다.
2. inner fold마다 `b`를 inner-train에 적합 → inner-val 예측 → outer-train
   전체에 대한 `b_oof` 를 조립.
3. δ는 `r_train = y_train − b_oof(X_train)` 로 학습한다.
4. outer-**VAL**에 쓰는 baseline은 **outer-train 전체**로 적합한 `b_full`
   → `ŷ_val = b_full(X_val) + δ(X_val)`.

> **비유.** baseline이 **자기 답안을 자기가 채점**하면 안 된다. δ는
> "남이 채점한 점수(OOF)"를 봐야 baseline의 **진짜 실수**를 배운다.
> 자기 채점(in-fold)은 항상 만점처럼 보여서 δ가 배울 게 없다.

- 근거: cross-fitting으로 과적합 편향 제거 → Chernozhukov et al. (2018).
  out-of-fold stacking 원조 → Wolpert (1992), Breiman (1996). (§Appendix)

---

## 4. 팔별 구현

### 4.0 공통
- z-score: scaler는 **해당 (inner/outer)-train만**으로 적합.
- `σ_c`(채널 std): train에서 계산, loss 정규화·NMAE에 사용.
- seed 전역 고정(`numpy / torch / xgboost`), 패키지 버전 로깅.

### 4.1 A — `xgb_direct`
- `xgboost.XGBRegressor` 채널별 5개(또는 `MultiOutputRegressor`).
- HP: **다각 이미 튜닝한 config를 재사용**한다. 없으면 아래 default를 쓰되,
  추가 튜닝은 **inner CV로만**(누설 금지).
- default: `n_estimators=800, max_depth=4, learning_rate=0.03,
  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
  min_child_weight=5`, early stopping은 inner split으로.

### 4.2 B — `ridge_delta` (현재 모델)
- `b`: `sklearn Ridge(alpha=1.0)`, per-channel, intercept 포함(기존 spec대로
  `[z(X) ⊕ 1]`).
- `δ`: **기존 코드 그대로**. `MACE-OFF23 (frozen) → std → proj → 4×CA + LN →
  AttentionPool → concat+diff (768) → MLP(768→64→64→5)`.
  loss `L1/σ_c`. `Adam lr=1e-5, weight_decay=1e-3, grad-clip=5.0`,
  early-stop(기존 patience).

### 4.3 C — `xgb_delta`
- `b`: **A와 동일한 xgb config**.
- `δ`: **B와 완전히 동일한** 코드·하이퍼파라미터·seed.

---

## 5. Metric & 통계

- 채널별:
  - `NMAE_c = MAE_c / MAD_c`, 단 `MAD_c = mean|y_c − ȳ_c|`
    (mean-predictor의 MAE). `ȳ_c`, `MAD_c` 는 **787 전체 기준으로 고정**
    (분모 안정화).
  - `RMSE_c`, `R²_c`, `slope_c` (parity 기울기: OLS of `ŷ_c` on `y_c`).
- barrier: `Σ_c ŷ` vs `Σ_c y` → `NMAE/RMSE/R²/slope`.
  cancellation ratio `ρ = |Σ_c ê_c| / Σ_c |ê_c|`.
- 모두 **pooled OOF(787)** 로 계산.
- 불확실성: reaction 단위 **bootstrap (B_boot = 1000)** → 각 metric 95% CI
  (기존 에러바 스타일). per-fold `mean ± std`는 별기.
- **arm 비교**: 반응별 `|error|` 차이를 bootstrap → **ΔNMAE(B−C) 95% CI**
  (+ 보조로 Wilcoxon signed-rank). "C가 B를 **유의하게** 이기나?"의 정량 답변.

---

## 6. 산출물 (Artifacts)

- `results/abc_metrics.csv` — `arm × channel × metric × (point, ci_low, ci_high)`.
- 그림(기존 navy/white minimal, Malgun Gothic / Noto Sans CJK KR):
  1. 그룹 NMAE 막대 — `A/B/C × (5채널 + barrier)` + CI 에러바.
  2. parity grid — `arm(행) × channel(열)`, 각 칸에 `NMAE/R²/slope` 주석.
  3. RMSE 막대.
- `results/oof_pred_{A,B,C}.parquet` — 반응별 OOF 예측(재활용).
- `results/REPORT.md` — 위 셋 baseline + 숫자·CI·판정 요약.

---

## 7. 환경 / SLURM / 재현

- 경로: `.../eda-asm-prediction/experiments/abc_ablation/`.
- A: CPU로 충분. B/C: GPU. `fold × arm = 5 × 2 = 10` 번의 δ 학습 →
  **SLURM array job**.
- split index를 1회 생성 후 공유. seed·패키지 버전 로깅
  (`xgboost, scikit-learn, torch, mace, tblite`).

---

## 8. 수용 기준 (Sanity Gates)

1. 모든 arm이 **동일 fold index** 사용 → assert.
2. 어떤 outer fold도 train/val에 **같은 reaction을 공유하지 않음** →
   reaction-level assert.
3. B/C의 δ 학습 타깃이 **OOF baseline**으로 만들어졌는지 assert.
   `median|r_train|`이 mean-predictor 대비 비정상적으로 0에 붕괴하면
   (= 누설 신호) **fail**.
4. **회귀 테스트(smoke)**: B가 기존 full-model NMAE의 같은 대역인지 확인
   → `strain≈0.66, Pauli≈0.62, Velst≈0.62, oi≈0.61, disp≈0.22, barrier≈0.43`
   (tol ±0.05; fold 구성이 달라 정확 일치는 아님). A는 기존 xgb-direct 대역
   → `strain≈0.76, Pauli≈0.65, Velst≈0.68, oi≈0.67, disp≈0.25` (tol ±0.05).
   크게 어긋나면 harness 버그로 간주, 중단.
5. `REPORT.md`에 **ΔNMAE(B−C) CI와 명확한 판정**
   (예: "C는 in-distribution에서 B와 통계적으로 구분 안 됨 → 단순·안정성
   우선인 B를 유지").

---

## 9. Claude Code 작업 분해 (Tasks)

- **T1 데이터 로더** — 787의 `X`(선택 descriptor set), `y`(5채널), δ용 MACE
  feature/geometry, family label 로더.
- **T2 split 생성기** — family-stratified reaction-level 5-fold →
  `splits/outer_folds.json`. gate #1,#2.
- **T3 baseline 모듈** — `fit_ridge(X,y)`, `fit_xgb(X,y)`;
  `cross_fit_oof(model_fn, X_train, y_train, K'=5)` → `b_oof`;
  그리고 `b_full`.
- **T4 arm A** — `xgb_direct` OOF 예측 → `oof_pred_A`.
- **T5 arm B/C** — 기존 δ trainer를 **최소 변경으로 대체**, baseline만
  주입(`b_oof`로 학습, `b_full`로 val 예측). gate #3.
- **T6 metric/bootstrap** — §5 전 지표 → `abc_metrics.csv`.
- **T7 plotting** — 기존 스타일로 3 그림.
- **T8 REPORT** — `REPORT.md` 작성 + gate #4,#5 실행.

---

## Appendix — 근거 (DOI 검증 완료)

- Δ-learning(잔차 = 이론 오차 보정) — Ramakrishnan, R.; Dral, P.O.; Rupp, M.;
  von Lilienfeld, O.A. (2015). *J. Chem. Theory Comput.* 11(5), 2087–2096.
  **`10.1021/acs.jctc.5b00099`**
- cross-fitting으로 과적합 편향 제거 — Chernozhukov, V. et al. (2018).
  "Double/debiased machine learning for treatment and structural parameters."
  *The Econometrics Journal* 21(1), C1–C68. **`10.1111/ectj.12097`**
- out-of-fold stacking 원조 — Wolpert, D.H. (1992). "Stacked Generalization."
  *Neural Networks* 5(2), 241–259. / Breiman, L. (1996). "Stacked Regressions."
  *Machine Learning* 24, 49–64.
- ridge baseline — Hoerl, A.E.; Kennard, R.W. (1970). *Technometrics* 12(1),
  55–67. **`10.1080/00401706.1970.10488634`**
- XGBoost — Chen, T.; Guestrin, C. (2016). *KDD '16*, 785–794.
  **`10.1145/2939672.2939785`**
