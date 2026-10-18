# D1PAIR rev 9 — Claude Code 실행 지시서 (이 파일 하나만 쓴다)

**대상:** `espley-rev6` `957f1fc` (`analysis/espley_xtb_repro/`) → 새 브랜치 `d1-pair`
**폴더:** 코드 `analysis/d1_pair/` · 결과 `analysis/d1_pair/results/` · scratch `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/` (= `$D1PAIR_SCRATCH`)
**이 파일:** `docs/specs/D1PAIR_REV9.md`로 commit한다. rev 7·8을 대체한다(받았더라도 쓰지 않는다). 그림 파일은 필요 없다.
**보고:** Phase마다 보고한다. STOP 조건이 나오면 다음 Phase로 가지 않고 사용자에게 묻는다. 버그를 찾으면 고치기 전에 보고한다.
**순서:** 0 준비 → G D1 xTB 구조 → F D1 feature → P 행·짝 고정 → F0 D0 모델 → S 분할·사전 등록 → M 학습 → E 평가·그림 → W 요약

---

## 0. 무엇을 하는가

### 0-1. 질문과 방법
- **질문:** 같은 반응 계열에서 작용기 X만 바꿨을 때 장벽과 8채널이 얼마나 변하는지(ΔΔ)를, 처음 보는 (dipole, 골격) 조합에서 맞힐 수 있는가. 장벽 Hammett은 직선인데 채널이 꺾이거나 V자인 계열도 맞히는가.
- **방법 (타깃 t마다 따로):**
  ```
  f0      : D0로 학습하고 고정한다 (D1을 보지 않는다)
  dd      = y_a − y_b                          짝 = 치환 반응 a − 같은 계열의 기준 반응 b
  목표    = dd − [f0(x_a) − f0(x_b)]           f0가 못 맞힌 차이
  g       : 목표를 학습. 입력 Δx = x_a − x_b, x̄ = (x_a + x_b)/2. 조건 g(−Δx, x̄) = −g(Δx, x̄)
  예측    = f0(x_a) − f0(x_b) + g(Δx, x̄)
  ```
- **예측 대상:** 반응 하나의 값 = TS 한 점의 에너지와 그 분해. 기준 = 그 반응 자신의 반응물(두 분자를 떼어 각자 가장 안정한 구조, 라벨의 frag_rel = orig/alt 중 낮은 것). 반응물 복합체·반응 경로·생성물·자유에너지는 예측하지 않는다. Hammett 값 = −ΔΔE‡ / 1.3642 ≈ log(k_a/k_b) (298 K, 전자 에너지 기준이라 자유에너지 속도비와 같지는 않다).
- **왜 짝인가:** rev 6 D0 모델은 비슷한 반응 106쌍(같은 seed 시험 반응 중 반응물 파일·원자 수가 같은 regio·입체 형제)에서 오차 상관이 elst 0.23, Pauli 0.14, OI 0.07, CPCM 0.15였다. 따로 예측해 빼면 차이 오차가 거의 줄지 않는다. 차이를 직접 배우는 모델이 나았다는 선행 결과: DeepDelta (Fralish et al., J. Cheminform. 2023, 15, 101, doi:10.1186/s13321-023-00769-x, ADMET 분야).

### 0-2. 데이터 사용 (D0 + D1 전체)

| 데이터 | 라벨 | 학습에 쓰는 것 | 쓰는 곳 |
|---|---|---|---|
| D0 | `labels_all.json` 5,265행 (rev 6 `R_inventory.json` n_labels 5,260) | **4,963행** = 라벨 ok ∩ G1 feature 성공 ∩ hygiene (rev 6 부록 A와 같은 행) | f0 |
| D1 | `labels_d1.json` 9,496행 | 제외 461 → 9,035 → 짝에 쓰이는 반응 **8,659** = 기준 짝 7,253(반응 7,410) + anchor 짝 1,159(반응 1,249) | g |
| 합 | 14,761 | **13,622** (D1 G1·feature 실패 전. D0는 이 단계에서 약 6 % 빠졌다) | |

- rev 6 **주 비교**의 Espley 부분집합(행 C 3,386)은 쓰지 않는다. rev 6에서 가져오는 것은 **코드**(feature 182개, grid, NN 구조, G1 절차)뿐이다.
- D1에서 빠지는 것: 제외 461(§2-1, 사용자 확인 2026-10-19), 짝을 만들 수 없는 376(입체 이성질체가 섞인 계열 361, 짝 없는 기준 6, 기타 direct 6, 반응 하나뿐인 계열 3).

### 0-3. 사용자 결정 (바꾸지 않는다)

| # | 결정 |
|---|---|
| U1 | D0와 D1 결합은 짝 차이 모델: f0(D0) 고정, g(D1 짝)만 학습 |
| U2 | g 모델 6종: Ridge, KRR, SVR, XGB, NN-2(은닉층 2개), NN-4(은닉층 4개) |
| U3 | feature = rev 6과 같은 xTB 182개(EXT_ALL), 같은 코드·같은 G1 절차. 짝 입력 [Δx, x̄] |
| U4 | σ, 계열·core 번호, X 이름, 위치 이름은 입력에 넣지 않는다 |
| U5 | 평가는 반응군(core)을 통째로 빼서 한다 |
| U6 | 판정 기준은 결과 전에 고정한다 (PREREG) |
| U7 | D''(DppC·DppT) 포함 제외 461행은 계속 뺀다 |
| U8 | D1 최종 라벨 = `labels_d1.json` 9,496행 |
| U9 | D0 전체 + D1 전체로 학습한다 |

**기본값으로 넣은 것** (바꾸려면 PREREG commit 전에 사용자에게 묻는다): anchor 짝(§2-3)을 g 학습에 넣음 · arm P(KRR) · 단계 2 기준 0.13 kcal/mol · 보조 분할 LODO·LOSO · V를 본 적 없는 모델 시험 T2(§4) · 원자 번호 SMILES 만드는 방법(§7-2).

---

## 1. 입력 (수정 금지, sha256 확인)

| 이름 | 위치 | 확인값 |
|---|---|---|
| D1 라벨 | `D1_build/d1_run/results/labels_d1.json` 또는 사용자 지정 경로 | sha256 `1eaa417102e20813beea9ba190ef8140233a4a9b0b1d9a69afac16aeaeb210a7`, 9,496행, 모두 status ok, rxn_id 1000001–1011595, job_id T00001–T11595 |
| D1 상태표 | `d1_status.csv` | sha256 `a9df873b5021fb11e0d2bd2b2ba1b012abad9e7981f32b676172ec269cabfcf7` |
| D1 job 폴더 | `<D1_JOBS>/<job_id>/`: `ts.xyz`, `ref_dipole.xyz`, `ref_dipolarophile.xyz`, `label.json`, `sp/eda.inp` | 라벨의 `ts_file`·`rel1_file`·`rel2_file`는 이 폴더 기준 상대 경로 |
| D0 라벨 | 저장소 루트 `labels_all.json` (5,265행) | Phase 0에서 sha256 기록 |
| D0 feature | `$R6_SCRATCH/xtb_r6/X182_all.parquet` (`rev6_common.X182_ALL`) | rev 6 기록과 같은 sha256 |
| D0 타깃·행 | `$R6_SCRATCH/S/S_targets.parquet`, `results_rev6/S_splits_appendixA.csv` | rev 6 `prereg_rev6.json`의 sha256 |

- 사용자가 D1 **중간** 파일을 `labels_all.json`이라는 이름으로 올린 적이 있다(sha256 `a2df136b…`, 9,459행). D0 `labels_all.json`과 **sha256과 행 수로 구분한다.**
- 조각 정보는 `sp/eda.inp`의 원자 표시 `(1)`(frag1 = dipole) / `(2)`(frag2 = dipolarophile)를 쓴다. D1 라벨의 원자 순서는 섞여 있다.
- `<D1_JOBS>`: 서버에서 `T00001/ts.xyz`와 `T00001/sp/eda.inp`를 가진 폴더를 찾아 `results/INPUTS.md`에 적는다. 못 찾거나 짝에 쓰이는 8,659개 job 폴더 중 하나라도 파일이 없으면 **STOP**.

---

## 2. 행과 짝 (Phase P, 부록 코드를 그대로 실행)

### 2-1. 행 속성 — 부록 A `build_rowattr.py`
- 입력: `labels_d1.json`, 부록 C σ 표(Hansch, Leo & Taft, Chem. Rev. 1991, 91, 165, doi:10.1021/cr00002a004).
- 만드는 열: dipole·dipolarophile SMILES, 기준 구조 초과 에너지(ex1, ex2), Xfull·Xb·pos(p/m/o/direct, 기준 = ref_aryl(X = Ph)·ref_direct(X = H)), cfg, orient, c3.
- 제외: D''(DppC·DppT), ex1 또는 ex2 > 20 kcal/mol, flag_async, flag_short_forming, flag_regio_ambiguous.
- **gate P-1:** 9,496 → 제외 461 → **9,035** (m 2,321 · p 2,283 · o 2,240 · direct 2,028 · ref_aryl 110 · ref_direct 53).

### 2-2. 기준 짝 — 부록 B `d1_pairs.py` (주 평가에 쓰는 짝)
- 계열 키 F = (dipole, skeleton, series, cfg, orient, face, c3). 아릴 X: b = 같은 F의 ref_aryl. direct X: b = 같은 dipole·skeleton·face의 H 행(여럿이면 orient·cfg·c3로 좁힘, 정확히 하나).
- 타깃 10개 `dd_<t>` (kcal/mol): d1, d2, elst, pauli, oi, disp, cpcm, cds, c_ghost(`bsse_shift_kcal`), barrier. d1 = dipole 변형, d2 = dipolarophile 변형. 항등식 dd_barrier = 9개 합(|잔차| 중앙값 0.0026, 최대 0.096).
- `fam` = F를 '/'로 이은 것, `core` = dipole/skeleton.
- **gate P-2:** 짝 **7,253** (p 1,878 · m 1,910 · o 1,734 · direct 1,731), 계열 146, core 46, 기준 반응 157(ref_aryl 104 · H 53), 반응 7,410, 점 12개 이상 para 계열 96. 기준 반응 35개를 여러 계열이 함께 쓴다(짝 1,403). 모든 기준 반응은 한 core 안에만 있다.
- 비교는 **개수와 (pair_id, a_job, b_job) 목록**으로 한다(sha256은 pandas 버전에 따라 다를 수 있음; 참고 `pairs_d1.csv` `227063285d65a05195a0d662df040618ac52bd175718837423ded2343c83ecce`).

### 2-3. anchor 짝 — 부록 E `d1_anchor_pairs.py` (g 학습을 늘리는 짝)
- 기준 반응이 **없는** 계열(아릴: 같은 F의 ref_aryl이 없음, direct: H 후보 0개)에서 계열 안 반응 하나를 anchor로 정하고 나머지를 그 anchor와 짝짓는다. 반응 n개 계열의 독립 차이는 n − 1개라 정보 손실이 없다.
- anchor 순서: hygiene 통과 → |σp(X)| 작은 것 → p < m < o < direct → X 이름. 기준 반응이 중복된 계열(입체 이성질체가 섞임)과 F·X·pos가 중복된 행은 쓰지 않는다.
- 열: §2-2와 같고 `anchor_Xb`, `anchor_pos`, `ref_kind = "anchor"`를 더한다. `fam` 끝에 `/anchor`. pair_id `A00000`부터.
- **gate P-3:** anchor 짝 **1,159** (o 354 · p 293 · m 284 · direct 228), 계열 75, 반응 1,249(기준 짝 반응과 겹치지 않음), core 33(그중 `azomethine ylide/D`, `carbonyl ylide/C`, `carbonyl ylide/D`, `carbonyl ylide/Dp` 4개는 기준 짝이 없는 core). hygiene로 빠지는 짝 37. 참고 sha256 `2121c684bd11a77d4b59128432d3e6e22d3dab23081bd25bd9864e5172a9bb6c`.
- **주 지표·Hammett·단계 판정·fold 배정에는 쓰지 않는다.**

### 2-4. gate P-4 — 입체 번호 진단
- 기준 짝 852개는 a와 b의 `stereo_index`가 다르다(nitrone·azomethine ylide·carbonyl ylide의 AppE·AppZ·B·Bp core). stereo_index는 입체 이성질체를 나열한 번호라 X에 따라 번호가 바뀔 수 있으므로(CIP 우선순위가 X에 따라 달라짐) 실제 입체가 같은지 확인한다.
- 방법: DFT TS에서 새로 생기는 5원자 고리(형성 결합 `formed_pairs_ts` 두 개와 조각 표시로 dipole 끝 원자 2개·중앙 원자·dipolarophile 원자 2개를 정함)의 각 원자에 대해, 고리 밖 가장 무거운 치환기가 고리 평균 평면의 어느 쪽인지(부호)를 구해 a와 b를 비교한다. X 자리는 X(또는 Ph)의 첫 원자로 비교한다. 같은 무게 치환기가 둘인 자리는 비교하지 않는다.
- 기준 짝 전체와 anchor 짝에 돌려 불일치 수를 보고한다. **불일치가 하나라도 있으면 목록을 만들고 STOP.**

### 2-5. 짝 집합

| 이름 | 정의 |
|---|---|
| P_all | 기준 짝 7,253 |
| P_feat | P_all ∩ (a와 b 모두 182 feature 있음) |
| **P_main (주 평가)** | P_feat ∩ hygiene (a와 b 모두 d1 ≥ 0, d2 ≥ 0, d2 ≤ 50; rev 6 `hygiene_mask`) |
| P_sens | P_feat (hygiene 무시) |
| A_all | anchor 짝 1,159 |
| A_feat | 182 feature가 있는 행만으로 부록 E를 **다시 실행**해 만든 anchor 짝 (anchor가 실패하면 다음 반응이 anchor) |
| A_main | A_feat ∩ hygiene |
| **T_main (g 학습)** | 학습 쪽의 P_main ∪ A_main |

- hygiene로 빠지는 짝: P_all 75(기준 반응 0), A_all 37. feature 필터 뒤에 다시 센다. 행 흐름표(P_all → P_feat → P_main, A_all → A_feat → A_main)를 사유별로 만든다.

---

## 3. 모델

### 3-1. arm

| arm | 예측 ΔΔŷ | 학습 | 역할 |
|---|---|---|---|
| null | 0 | 없음 | 하한선 |
| A | f0(x_a) − f0(x_b) | 없음 | D0 모델만 |
| **B** | f0(x_a) − f0(x_b) + g | g: 목표 dd − Δf0, 학습 짝 T_main | **제안** |
| C | g | g: 목표 dd, 학습 짝 T_main | D0의 기여 확인 |
| P (보조) | f_pool(x_a) − f_pool(x_b) | KRR 하나를 D0 4,963행 + 학습 쪽 D1 반응(T_main의 a·b)으로 학습 | "D0 + D1을 한 모델에 그냥 합치면?" |
| B−ref (부 비교) | B와 같음 | g(KRR만)를 학습 쪽 P_main만으로 학습 | anchor 짝 효과 |

- 장벽은 dd_barrier 직접 예측과 9개 채널 예측의 합(Σ)을 둘 다 보고한다.
- disp: b_disp feature가 분산 채널과 거의 같은 양이라 결과는 보고하되 헤드라인으로 쓰지 않는다.

### 3-2. 짝 입력과 스케일 (반대칭을 지키는 것이 핵심)
- z = [Δx / s_Δ, (x̄ − μ) / s]. s_Δ = 학습 짝 Δx 표준편차, **평균은 빼지 않는다**. μ, s = 학습 짝 x̄ 평균·표준편차.
- 학습 짝에서 s_Δ < 1e-12인 Δx 열, s < 1e-12인 x̄ 열은 빼고 기록한다(D1은 모두 중성이라 `is_charged`가 상수).
- 목표 y: 학습 짝 표준편차로 나누기만 한다(평균 빼지 않음).
- 스케일은 바깥 학습 짝으로 정하고, 안쪽 CV에서는 안쪽 학습 짝으로 다시 정한다.
- rev 6 `_make_pipe`(평균을 빼는 스케일)는 g에 쓰지 않는다.

### 3-3. g의 6개 모델 — 반대칭 g(−Δx, x̄) = −g(Δx, x̄), g(0, x̄) = 0

| 모델 | 구현 | grid |
|---|---|---|
| Ridge | Δx만 입력, `fit_intercept=False` (대칭화하면 x̄ 항이 사라짐; 선형 기준선) | alpha: `train_ml_single.GRIDS["Ridge"]` |
| KRR | `KernelRidge(kernel="precomputed")`, K(p,q) = [k(Δx_p, Δx_q) − k(Δx_p, −Δx_q)] · k(x̄_p, x̄_q), k = RBF (절편이 없어 정확히 성립) | alpha: `GRIDS["KRR_rbf"]`; γ_Δ, γ_x̄ ∈ {3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1} |
| SVR | `SVR(kernel="precomputed")`, KRR과 같은 커널. 절편 b가 있으므로 짝을 양방향(Δx, −Δx; 목표 부호 반대)으로 학습하고 예측은 ½[s(Δx, x̄) − s(−Δx, x̄)] | C {1, 10, 100, 1000}, epsilon {0.05, 0.1, 0.5}, γ는 KRR과 같음 |
| XGB | 양방향으로 학습, 예측 ½[h(Δx, x̄) − h(−Δx, x̄)] | `GRIDS["XGB"]` |
| NN-2, NN-4 | **은닉층 2개, 4개**(rev 6 `r6_nn`의 `model_size` 2, 4). 층마다 ELU + L2 + dropout(마지막 은닉층 뒤에는 없음), 출력 1개, Adam, 손실 MAE, batch 32, 500 epoch. 공유 부분망 h를 `r6_nn._layers`로 만들고 Keras 모델 안에서 g = h([Δx, x̄]) − h([−Δx, x̄]). `r6_nn.fit_predict`·`tune`은 y 평균을 빼므로 그대로 쓰지 않는다 | `r6_nn.HP_STORAGE`(층마다 뉴런 {32, 64, 128, 256, 512}, dropout {0.1–0.6}, L2 {0.2, 0.1, 0.01, 0.001}; 학습률 {1e-3, 1e-4, 1e-5}), `HB`(Hyperband, max_epochs 100, factor 3), seed 23 |

- grid 키의 rev 6 접두어(`regressor__ridge__alpha` 등)는 떼고 값만 쓴다.
- 하이퍼파라미터: 바깥 학습 짝 안에서 `GroupKFold(4)`(group = core). NN은 첫 안쪽 fold를 Hyperband 검증으로 쓰고, 검증 MAE는 안쪽 4-fold 전체로 다시 학습·검증한 평균.
- **대표 모델:** arm·타깃마다 바깥 fold 평균 안쪽 CV MAE가 가장 낮은 모델. 시험 점수로 고르지 않는다. 모든 모델 결과를 보고한다.
- **gate M-1 (반대칭):** 학습한 g마다 무작위 시험 짝 200개에서 |g(Δx, x̄) + g(−Δx, x̄)| ≤ 1e-6 kcal/mol, |g(0, x̄)| ≤ 1e-6. 넘으면 **STOP**.

### 3-4. f0 — D0 모델 (Phase F0)
- 행: **D0 전체 4,963행**(`S_splits_appendixA.csv`의 rxn_id 합집합에서 읽고 4,963인지 확인). 시험 분할 없이 전부 학습에 쓴다. 행 C(3,386)로 줄이지 않는다.
- feature: `X182_all.parquet`의 182열(`rev6_common.our_columns(train_ml_single.FEATURE_SETS)`). 타깃 10개는 rev 6 `r6_train.tcol`의 열 이름(barrier = `dft_barrier_kcal`, c_ghost = `dft_c_ghost_kcal`).
- Ridge, KRR, SVR, XGB: `train_ml_single.GRIDS` + `_make_pipe` + `GridSearchCV(cv=KFold(5, shuffle=True, random_state=22), scoring="neg_mean_absolute_error")`, `best_estimator_` 저장. (`r6_train.run_sklearn`은 학습한 모델을 돌려주지 않아 쓸 수 없다.)
- NN-2, NN-4: rev 6 `r6_nn.tune`·`fit_predict` 그대로. 고정 10 %(seed 22)는 validation_data로만 쓰고 학습에 넣지 않는다. 검증 MAE는 5-fold CV로 따로 낸다.
- 타깃마다 대표 모델 = 5-fold CV MAE 최저. A·B arm은 대표 모델만 쓴다(부록 결과로 6개 모델 전부의 Δf0도 낸다).
- **gate F0-1:** 대표 모델 5-fold CV MAE ≤ 1.2 × rev 6 부록 test MAE (d1 1.70, d2 1.25, elst 3.55, Pauli 6.45, OI 3.98, disp 0.470, CPCM 1.54, CDS 0.222, c_ghost 0.233(rev 6 주 결과 NN-4), barrier 1.70(rev 6 B의 OUR-O KRR)). 넘거나 학습 행이 4,963이 아니면 **STOP**.
- 산출: `f0_models/`, `f0_cv.csv`, `f0_d1.parquet`(D1 rxn_id × 10 타깃 × 6 모델 예측).

---

## 4. 분할 (Phase S)

- **주 분할 T1 — core 5-fold:** core 이름을 정렬하고 `numpy.random.default_rng(20261019).permutation`으로 섞은 뒤, 그 순서대로 P_main 짝 수가 가장 적은 fold에 넣는다(같으면 번호가 작은 fold). `S_folds.csv`에 쓴다. 한 core의 계열·기준 반응·짝은 모두 같은 fold다.
- **anchor 짝의 자리:** fold 배정은 기준 짝 46 core로만 정한다. anchor 짝은 자기 core(보조 분할에서는 자기 dipole·skeleton)를 따라가고, 그 묶음이 시험 쪽이면 학습에서 빼고 부 지표로만 채점한다. 기준 짝이 없는 core 4개는 주 분할에서 늘 학습 쪽이다.
- **보조 분할:** leave-one-dipole-out(5), leave-one-skeleton-out(D'' 뺀 10).
- **보조 분할 T2 — V를 본 적 없는 모델:** T1과 같은 fold. 단, DFT 값에서 V 판정(부록 D `calls`, P_all, para)이 하나라도 있는 계열 11개의 짝(모든 위치, P_all 기준 827짝)을 **모든 fold의 학습에서 뺀다.** arm B, 모델 6개.
  - 11계열: `azide/A/R2//N-//`, `azide/AppZ/R1////`, `carbonyl ylide/Bp/R1/t//endo/`, `nitrile oxide/B/R1///endo/`, `nitrile oxide/B/R1///exo/`, `nitrone/A/R2//C//`, `nitrone/A/R2//O-//`, `nitrone/B/R1/c//exo/cis`, `nitrone/B/R2//O-/exo/`, `nitrone/C/R2//O-//`, `nitrone/Cp/R1/t///`.
  - 시험 대상: **숨은 V** 6계열(장벽 직선 + 채널 V: 위 목록의 carbonyl ylide/Bp, nitrone/A 2개, nitrone/B/R1, nitrone/C, nitrone/Cp)과 **숨은 꺾임** 30계열(장벽 직선 + 채널 꺾임).
  - Phase S에서 위 목록·개수(11, 827, 6, 30)를 부록 D로 다시 만들어 맞는지 확인한다. 다르면 **STOP**.
- **안쪽 분할:** 바깥 학습 짝(T_main) 안에서 `GroupKFold(4)`, group = core.
- **gate S-1 (누출 감사):** 바깥 fold마다 시험 core가 학습에 없고, 시험 짝의 a_job·b_job이 학습 짝(anchor 포함)에 없다. T2에서는 11계열 짝이 학습에 없다. 어기면 **STOP**.
- **같은 반응물 분자 (보고):** 기준 짝 7,253개 중 7,249개에서 X가 붙은 반응물이 다른 core에도 나온다. 주 분할의 "처음 보는 것"은 (dipole, 골격) **조합**이지 분자가 아니다. LODO(R1 짝)와 LOSO(R2 짝)가 X가 붙은 분자를 학습에서 빼므로 보완한다.
- **D0와 D1의 겹침 (보고):** 원자 번호·입체 표시를 지운 반응 SMILES(반응물 정렬 + 생성물, RDKit 정규화)로 같은 반응을 센다. D0 SMILES는 `g1_geom.SMILES_CSV`. 겹치면 뺀 민감도를 낸다.

---

## 5. 사전 등록 (Phase S, 어떤 학습보다 먼저)

`results/PREREG_D1PAIR.md` + `prereg_d1pair.json`을 쓰고 **commit한다.** 학습 스크립트는 commit 여부와 입력 sha256을 먼저 확인한다(`rev6_common.gate_prereg` 방식). 결과를 본 뒤 arm·모델·짝·지표·기준을 바꾸지 않는다. 바꾸면 이탈로 기록한다.

### 5-1. 담을 것
입력 sha256, P_main·P_sens·A_main 짝 목록 sha256, `S_folds.csv` sha256, T2 계열 목록, arm × 모델 × 타깃 × 분할(T1, LODO, LOSO, T2), grid, 스케일 규칙, 대표 모델 규칙, 단계 기준.

### 5-2. 지표 (시험 짝, 바깥 fold를 모두 합쳐서)
- **주 지표는 기준 짝(P_main)만으로.** anchor 짝 시험 MAE는 따로 한 줄.
- ΔΔ MAE(kcal/mol), RMSE, 위치별(p/m/o/direct) MAE, train MAE.
- 계열 안 R²: para 계열(H 포함 점 12개 이상)마다 y = −dd/1.3642에 H(0, 0)를 넣어 R²_fam = 1 − Σ(ŷ − y)²/Σ(y − ȳ)². 중앙값과 사분위.
- ρ 오차: para 계열마다 |ρ̂ − ρ| (σp 기울기, log 단위).
- 꺾임 판정 일치: 부록 D `hammett_eval.calls`(ΔAICc ≥ 4, 양쪽 3점 이상, 한 점 빼도 ≥ 2; V = 두 팔 부호 반대·둘 다 |기울기| ≥ 0.46)를 DFT 값과 예측값에 똑같이 적용해 채널마다 민감도와 위양성률, V 판정 일치를 낸다.
- **부록 D 자기 검사 기대값** (P_all DFT, para 96계열): 꺾임 — 장벽 33, d1 34, d2 15, elst 21, Pauli 24, OI 27, disp 5, cpcm 26, cds 12; V — 장벽 2, d1 3, elst 3, Pauli 3, OI 5; 장벽 직선 + 채널 꺾임 30; 장벽 직선 + 채널 V 6(마지막 값은 스크립트가 출력하지 않으므로 같은 `calls` 결과로 따로 센다).

### 5-3. 판정 기준 (arm B 대표 모델, T1 시험 짝)

| 단계 | 뜻 | 기준 (타깃마다) |
|---|---|---|
| 0 | 쓸모가 있다 | B의 MAE < null의 MAE, 그리고 B < A (차이 A − B의 95 % CI 하한 > 0, Holm p < 0.05) |
| 1 | 계열 경향을 재현 | 시험 para 계열 R²_fam 중앙값 ≥ 0.5 |
| 2 | 꺾임을 판별 | 궤도·cpcm의 para 짝 MAE ≤ 0.13 kcal/mol, 꺾임 판정 민감도 ≥ 0.7, 위양성률 ≤ 0.1 |

- 단계 2 근거: 숨은 꺾임 크기 중앙값이 궤도 0.47, 용매 0.45 kcal/mol. 모의실험(σp 18점)에서 잡음 SD ≤ 0.35 × 꺾임 크기일 때 판정률 0.92–0.99 → SD ≤ 약 0.16, MAE ≈ 0.8 × SD ≤ 약 0.13. 이 모의실험은 한 점 빼기 검사를 넣지 않았고 잡음이 점마다 독립이라고 가정했으므로 상한이다.
- 참고 — "0으로 예측"의 MAE(최종 DFT 기준 짝, kcal/mol):

| 타깃 | p | m | o | direct |
|---|---|---|---|---|
| barrier | 0.78 | 0.61 | 1.43 | 3.82 |
| d1 | 0.60 | 0.38 | 0.95 | 2.96 |
| d2 | 0.48 | 0.41 | 0.91 | 2.01 |
| elst | 1.32 | 0.92 | 2.13 | 5.94 |
| pauli | 2.59 | 1.70 | 3.87 | 11.34 |
| oi | 1.49 | 0.89 | 2.10 | 6.15 |
| disp | 0.15 | 0.35 | 1.09 | 1.81 |
| cpcm | 0.42 | 0.34 | 0.82 | 1.39 |
| cds | 0.03 | 0.05 | 0.17 | 0.29 |

### 5-4. 검정
- **주 비교 (확증, 9개):** 8채널 + barrier(직접)마다 B 대표 모델 대 A, 지표 = 시험 짝 MAE 차이(A − B). 95 % CI와 양측 p는 **core 단위 cluster bootstrap** 2,000회(seed 20261019, 시험 짝 전체에서 core 46개를 복원 추출). 9개에 **Holm 보정**. B가 낮은 fold 수도 센다.
- **부 비교 (설명, 보정 없음):** B 대 C, B 대 null, B 대 P, B(KRR) 대 B−ref(KRR), 모델별 모든 쌍, Σ장벽 대 직접 장벽, LODO·LOSO, P_sens, D0 겹침 제외 민감도.
- **T2 대 T1 (설명):** 숨은 V 6계열과 숨은 꺾임 30계열에서 채널별 꺾임·V 판정 민감도, 위양성률(그 채널이 DFT에서 직선인 시험 계열 기준), MAE. 민감도에는 Clopper–Pearson 95 % CI를 붙인다(예: 6계열 중 5개면 0.36–1.00이므로 숨은 V는 사례 제시 수준이다).
- 반올림: MAE 소수 둘째(disp, cds, c_ghost는 셋째), R² 셋째, p 유효숫자 둘.

---

## 6. Phase 0 — 준비
1. `espley-rev6` `957f1fc`에서 브랜치 `d1-pair`를 만든다.
2. 이 파일을 `docs/specs/D1PAIR_REV9.md`로 commit한다.
3. `analysis/d1_pair/`, `results/`, scratch를 만든다.
4. 상수 원본 `d1pair_common.py`: `rev6_common`을 import하고 상수를 복사하지 않는다. 경로, seed(20261019, 22, 23), arm, 모델, 타깃, 스케일 규칙, 단계 기준, gate 허용치, T2 계열 목록.
5. 부록 A–E를 그대로 `analysis/d1_pair/`에 둔다. 코드 블록 내용에 **줄바꿈 하나를 붙여** 저장했을 때의 sha256이 부록의 값이다. 다르면 공백·줄바꿈부터 확인하고, 그래도 다르면 **STOP**.
6. §1 입력 sha256을 `results/INPUTS.md`에 적는다. 다르면 **STOP**.
7. 수정 금지: `labels_d1.json`, D0 `labels_all.json`, rev 4–6의 기하·feature·결과·사전 등록 파일, D1 job 폴더.

---

## 7. Phase G — D1 xTB 구조 (G1)

### 7-1. 대상과 방법
- 대상: **8,659개** = 기준 짝 반응 7,410 + anchor 계열 반응 1,249. 순서: 기준 반응 157 → anchor 75 → 나머지(job_id 순).
- 방법은 rev 4 `g1_geom.process`와 같다: TS = ORCA 6.1.1 `! XTB2 ALPB(water) OptTS Freq`(출발 `ts.xyz`), 반응물 = xtb 6.7.1 `--opt tight --alpb water --gfn 2`(출발 `ref_dipole.xyz`, `ref_dipolarophile.xyz`). gate는 `g1_geom.GATES` 그대로(첫 허수 ≤ −40 cm⁻¹, mode share ≥ 0.5, 형성 결합, 외부 결합, partition, 참조 그래프). **느슨하게 하지 않는다.**

### 7-2. 어댑터 `d1_g1.py` (입력만 D1으로 연결, 계산 방법은 그대로)

| 코드가 읽는 입력 | D1에서 |
|---|---|
| 반응 폴더 `ESPLEY_PROF/<rxn_id>/` | `$D1PAIR_SCRATCH/prof/<rxn_id>` → `<D1_JOBS>/<job_id>` 심볼릭 링크 |
| 조각 A_idx `ESPLEY_META` | `sp/eda.inp`의 `(1)`/`(2)`로 만든 같은 형식 CSV(또는 함수에 직접 전달) |
| 원자 번호 SMILES `ESPLEY_SMILES` | 아래 방법으로 만든 CSV |
| 라벨 `ESPLEY_LABELS` | D1 라벨 사본(필드 이름만 rev 6 코드에 맞춤). G는 8,659행, F는 G1 성공 행만 |
| 출력·캐시 `G1_ROOT` 등 | `$D1PAIR_SCRATCH/` 아래 새 폴더 |

- **원자 번호 SMILES:** D1 `rxn_smiles`에는 원자 번호가 없어(예: T00879 `C#CC(=O)OC.N#[N+][NH-]>>…`) 외부 결합 gate(`fragmenter.audit_forming_bonds`)가 모두 실패한다.
  1. D1_build 산출물에 TS 원자 순서와 맞는 원자 번호 SMILES가 있으면 쓴다.
  2. 없으면 만든다: 반응물 = DFT TS 결합 그래프에서 `formed_pairs_ts` 두 결합을 뺀 두 조각, 원자 번호 = TS 원자 번호 + 1, 결합 차수·전하는 라벨 `rxn_smiles` 반응물과 그래프 동형으로 맞춰 옮긴다. 생성물 = 반응물 + 두 결합.
  3. **gate G-2:** 8,659개 모두에서 만든 반응물의 heavy 그래프(원소 + H 수)가 라벨 반응물과 동형이고, `audit_forming_bonds`가 찾은 형성 결합이 `formed_pairs_ts`와 같다. 실패 > 87개(1 %)면 **STOP**, 그 아래면 실패 행만 빼고 보고.
- **환경 변수 전수 점검:** `g1_geom`, `xtb_slice`, `ext_features`, `aggregate`, `r6_partition`, `r6_recover`, `rev5_common`, `rev6_common`에서 `os.environ`을 읽는 곳을 모두 찾아 `results/ENV_D1.md`에 변수별 D1 값을 적는다. 알려진 것: `ESPLEY_LABELS`, `ESPLEY_PROF`, `ESPLEY_SMILES`, `ESPLEY_META`, `ESPLEY_PARTITION=label`, `G1_ROOT`, `G1_ROOT_REV4`, `G1_ROOT_R6`, `ESPLEY_OUT`, `R5_SCRATCH`, `R6_SCRATCH`, `D1_SNAPSHOT`. rev 4–6 경로를 가리키는 기본값은 모두 새 scratch로 돌린다. **쓰기 경로가 rev 4–6 scratch나 `results_rev4/5/6/` 안이면 STOP.**

### 7-3. 어댑터 검사
- **G-0 (D0 쪽):** D0 반응 5개(seed 20261019, rev 6 R_feat에서)를 어댑터로 준비한 ORCA·xtb 입력이 rev 4 입력과 경로 빼고 같다.
- **G-1 (D1 쪽, 8,659개):** (a) `sp/eda.inp` 좌표 = `ts.xyz`(같은 순서, ≤ 1e-6 Å) (b) `(1)` 원자 수 = `n_f1`, `(2)` = `n_f2` (c) DFT TS에서 `formed_pairs_ts` 두 결합 길이를 **정렬해서** 라벨 `[formed_d1, formed_d2]`와 1e-3 Å 안에서 같다(라벨은 formed_d1 ≤ formed_d2로 정렬돼 있다). (d) 진단: `r6_partition.dft_redetect_matches` 결과 기록. (a)–(c) 위반 시 **STOP**.

### 7-4. 실패 되살리기
- rev 6 Phase R 사다리(`results_rev6/R_plan.md`)를 같은 gate로 적용한다. **`r6_recover.py`는 그대로 돌리지 않는다**(`results_rev6/` 파일을 덮어쓴다).
- `d1_recover.py`: `r6_recover`의 LADDERS와 gate 판정 함수만 import하고, `analysis/d1_pair/results/`(`G_attempts.csv`, `G_by_reason.csv`, `G_report.json`)와 새 scratch에만 쓴다. 출발 구조 = D1 첫 G1 시도 결과. 순서 = 기준 → anchor → job_id. 되살린 행에 `recovered_by` 열.
- 보고: 사유별 실패·되살림 표, 되살린 TS와 DFT TS의 heavy RMSD·형성 결합 차이.
- **STOP:** 기준 반응 157개 중 16개 이상 실패, 또는 한 core의 기준 짝이 절반 넘게 사라짐.

---

## 8. Phase F — D1 feature 182개
- rev 6 `r6_features.cmd_g1feat`과 같은 순서: `xtb_slice.process_one(label, "g1")`(72열) → `aggregate.derive`(ESPLEY73, `is_charged` 등) → `ext_features.run_rxn`(`r5-ext-3`, B1–B6 109열) = 182열.
- 캐시·중간 파일(`ESPLEY_OUT`, `R5_SCRATCH/ext`)은 새 scratch에. 출력 `$D1PAIR_SCRATCH/X182_d1.parquet`(rxn_id + 182열, `X182_all.parquet`와 같은 열 순서).
- **F-0 (재현):** rev 6 성공 D0 반응 20개(seed 20261019, rxn 4994 제외)의 rev 6 G1 기하를 어댑터 경로로 다시 계산해 `X182_all.parquet`와 비교: B1 외 ≤ 1e-8, B1(tblite) ≤ 1e-6. 넘으면 **STOP**.
- **F-1:** 182열 모두 유한값. B1 엔진 일치 gate(1e-6 Eh) 실패 행은 실패로 남긴다.
- 보고: feature 성공 수, P_feat·P_main·A_feat·A_main 짝 수, anchor가 바뀐 계열, 사라진 짝의 사유, §0-2 표의 실제 수. 한 core의 기준 짝이 절반 넘게 사라지면 **STOP**.

---

## 9. Phase M — 학습
- 단위 = arm(B, C) × 모델 6 × 타깃 10 × 바깥 fold, 분할 T1·LODO·LOSO 각각. T2는 arm B × 모델 6 × 타깃 10 × fold 5. arm P와 B−ref는 KRR × 타깃 10 × T1 fold 5.
- g 학습 짝 = 바깥 학습 쪽 T_main(T1 한 fold 약 6,700짝; SVR·XGB는 양방향이라 두 배).
- arm P: rev 6 `GRIDS["KRR_rbf"]` + `_make_pipe`(단일 반응 모델이라 평균 빼는 스케일 사용). 학습 행 = D0 4,963 + 학습 쪽 D1 반응(약 7,000). 안쪽 튜닝 = D1 반응을 core로 묶은 GroupKFold(4), D0 행은 모든 안쪽 학습 fold에 넣는다. 커널이 약 12,000²(float64 약 1.2 GB, GridSearch 중 여러 개)이므로 `sbatch --mem`을 넉넉히 잡고 기록한다.
- arm A·null은 f0 예측에서 계산한다.
- 단위마다 캐시(예측 parquet + 완료 JSON, 원자적 쓰기, idempotent). 입력 지문(데이터 sha256, 열, grid, fold, 패키지·코드 버전)이 다른 캐시를 만나면 exit 3.
- 예측 저장 `M_predictions.parquet`: pair_id, ref_kind, fold, split, arm, model, target, y, yhat. train 예측은 MAE용으로 요약 저장.
- gate M-1은 단위마다 확인한다. 시험 짝은 튜닝·대표 모델 선택·스케일 계산에 쓰지 않는다. 첫 실행에 튜닝 캐시에 시험 pair_id가 없는지 감사하고, 나오면 **STOP**.

---

## 10. Phase E — 평가와 그림

### 10-1. 산출
- `E_metrics.csv`: arm × 모델 × 타깃 × 분할 × 위치 × ref_kind의 train/test MAE, RMSE, R²_fam·ρ 오차(기준 짝만)
- `E_primary_tests.csv`(주 비교 9개), `E_secondary_tests.csv`(부 비교, T2 대 T1)
- `E_representative.json`, `E_stages.csv`(단계 판정)
- `E_bends.csv`: 시험 para 계열 × 채널 × 분할(T1, T2)의 DFT/예측 꺾임·V 판정, ψ, 두 팔 기울기
- `E_data_flow.csv`: D0·D1 라벨 → 제외 → G1 → feature → hygiene → 학습의 실제 수
- `E.json`: 설정, 입력 sha256, gate 기록

### 10-2. 그림 (183 mm, 7 pt, PDF + PNG 300 dpi + 그린 값 CSV)

| 그림 | 내용 |
|---|---|
| fig1_mae | 타깃 × arm(null, A, B, C, P) 대표 모델 MAE, 위치별 |
| fig2_parity | B 대표 모델 ΔΔ 패리티, 채널별(para 짝) |
| fig3_hammett | 미리 정한 6계열의 Hammett(DFT 점 vs B 예측, 장벽·궤도·정전기·cpcm): `nitrile oxide/C/R2//O-//`, `nitrile oxide/C/R1////`, `nitrile oxide/Cp/R1////`, `azide/D/R1////`, `carbonyl ylide/B/R1/c//exo/trans`, `azomethine ylide/Bp/R1/c//endo/` |
| fig4_bends | 채널별 꺾임 판정 일치(민감도, 위양성률) |
| fig5_r2fam | 채널별 R²_fam 분포(A, B, C) |
| fig6_lodo | T1 대 LODO MAE |
| fig7_vblind | 숨은 V 6계열: V가 나온 채널의 Hammett(DFT 점, T1 예측, T2 예측)와 장벽 Hammett |

- fig3 첫 계열은 실험 V자 Hammett(nitrile oxide + para-X-phenylacetylene, ρ = −0.34 / +0.61; J. Chem. Soc. B 1967, doi:10.1039/J29670000867)과 같은 반응 종류다. 쌍극자·용매가 달라 직접 비교가 아님을 그림 설명에 적는다.

---

## 11. Phase W — `results/SUMMARY.md` (한국어)
- 맨 앞 세 문장: 무엇을 했는지, 주 비교 결론(9개, Holm 후), 단계 판정 한 줄.
- 절 순서: 데이터 사용 표 → 주 비교 → 단계 판정 → 위치별 → Hammett 수준(R²_fam, ρ, 꺾임) → T2(V를 본 적 없는 모델) → 부 비교(C, P, B−ref, LODO·LOSO, Σ장벽) → G·F 행 흐름 → 이탈 → 한계 → 산출물 → 재현.
- 수치는 모두 `results/` 파일에서 옮긴다.
- 반드시 적을 한계: 구조 출발점이 DFT TS(G1)라 SMILES에서 시작하는 예측이 아니다 · 채널 값은 분해 방식에 따른 양이다(관측량 아님) · 같은 반응물 분자가 여러 core에 나온다 · 숨은 V는 6계열(5개가 nitrone)뿐이다 · 실패·되살림 행 · 모델별 반대칭 구현 차이.
- 이탈과 HPC 규칙 위반은 자진 보고한다.

---

## 12. 규칙과 금지
- **HPC:** 모든 계산은 `sbatch`, `#SBATCH --time=48:00:00`. 로그인 노드에서 python·xtb·ORCA·루프·`nohup`·폴러 금지. 제출 전 `squeue -u $USER -h | wc -l`; MaxJobs 10, MaxSubmit 20(array 원소 하나씩 셈), 20을 넘으면 `--dependency=afterany` 사슬. CPU 파티션은 `sinfo`로 고른다.
- idempotent, 원자적 쓰기(`rev5_common.write_atomic`). BLAS 스레드 1, 병렬은 `n_jobs`로만. rev 6 환경(xtb 6.7.1 `edcfbbe`, tblite 0.7.0, ORCA 6.1.1, NN은 `env_espley`)과 버전 기록. 수치는 파일에서 직접 읽는다.
- **하지 말 것:** g 입력에 σ·계열·core·X 이름 넣기 · f0를 D1으로 다시 학습 · f0 학습 행을 행 C로 줄이기 · 시험 점수로 모델·grid·feature 고르기(feature는 182개 고정) · g에 평균 빼는 스케일러 · G1 gate 완화 · 제외 461행(D'' 포함) 사용 · anchor 짝을 주 지표·Hammett·단계 판정·fold 배정에 넣기 · 라벨·rev 4–6 파일·D1 job 폴더 덮어쓰기 · 결과를 본 뒤 arm·모델·짝·지표·기준 바꾸기.

## 13. STOP 조건

| Phase | 조건 |
|---|---|
| 0 | 입력·부록 sha256 불일치; D1 job 폴더·파일 없음 |
| G | G-0, G-1 실패; G-2 실패 > 87; 쓰기 경로가 rev 4–6 폴더 안; 기준 반응 실패 > 10 %; 한 core의 기준 짝이 절반 넘게 사라짐; R_plan에 없는 설정 |
| F | F-0 실패; 이 단계 뒤 한 core의 기준 짝이 절반 넘게 사라짐 |
| P | P-1, P-2, P-3 개수·목록 불일치; P-4 입체 불일치 |
| F0 | F0-1 실패; 학습 행 ≠ 4,963 |
| S | T2 목록·개수 불일치; 누출 감사 실패; PREREG commit 전 학습 시도 |
| M | M-1 실패; 캐시 지문 불일치; 튜닝 캐시에 시험 pair_id |

---

# 부록

부록 코드는 이 사양 작성 환경(Python 3, pandas, RDKit)에서 최종 `labels_d1.json`으로 실행해 §2의 기대값을 얻은 것이다. 그대로 쓴다.

## 부록 A. `build_rowattr.py` (sha256 `81cb660e8b0852fd6d114d179955039b80b422c056752627dca0b432a4ef86d4`)
사용법: `python -I build_rowattr.py labels_d1.json sigma_hansch1991.csv rowattr.pkl` → `rowattr.pkl`(남은 행), `rowattr.pkl.all.pkl`(전체)

```python
"""Rebuild the row-attribute table from labels_all.json (container was replaced; earlier rowattr.pkl lost).
Columns rebuilt: dipole/dipolarophile SMILES, reference excess ex1/ex2, flags, Xfull/Xb/pos/sigma, cfg (= TS dipole
configuration tag), orient (which dipole terminus the more-substituted dipolarophile carbon bonds to; from the
product structure, no atom map needed), c3 (cis/trans of X on C3 vs the neighbouring ring carbon's substituent).
usage: python -I build_rowattr.py labels_all.json sigma.csv out.pkl"""
import sys, json, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
H = 627.5094740631

d = pd.DataFrame(json.load(open(sys.argv[1])))
sg = pd.read_csv(sys.argv[2]).set_index("X")
for k in ["flag_async", "flag_short_forming", "flag_regio_ambiguous", "flag_small_extra_imag"]:
    d["fl_" + k] = d["flags"].map(lambda f: bool(f.get(k)) if isinstance(f, dict) else False)

DIP = {"nitrile oxide": "[#6]#[N+]-[O-]", "azide": "[#7]~[N+]~[#7-,#7]", "nitrone": "[#6]=[N+](-[#6])-[O-]",
       "azomethine ylide": "[#6-]-[N+]=[#6]", "carbonyl ylide": "[#6-]-[O+]=[#6]"}
PAT = {k: Chem.MolFromSmarts(v) for k, v in DIP.items()}


def split(row):
    react, prod = row.rxn_smiles.split(">>")
    comps = react.split(".")
    dip = [c for c in comps if Chem.MolFromSmiles(c).HasSubstructMatch(PAT[row.dipole])]
    assert len(dip) == 1, (row.ts_id, comps)
    return dip[0], [c for c in comps if c != dip[0]][0], prod


sp = d.apply(split, axis=1, result_type="expand"); d["dip_smi"], d["dph_smi"], d["prod_smi"] = sp[0], sp[1], sp[2]
d["dip_smi"] = d.dip_smi.map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)))
d["dph_smi"] = d.dph_smi.map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)))
# reference excess: this row's relaxed reference vs the lowest reference of the same molecule (kcal/mol)
d["ex1"] = (d.e_frag1_rel_eh - d.groupby("dip_smi").e_frag1_rel_eh.transform("min")) * H
d["ex2"] = (d.e_frag2_rel_eh - d.groupby("dph_smi").e_frag2_rel_eh.transform("min")) * H

d["cfg"] = d.ts_dipole_config.fillna("").astype(str)
d["Xfull"] = d.R1.where(d.series == "R1", d.R2)
d["Xb"] = d.Xfull.str.replace(r"^[omp]-", "", regex=True)
d["pos"] = np.where(d.Xfull == "Ph", "ref_aryl", np.where(d.Xfull == "H", "ref_direct", d.attach))
smap, pmap = d.Xb.map(sg.sigma_m), d.Xb.map(sg.sigma_p)
d["sigma"] = np.select([d.pos.isin(["ref_aryl", "ref_direct"]), d.pos == "m", d.pos.isin(["p", "direct", "o"])], [0.0, smap, pmap], np.nan)


# ---------------------------------------------------------------- orient (needs only the product + reactants)
def any_bond_query(m):
    """same heavy atoms (element only), any bond order: finds the reactant skeleton inside the product"""
    m2 = Chem.RWMol(m)
    for at in m2.GetAtoms():
        at.SetFormalCharge(0); at.SetNumExplicitHs(0); at.SetNoImplicit(False)
    m2 = m2.GetMol(); m2.UpdatePropertyCache(False)
    prm = Chem.AdjustQueryParameters.NoAdjustments(); prm.makeBondsGeneric = True
    return Chem.AdjustQueryProperties(m2, prm)


def backbone_in_product(row):
    """-> product mol, (pa, pc, pb) product indices of the dipole backbone, labels, 5-ring"""
    dip = Chem.MolFromSmiles(row.dip_smi); prod = Chem.MolFromSmiles(row.prod_smi)
    (a, c, b), lab = termini(dip, row.dipole)
    for mt in prod.GetSubstructMatches(any_bond_query(dip), useChirality=False, uniquify=False, maxMatches=5000):
        pa, pc, pb = mt[a], mt[c], mt[b]
        # every dipole atom keeps its hydrogens (HN3: NH end vs bare N end; nitrone/ylide: ring CH2 vs N-CH3)
        if any(prod.GetAtomWithIdx(mt[i]).GetTotalNumHs() != dip.GetAtomWithIdx(i).GetTotalNumHs() for i in range(dip.GetNumAtoms())):
            continue
        ring = [set(r) for r in prod.GetRingInfo().AtomRings() if len(r) == 5 and {pa, pc, pb} <= set(r)]
        if ring:
            return prod, (pa, pc, pb), {pa: lab[a], pb: lab[b]}, ring[0]
    return None


def termini(dip, name):
    """backbone atoms of the dipole (a, centre, b) and the terminus labels: anionic end -> 'O-'/'N-'/'C-', other -> symbol"""
    mt = dip.GetSubstructMatches(PAT[name])[0]
    if name == "nitrone":
        a, c, b = mt[0], mt[1], mt[3]
    else:
        a, c, b = mt[0], mt[1], mt[2]
    lab = lambda i: dip.GetAtomWithIdx(i).GetSymbol() + ("-" if dip.GetAtomWithIdx(i).GetFormalCharge() < 0 else "")
    return (a, c, b), {a: lab(a), b: lab(b)}


def heaviness(m, i, exclude):
    """substituent weight around atom i (radius 1, then 2, 3) excluding the given atoms"""
    out = []
    for rad in (1, 2, 3):
        env = set(); front = {i}; seen = {i} | set(exclude)
        for _ in range(rad):
            nxt = set()
            for f in front:
                for n in m.GetAtomWithIdx(f).GetNeighbors():
                    if n.GetIdx() not in seen and n.GetAtomicNum() > 1:
                        nxt.add(n.GetIdx())
            seen |= nxt; env |= nxt; front = nxt
        out.append(sum(m.GetAtomWithIdx(j).GetMass() for j in env))
    return tuple(out)


def orient(row):
    try:
        bb = backbone_in_product(row)
        if bb is None:
            return "?"
        prod, (pa, pc, pb), lab, ring = bb
        dc = [i for i in ring if i not in (pa, pc, pb)]
        ca = [i for i in dc if prod.GetBondBetweenAtoms(i, pa)][0]
        cb = [i for i in dc if prod.GetBondBetweenAtoms(i, pb)][0]
        ha, hb = heaviness(prod, ca, ring), heaviness(prod, cb, ring)
        if ha == hb:
            return "sym"
        return lab[pa] if ha > hb else lab[pb]
    except Exception:
        return "?"


two = d.regio.isin(["r1/2", "r2/2"])
d["orient"] = ""
d.loc[two, "orient"] = d[two].apply(orient, axis=1)


# ---------------------------------------------------------------- C3 orientation (nitrone / AY / CY, X on the dipole)
def c3_orientation(row):
    try:
        bb = backbone_in_product(row)
        if bb is None:
            return "?"
        prod, (pa, pc, pb), lab, ring = bb
        p = Chem.AddHs(prod)
        c3 = None
        for t in (pa, pb):
            at = p.GetAtomWithIdx(t)
            if at.GetSymbol() == "C" and any(n.GetAtomicNum() > 1 and n.GetIdx() not in ring for n in at.GetNeighbors()):
                c3 = at
        if c3 is None:
            return ""
        xs = [n for n in c3.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in ring][0]
        c4 = [n for n in c3.GetNeighbors() if n.GetIdx() in ring and n.GetIdx() not in (pa, pc, pb)][0]
        ms4 = [n for n in c4.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in ring]
        if len(ms4) != 1 or c4.GetHybridization() != Chem.HybridizationType.SP3:
            return ""
        if AllChem.EmbedMolecule(p, randomSeed=7) != 0:
            return "?"
        AllChem.MMFFOptimizeMolecule(p); pos = p.GetConformer().GetPositions()
        rp = pos[sorted(ring)]; nrm = np.linalg.svd(rp - rp.mean(0))[2][-1]
        u = np.dot(pos[xs.GetIdx()] - pos[c3.GetIdx()], nrm); v = np.dot(pos[ms4[0].GetIdx()] - pos[c4.GetIdx()], nrm)
        return "cis" if u * v > 0 else "trans"
    except Exception:
        return "?"


need = d.dipole.isin(["nitrone", "azomethine ylide", "carbonyl ylide"]) & (d.series == "R1") & d.skeleton.isin(["AppE", "AppZ", "B", "Bp"])
d["c3"] = ""
d.loc[need, "c3"] = d[need].apply(c3_orientation, axis=1)

drop = (d.skeleton.isin(["DppC", "DppT"]) | (d.ex1 > 20) | (d.ex2 > 20) | d.fl_flag_async | d.fl_flag_short_forming | d.fl_flag_regio_ambiguous)
print("rows", len(d), "D''", int(d.skeleton.isin(['DppC', 'DppT']).sum()), "ref excess>20 (non-D'')",
      int((((d.ex1 > 20) | (d.ex2 > 20)) & ~d.skeleton.isin(['DppC', 'DppT'])).sum()), "excluded", int(drop.sum()), "kept", int((~drop).sum()))
print("orient:", d.loc[two, "orient"].value_counts().to_dict())
print("c3:", d.loc[need, "c3"].value_counts().to_dict())
d.to_pickle(sys.argv[3] + ".all.pkl")
d[~drop].to_pickle(sys.argv[3])
print(d[~drop].pos.value_counts().to_dict())
```

## 부록 B. `d1_pairs.py` (sha256 `424c2bc24b4ed5111c35a00688a88d43f47dffb2e61e50c5326256f31be603e1`)
사용법: `python -I d1_pairs.py rowattr.pkl sigma_hansch1991.csv pairs_d1.csv`

```python
"""d1_pairs.py — D1 matched pairs for the pair-difference model.

pair p = (a, b): a = substituted reaction, b = its reference in the same reaction family.
  family key F = (dipole, skeleton, series, cfg, orient, face, c3)   (columns of rowattr.pkl, build_rowattr.py)
  aryl X (pos p / m / o): b = the unsubstituted-phenyl row (pos ref_aryl) with the same F; F must be unique
  direct X (pos direct):  b = the H row (pos ref_direct) with the same dipole, skeleton, face, narrowed by orient,
                          cfg, c3 only when every candidate H row has that field; exactly one candidate required
  rows / references that are duplicated within their key are dropped (keep=False)
dd_<t> = y_a - y_b in kcal/mol for every target t (raw energies, NOT log units; Hammett value = -dd / 1.3642)
core = (dipole, skeleton) -> outer CV group: every pair whose a or b belongs to a core is in that core's fold
usage: python -I d1_pairs.py rowattr.pkl sigma.csv out_pairs.csv"""
import sys, numpy as np, pandas as pd

TARGETS = {"d1": "d1_kcal", "d2": "d2_kcal", "elst": "elst_dft", "pauli": "pauli_dft", "oi": "oi_dft",
           "disp": "disp_dft", "cpcm": "cpcm_dft", "cds": "cds_dft", "c_ghost": "bsse_shift_kcal",
           "barrier": "barrier_kcal"}
F = ["dipole", "skeleton", "series", "cfg", "orient", "face", "c3"]


def build(r, sg):
    r = r.copy()
    for c in F:
        r[c] = r[c].fillna("").astype(str)
    keep = ["job_id", "rxn_id", "Xb", "pos"] + F + list(TARGETS.values())
    ref = r[r.pos == "ref_aryl"]
    ref = ref[~ref.duplicated(F, keep=False)].set_index(F)
    out = []
    for pos in ("p", "m", "o"):
        q = r[r.pos == pos]
        q = q[~q.duplicated(F + ["Xb"], keep=False)]
        j = q[keep].join(ref[["job_id", "rxn_id"] + list(TARGETS.values())].add_suffix("_b"), on=F, how="inner")
        out.append(j)
    H = r[r.pos == "ref_direct"]
    rows = []
    for _, x in r[r.pos == "direct"].iterrows():
        c = H[(H.dipole == x.dipole) & (H.skeleton == x.skeleton) & (H.face == x.face)]
        for col in ("orient", "cfg", "c3"):
            if len(c) > 1 and (c[col] != "").all():
                c = c[c[col] == x[col]]
        if len(c) == 1:
            h = c.iloc[0]
            rows.append({**{k: x[k] for k in keep}, **{f"{v}_b": h[v] for v in ["job_id", "rxn_id"] + list(TARGETS.values())}})
    out.append(pd.DataFrame(rows))
    P = pd.concat(out, ignore_index=True)
    P = P.rename(columns={"job_id": "a_job", "rxn_id": "a_rxn", "job_id_b": "b_job", "rxn_id_b": "b_rxn"})
    for t, col in TARGETS.items():
        P[f"dd_{t}"] = P[col] - P[col + "_b"]
    P["fam"] = P[F].agg("/".join, axis=1)
    P["core"] = P.dipole + "/" + P.skeleton
    P["sigma"] = np.where(P.pos == "m", P.Xb.map(sg.sigma_m), P.Xb.map(sg.sigma_p))
    P = P.drop(columns=[c for c in P.columns if c.endswith("_b") and c not in ("b_job", "b_rxn")] + list(TARGETS.values()))
    P.insert(0, "pair_id", [f"P{i:05d}" for i in range(len(P))])
    return P


if __name__ == "__main__":
    r = pd.read_pickle(sys.argv[1]); sg = pd.read_csv(sys.argv[2]).set_index("X")
    P = build(r, sg)
    tmp = sys.argv[3] + ".tmp"; P.to_csv(tmp, index=False)
    import os; os.replace(tmp, sys.argv[3])
    print("pairs", len(P), P.pos.value_counts().to_dict())
    print("families", P.fam.nunique(), "| para families with >= 12 X having sigma_p:",
          int((P[(P.pos == "p") & P.sigma.notna()].groupby("fam").Xb.nunique() >= 11).sum()))
    print("cores", P.core.nunique(), "| pairs per core min/median/max:", P.groupby("core").size().agg(["min", "median", "max"]).to_dict())
    print("pairs by dipole:", P.dipole.value_counts().to_dict())
    print("distinct a rows", P.a_job.nunique(), "| distinct references", P.b_job.nunique())
```

## 부록 C. `sigma_hansch1991.csv` (sha256 `2ba63224928301657f09707cb3be1b0e05a6c69ef1e733b490b9a4ac6261d06f`)
Hansch, Leo & Taft, Chem. Rev. 1991, 91, 165 (doi:10.1021/cr00002a004)의 σm, σp, F, R. Ts는 표에 없으므로 σ가 필요한 평가에서만 빠진다(학습에는 들어간다).

```
X,sigma_m,sigma_p,F,R
H,0.00,0.00,0.03,0.00
Br,0.39,0.23,0.45,-0.22
Cl,0.37,0.23,0.42,-0.19
F,0.34,0.06,0.45,-0.39
NO2,0.71,0.78,0.65,0.13
OH,0.12,-0.37,0.33,-0.70
NH2,-0.16,-0.66,0.08,-0.74
CF3,0.43,0.54,0.38,0.16
CN,0.56,0.66,0.51,0.15
CHO,0.35,0.42,0.33,0.09
Me,-0.07,-0.17,0.01,-0.18
OMe,0.12,-0.27,0.29,-0.56
SMe,0.15,0.00,0.23,-0.23
Ac,0.38,0.50,0.33,0.17
CO2Me,0.37,0.45,0.34,0.11
NMe2,-0.16,-0.83,0.15,-0.98
SiMe3,-0.04,-0.07,0.01,-0.08
tBu,-0.10,-0.20,-0.02,-0.18
Ph,0.06,-0.01,0.12,-0.13
Bn,-0.08,-0.09,-0.04,-0.05
```

## 부록 D. `hammett_eval.py` (sha256 `e87d944ef6b35f1a8ae44884270a2d5530bbc32f1f53a55ef7e194ffbffac74b`)
사용법(자기 검사): `python -I hammett_eval.py pairs_d1.csv` → §5-2의 기대값과 같아야 한다.

```python
"""hammett_eval.py — Hammett-level evaluation of pair predictions (same bend rule as the D1 analysis, hidden_kink.py).

series of one para family: points (sigma_p of X, y_X) with y = -dd / 1.3642 (log k units) plus H at (0, 0)
bend call: one hinge at psi between neighbouring sigma values, >= 3 points each side,
           AICc(line) - AICc(hinge) >= 4, and still >= 2 after removing any one non-H point
V call:    bend call and the two arms have opposite signs, both |arm| >= 0.46
usage (self-test on DFT values): python -I hammett_eval.py pairs_d1.csv"""
import sys, numpy as np, pandas as pd

K = 1.3642
CH = ["barrier", "d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds"]


def aicc(rss, n, k):
    return n * np.log(np.maximum(rss, 1e-12) / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def bent_gain(x, Y):
    """x (n,), Y (n, m) -> gain (m,), psi (m,), left slope (m,), right slope (m,)"""
    n = len(x)
    A = np.c_[np.ones(n), x]
    rl = ((Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]) ** 2).sum(0)
    xs = np.unique(x); best = np.full(Y.shape[1], np.inf); psi = np.full(Y.shape[1], np.nan)
    L = np.full(Y.shape[1], np.nan); R = np.full(Y.shape[1], np.nan)
    for i in range(len(xs) - 1):
        p = (xs[i] + xs[i + 1]) / 2
        if (x < p).sum() < 3 or (x > p).sum() < 3:
            continue
        B = np.c_[np.ones(n), x, np.maximum(0, x - p)]
        C = np.linalg.lstsq(B, Y, rcond=None)[0]
        rss = ((Y - B @ C) ** 2).sum(0); better = rss < best
        best[better] = rss[better]; psi[better] = p; L[better] = C[1, better]; R[better] = C[1, better] + C[2, better]
    return aicc(rl, n, 3) - aicc(best, n, 5), psi, L, R


def calls(x, Y, names):
    """bend and V calls for every column of Y; names = point labels ('H' is never removed)"""
    g, psi, L, R = bent_gain(x, Y)
    ok = g >= 4
    for i, nm in enumerate(names):
        if nm == "H" or not ok.any():
            continue
        keep = np.arange(len(x)) != i
        g2, *_ = bent_gain(x[keep], Y[keep][:, ok])
        ok[np.where(ok)[0][g2 < 2]] = False
    v = ok & (L * R < 0) & (np.abs(L) >= 0.46) & (np.abs(R) >= 0.46)
    return ok, v, g, psi, L, R


def family_series(P, fam, col, min_x=12):
    """x (sigma_p) and y (log units) of one para family from pair rows P (column col = dd in kcal); H appended"""
    t = P[(P.fam == fam) & (P.pos == "p") & P.sigma.notna()].drop_duplicates("Xb")
    if len(t) + 1 < min_x:
        return None
    x = np.r_[t.sigma.to_numpy(float), 0.0]; y = np.r_[-t[col].to_numpy(float) / K, 0.0]
    return x, y, list(t.Xb) + ["H"]


def rho_r2(x, y):
    c = np.polyfit(x, y, 1); r2 = 1 - ((y - np.polyval(c, x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return c[0], r2


if __name__ == "__main__":
    P = pd.read_csv(sys.argv[1])
    fams = sorted(P[P.pos == "p"].fam.unique()); rows = []
    for f in fams:
        s = family_series(P, f, "dd_barrier")
        if s is None:
            continue
        x, _, names = s
        Y = np.c_[[family_series(P, f, f"dd_{c}")[1] for c in CH]].T
        ok, v, *_ = calls(x, Y, names)
        rows.append(dict(fam=f, **{f"bent_{c}": bool(ok[k]) for k, c in enumerate(CH)}, **{f"V_{c}": bool(v[k]) for k, c in enumerate(CH)}))
    T = pd.DataFrame(rows)
    print("para families evaluated:", len(T))
    print("DFT bend calls per target:", {c: int(T[f"bent_{c}"].sum()) for c in CH})
    print("DFT V calls per target:", {c: int(T[f"V_{c}"].sum()) for c in CH})
    hid = (~T.bent_barrier) & T[[f"bent_{c}" for c in CH[1:]]].any(axis=1)
    print("barrier straight & >= 1 channel bent:", int(hid.sum()))
```

## 부록 E. `d1_anchor_pairs.py` (sha256 `f979e1e336df850a0b55d6ed762718b035bf2b151468263919b53a5aa40bac18`)
사용법: `python -I d1_anchor_pairs.py rowattr.pkl sigma_hansch1991.csv pairs_d1.csv pairs_d1_anchor.csv` → §2-3의 기대값과 같아야 한다. A_feat는 `rowattr.pkl`에서 182 feature가 있는 행만 남긴 파일로 다시 실행한다.

```python
"""d1_anchor_pairs.py — extra D1 pairs for families that have no usable reference (training data for g).

Why: d1_pairs.py pairs every substituted row with the X = Ph (aryl) or X = H (direct) reference of its family.
Rows whose family has no such reference are left out (2026-10-19: 1,011 aryl + 297 direct rows). A family of n rows
holds n - 1 independent differences, so one anchor row per family keeps all of them.

family key F = (dipole, skeleton, series, cfg, orient, face, c3), separately for aryl rows (pos p / m / o) and direct rows
  aryl families:   F has no row with pos ref_aryl at all (families whose reference is duplicated are stereo-mixed -> skipped)
  direct families: direct rows with no H candidate in d1_pairs.py (same dipole, skeleton, face) -> grouped by F
rows duplicated within F + Xb + pos are dropped (keep=False), as in d1_pairs.py
anchor b = first row after sorting by: passes the rev 6 hygiene rule (d1 >= 0, d2 >= 0, d2 <= 50) first,
  then |sigma_p(Xb)| (missing sigma last), pos p < m < o < direct, Xb; a = every other row
families with one row give no pair
output columns = pairs_d1.csv columns + ref_kind ("anchor"); pair_id A00000...; the Hammett analysis never uses these rows
usage: python -I d1_anchor_pairs.py rowattr.pkl sigma.csv pairs_d1.csv out_anchor.csv"""
import sys, os, numpy as np, pandas as pd

TARGETS = {"d1": "d1_kcal", "d2": "d2_kcal", "elst": "elst_dft", "pauli": "pauli_dft", "oi": "oi_dft",
           "disp": "disp_dft", "cpcm": "cpcm_dft", "cds": "cds_dft", "c_ghost": "bsse_shift_kcal",
           "barrier": "barrier_kcal"}
F = ["dipole", "skeleton", "series", "cfg", "orient", "face", "c3"]
POS_ORDER = {"p": 0, "m": 1, "o": 2, "direct": 3}


def build(r, sg):
    r = r.copy()
    for c in F:
        r[c] = r[c].fillna("").astype(str)
    refk = set(map(tuple, r.loc[r.pos == "ref_aryl", F].to_numpy()))
    ar = r[r.pos.isin(["p", "m", "o"])]
    ar = ar[[tuple(t) not in refk for t in ar[F].to_numpy()]]
    H = r[r.pos == "ref_direct"]
    dr = r[r.pos == "direct"]
    noh = [len(H[(H.dipole == x.dipole) & (H.skeleton == x.skeleton) & (H.face == x.face)]) == 0 for _, x in dr.iterrows()]
    dr = dr[noh]
    rows = []
    for kind, q in (("aryl", ar), ("direct", dr)):
        q = q[~q.duplicated(F + ["Xb", "pos"], keep=False)].copy()
        q["_s"] = q.Xb.map(sg.sigma_p).abs().fillna(np.inf)
        q["_p"] = q.pos.map(POS_ORDER)
        q["_h"] = (~((q.d1_kcal >= 0) & (q.d2_kcal >= 0) & (q.d2_kcal <= 50))).astype(int)
        for _, g in q.groupby(F, sort=True):
            if len(g) < 2:
                continue
            g = g.sort_values(["_h", "_s", "_p", "Xb"])
            b = g.iloc[0]
            for _, a in g.iloc[1:].iterrows():
                d = {"a_job": a.job_id, "a_rxn": a.rxn_id, "Xb": a.Xb, "pos": a.pos, **{c: a[c] for c in F},
                     "b_job": b.job_id, "b_rxn": b.rxn_id, "anchor_Xb": b.Xb, "anchor_pos": b.pos}
                for t, col in TARGETS.items():
                    d[f"dd_{t}"] = a[col] - b[col]
                rows.append(d)
    A = pd.DataFrame(rows)
    A["fam"] = A[F].agg("/".join, axis=1) + "/anchor"
    A["core"] = A.dipole + "/" + A.skeleton
    A["sigma"] = np.where(A.pos == "m", A.Xb.map(sg.sigma_m), A.Xb.map(sg.sigma_p))
    A["ref_kind"] = "anchor"
    A.insert(0, "pair_id", [f"A{i:05d}" for i in range(len(A))])
    return A


if __name__ == "__main__":
    r = pd.read_pickle(sys.argv[1]); sg = pd.read_csv(sys.argv[2]).set_index("X"); P = pd.read_csv(sys.argv[3])
    A = build(r, sg)
    assert not (set(A.a_job) | set(A.b_job)) & (set(P.a_job) | set(P.b_job)), "anchor rows overlap the reference pairs"
    tmp = sys.argv[4] + ".tmp"; A.to_csv(tmp, index=False); os.replace(tmp, sys.argv[4])
    print("anchor pairs", len(A), A.pos.value_counts().to_dict())
    print("anchor families", A.fam.nunique(), "| rows used", len(set(A.a_job) | set(A.b_job)))
    print("cores", A.core.nunique(), "| cores not in reference pairs:", sorted(set(A.core) - set(P.core)))
    print("pairs by dipole:", A.dipole.value_counts().to_dict())
```
