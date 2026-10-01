# PREREG_REV6 — rev 6 사전 등록 (모든 학습 전, 2026-10-02)

명세: `docs/specs/REV6_FAIR_ESPLEY_COMPARISON.md` §7. 이 문서와 `prereg_rev6.json`을 commit 한 뒤에만 학습이 돈다.
학습 스크립트(`r6_train.py`, `r6_espc.py arm`, `r6_analyze.py`)는 시작할 때 `rev6_common.gate_prereg()`로 두 파일이 commit 되어
있고 바뀌지 않았는지, `prereg_rev6.json`의 입력 파일 sha256이 그대로인지 확인한다(아니면 exit 3).
상수의 원본은 `rev6_common.py`(이 commit에 포함)다. 아래 값은 그 파일과 같다.

이 시점까지 본 것:
- Phase R: 되살리기와 gate 기록(`R2_report.json`, `R5_report.json`).
- Phase P:
  - P-1 재현(`P1_reproduction.json`): sklearn 75/75가 3.9e-11 안에서 일치했다. NN 50/50은 비트 단위로 일치하지 않았고, 사용자 결정으로 수용했다. 그들의 NN 튜닝은 keras_tuner 1.4.5에서 돌지 않아 사용자 결정으로 1.4.7을 쓴다.
  - P-2(`P2_*`): 우리 NN 재구현이 같은 노드의 그들 코드와 50/50 완전 일치했다(기본 스레드·운영 스레드 모두). 패치·래퍼 경로는 ml 실행이 끝났고, 그 비교와 튜닝 비교는 이 commit 시점에 진행 중이다(§8 이탈 9).
  - P-3 역할 검사(`P3_role_check.json`).
- Phase S: 행, 분할, 라벨 관계.

rev 6 모델은 어느 것도 학습하지 않았다. P-1·P-2는 그들의 데이터·타깃으로 그들의 코드를 돌린 것이고, 우리 라벨로 학습한 모델이 아니다.

## 1. 행과 분할 (Phase S)

| 항목 | 값 |
|---|---|
| R_esp | Espley ds3 3,510행 (`manual_tt_solvent.pkl`, ds3 행 순서 = 위치 0..3509) |
| R_lab | 3,509 (라벨 없는 1행) |
| R_feat (∩ R_lab) | 3,390 (Phase R 뒤; 빠진 119행 = G1 TS 되살리기 실패) |
| **C (주 분석)** | **3,386** = R_lab ∩ R_feat ∩ R_hyg (hygiene 4행 제외) |
| C_all (민감도) | 3,390 = R_lab ∩ R_feat |
| C_nocref (민감도) | 3,375 = C − `constrained_ref` 행 11 |
| D (편향 점검) | 119 = (R_lab ∩ R_hyg) \ R_feat |
| 부록 A 행 | 4,963 = 라벨 ok ∩ R_feat ∩ R_hyg (Espley 반응이 아닌 행 포함) |

seed별 train_s / test_s / val_s(C): 22 = 2,705 / 341 / 340, 23 = 2,705 / 341 / 340, 14 = 2,710 / 334 / 342,
1 = 2,707 / 340 / 339, 2 = 2,707 / 338 / 341.

- 분할: seed 22, 23, 14, 1, 2. `rev6_common.espley_split3(3510, seed)` = Espley `_perform_train_test_split`
  (train 80 %, test = 20 % hold-out의 첫 절반, val = 둘째 절반), `compare_espley.espley_split`과 같음을 assert 했다.
  train_s / test_s / val_s = 각 부분 ∩ C (분할 순서 유지). 모든 arm·모델이 seed마다 같은 train_s로 학습하고 같은 test_s로 채점한다.
- 행·분할 파일과 sha256: `S_rows.csv`, `S_splits.csv`, `S_splits_appendixA.csv`, `S.json`(seed·부분·행 집합별 sha256).
- 라벨 관계(C, n = 3,386, `S_label_checks.json`)
  - d1 + d2 + eint_spe − barrier: 최대 |차| 5.0e-10
  - Σ6채널 − e_bond: 최대 |차| 0.105 kcal/mol (ORCA EDA 표의 반올림, `SMD_RELABEL.md`)
  - eint_spe − e_bond − bsse_shift: 0
  - c_ghost: 평균 0.078 kcal/mol, sd 0.667. 그래서 Int는 6채널 합과 같지 않다(§1-1).
- test 행 보호(§12, `S.json` test_row_audit)
  - 우리 파이프라인의 튜닝 행(train_s, NN은 val_s도)은 그 seed의 test_s와 겹치지 않는다(assert).
  - Espley 프로토콜의 seed 23 튜닝 행과 다른 seed의 test_s가 겹치는 수: seed 22는 269/341, 14는 260/334, 1은 284/340, 2는 265/338. 알려진 특이점이고 Espley에 유리하다.

## 2. arm, 모델, 타깃

| arm | feature | 학습·튜닝 |
|---|---|---|
| ESP-C | Espley AM1 46 (+ 튜닝 전용 `sum_distortion_energies_am1` 1열, 그들의 코드 그대로), 두 AM1 변형 feature는 역할 순서(P-3 swap) | 그들의 코드 + P-4 패치 |
| OUR-O | 우리 182 (EXT_ALL) | 우리 파이프라인 |
| ESP-O | Espley AM1 46 (역할 순서) | 우리 파이프라인 |
| OUR-C | 우리 182 | 그들의 코드 + P-4 패치 (`espley_protocol()`) |

- 모델 6개: Ridge, KRR(rbf), SVR(rbf), XGB, NN-2, NN-4.
- 비교 타깃(Phase B): d1 = `dft_d1_kcal`(dipole), d2 = `dft_d2_kcal`(dipolarophile), Int = `dft_eint_spe_kcal`, ΔE‡ = `dft_barrier_kcal`.
  모든 arm이 같은 값(`labels_all.json`)으로 학습·채점한다. ΔG‡는 뺀다.
- 8채널 실험(Phase A): d1, d2, elst, Pauli, OI, disp, CPCM, CDS (+ Σ채널용 c_ghost = barrier − d1 − d2 − e_bond).

### 2-1. Espley 프로토콜 (ESP-C, OUR-C) — 그들의 코드를 실행한다

- 코드: 그들의 `hyperparameter_tuning/hyp_tuning.py`(sha256 d1640730…)와 `machine_learning/ml_analysis.py`(202ccba0…)에
  `P4_their_code_diff.patch`만 적용한 파일(sha256 `P4_patched_sha256.json`). 패치 내용 = (i) 그들의 분할 직후 R6_KEEP(train·val)과
  R6_KEEP_TEST(test)에 든 행만 남김, (ii) XGB 추가(`XGBRegressor(random_state=42, n_jobs=1, tree_method='hist')`,
  grid = `train_ml_single.GRIDS["XGB"]` 값). 그 밖의 줄은 그대로다.
- 정답 열 교체는 데이터셋으로 한다(코드 변경 없음): 그들의 열 이름에 우리 값 — `distortion_energy_1_dft` = d1,
  `distortion_energy_2_dft` = d2, `interaction_energies_dft` = eint_spe(우리 부호), `e_barrier_dft` = barrier,
  `sum_distortion_energies_dft` = d1 + d2(그들의 튜닝 코드만 쓰고 학습 코드는 버림, 그들과 같음). `q_barrier_dft`(ΔG‡)는 뺐다.
- hps: `hps.pkl`을 쓰지 않는다. 그들의 `hyp_tuning.py`를 우리 데이터셋으로 돌린다(seed 23 train ∩ C, GridSearchCV(cv=5),
  X만 표준화; NN은 keras_tuner Hyperband, 목적함수 = train MAE + val MAE, validation = seed 23 val ∩ C).
- 그들의 코드는 RF도 튜닝·학습한다(공개 `hps.pkl`·`ml_results.pkl`에는 RF가 없다). 코드를 고치지 않으므로 RF도 돌지만
  6개 모델에 들지 않고 보고에서 뺀다(부록에 수치만).
- 학습: seed마다 `ml_analysis.py <seed> <dataset> <hps>`; NN은 호출마다 하나씩(그들의 방식), NN 키 순서 × seed.
- 행: ESP-C test는 test ∩ (C ∪ D)로 예측하고(편향 점검용), 주 채점은 C 행만 쓴다. 모델은 test 행을 보지 않으므로
  C 행 예측은 test ∩ C만 넣었을 때와 같다.

### 2-2. 우리 파이프라인 (OUR-O, ESP-O)

- Ridge / KRR / SVR / XGB: rev 4/5 Protocol A와 같다 — `train_ml_single._make_pipe`(X, y 표준화) + `GRIDS`,
  seed마다 train_s에서 `GridSearchCV(cv=KFold(5, shuffle=True, random_state=seed), neg MAE)`.
- NN-2 / NN-4: 그들의 NN 정의·학습 코드(`r6_nn.py` = ml_analysis.py :523-570, hyp_tuning.py :368-456; P-2에서 그들의 출력과
  비교). seed마다: Hyperband(그들의 탐색 공간·설정, seed 23)를 train_s로, validation = val_s(그들과 같은 방식으로 hold-out
  둘째 절반 ∩ C) → 최적 hps로 train_s 전체에 500 epoch 학습(validation_data = val_s는 history만) → test_s 예측.
- 대표 모델용 검증 MAE: sklearn·XGB는 inner CV `best_score_`; NN은 튜닝한 hps로 train_s에서 KFold(5, shuffle, seed) 5-fold CV.

### 2-3. NN 처리가 네 arm에서 같은 점

- 같은 NN 정의(층, ELU, l2, dropout, Adam, MAE loss, batch 32, 500 epoch, shuffle False, X·y 표준화), 같은 Hyperband 설정,
  같은 val 행 사용 방식(튜닝 목적함수에만). 다른 것은 프로토콜(Espley: seed 23에서 한 번 튜닝 / 우리: seed마다 튜닝)뿐이다.
- 스레드: 모든 rev 6 학습에서 BLAS 1, TensorFlow intra-op 2 / inter-op 1(`r6_run.sh`). 튜닝 프로세스는 그들의 튜닝 스크립트처럼
  `TF_ENABLE_ONEDNN_OPTS=0`, 학습 프로세스는 그들의 학습 스크립트처럼 기본값. 한 할당 안에서 학습 단위 여러 개를 별도
  프로세스로 동시에 돈다(명세 §12 "병렬은 GridSearchCV n_jobs로만"과 다름 — 각 단위는 단일 스레드 BLAS라 과다 할당은
  없다). P-1·P-2의 NN 재현 검사는 그들의 기본 스레드로 했고, 운영 스레드(TF 2)에서의 차이는 `P2_nn_reimpl_tf2.csv`에 따로 적는다.

## 3. 대표 모델 규칙 (§3-3)

- arm·타깃마다 6개 모델 가운데 **5 seed 평균 검증 MAE가 가장 낮은 모델**. 동점이면 모델 이름 순.
- 검증 MAE: 우리 파이프라인 = §2-2; Espley 프로토콜 = 그들의 GridSearchCV `best_score_`(`hps.pkl`의 `train_mae`, seed 23
  튜닝에서 한 번이라 seed마다 같다), NN은 튜닝한 hps로 seed 23 train ∩ C에서 KFold(5)(셔플 없음 = 그들의 cv=5) 5-fold CV.
  알려진 특이점: 그들의 KRR `best_score_`는 poly 커널(튜닝)의 값이고 실제 학습은 rbf다. 규칙은 그대로 쓰고,
  그들의 `y_train_CV`(실제 모델의 seed별 5-fold CV)로 고른 결과를 민감도로 함께 적는다.
- 시험 점수로는 아무것도 고르지 않는다. 모든 모델 결과를 보고한다.

## 4. 주 비교 (확증적, 4개)

- 타깃 d1, d2, Int, ΔE‡마다 **OUR-O 대표 대 ESP-C 대표**, 행 C.
- seed별 test MAE 차이 d_s = MAE_s(ESP-C) − MAE_s(OUR-O), s = 22, 23, 14, 1, 2.
- Nadeau–Bengio 보정 t(`rev5_common.nadeau_bengio`, J = 5, n_test / n_train = seed 평균): 양측 p, 95 % CI.
- 4개 타깃에 Holm 보정(α = 0.05). seed 5개 중 OUR-O가 낮은 수.

## 5. 부 비교 (설명적, 보정 없음)

1. ESP-O 대표 대 OUR-O 대표 (튜닝 같음)
2. ESP-C 대표 대 OUR-C 대표 (Espley 프로토콜 안)
3. 모델별(6개) 같은 모델끼리 네 arm의 모든 쌍(6쌍)
4. Espley test best-of(ESP-C 6개 중 test MAE 최저, Espley에 유리) 대 OUR-O 대표
5. Espley 대표를 그들의 `y_train_CV`(실제 학습 모델의 seed별 5-fold CV; NN은 §3의 CV)로 고른 경우 대 OUR-O 대표
6. C_all 민감도: ESP-C(그들의 코드를 C_all로 다시 튜닝·학습)와 OUR-O(C_all에서 다시 학습), 대표는 C_all에서 같은 규칙으로
7. `constrained_ref` 행 민감도: C에서 그 행을 뺀 C_nocref로 ESP-C(그들의 코드 재튜닝·재학습)와 OUR-O를 다시 학습하고
   C_nocref test로 채점, 대표는 C_nocref에서 같은 규칙으로. 같은 모델로 test에서만 뺀 재채점도 함께 적는다.
8. Σ채널 arm: OUR-O 채널별 대표의 합 Int̂ = Σ6̂ + ĉ_ghost, ΔE‡̂ = d1̂ + d2̂ + Int̂ 대 OUR-O 직접 대표, 대 ESP-C 대표
- 편향 점검(§2-2): ESP-C 모델(모델별, 대표 표시)의 test ∩ D 행과 test ∩ C 행의 평균 |오차| 차이. 반응마다 test였던
  seed들의 |오차|를 평균한 뒤 반응 단위 bootstrap 10,000회(rng 20261001, 두 집단 따로 재표본) 95 % CI.
- 행 동일성(§2-3): 집계(`r6_analyze.py`)는 (타깃, 행 집합, seed)마다 모든 arm·모델의 test 행 sha256이 하나이고 Phase S
  값과 같은지 확인한다(아니면 STOP).

## 6. 지표와 반올림

- 주 지표 test MAE. 보조: NMAE(`rev5_common.nmae`), r², |오차| ≤ 1 kcal/mol 비율. 모두 저장한 반응별 예측에서 계산한다.
- 반올림(보고서): MAE 소수 둘째(disp, CDS는 셋째), NMAE·r² 셋째, p 유효숫자 둘.

## 7. Phase A

- 주 결과: 행 C, §1 분할, OUR-O 6 모델 × 8채널 + c_ghost. ESP-O 6 모델 × 8채널을 참고열로 둔다(설명적).
- 부록: 부록 A 행, Protocol A(80/10/10, test = hold-out 둘째 절반, seed 22/23/14/1/2), OUR-O 6 모델 × 8채널.

## 8. 바꾸지 않는 것과 알려진 특이점

- 결과를 본 뒤 arm, 모델, 행, 지표, 대표 규칙을 바꾸지 않는다. 바꾸면 이탈로 기록한다.
- Espley 프로토콜의 seed 23 튜닝은 seed 23 train 행이 다른 seed의 test와 겹친다(하이퍼파라미터 수준 누출, Espley 쪽에 유리).
  그대로 둔다.
- 그 밖의 특이점(그들의 코드 그대로): KRR poly 튜닝 / rbf 학습, 튜닝 47열 / 학습 46열(ESP-C만), RF 튜닝·학습.
- 이탈·사양과의 차이(지금까지)
  1. **명세 §3-1 서술과 그들의 코드:** `hyp_tuning.py`는 KRR·SVR뿐 아니라 Ridge(alpha, tol)와 RF도 GridSearchCV로 튜닝하고, NN은 keras_tuner Hyperband로 튜닝한다. 그래서 ESP-C·OUR-C의 Ridge와 NN도 그들의 튜닝을 거친다. 공개 `hps.pkl`·`ml_results.pkl`에는 RF가 없다.
  2. **keras_tuner 1.4.7 (사용자 결정, P-1 STOP 뒤):** 그들의 `requirements.txt`는 1.4.5인데, 1.4.5는 keras 3.3.3에서 Hyperband 체크포인트를 저장하지 못한다("filename must end in .weights.h5"). 나머지 고정 버전은 그대로다(`--no-deps`).
  3. **NN 비트 단위 재현 불가 (사용자 결정으로 수용):** 그들의 저장 예측 대비 반응별 |차| 중앙값 0.18–1.34 kcal/mol, seed별 MAE 차이 ±0.22 kcal/mol 이내. 우리 환경 안에서는 결정적이다(P-2 같은 노드 50/50 일치).
  4. `q_barrier_dft`(ΔG‡)는 데이터셋에서 뺐다. 우리 라벨에 없다.
  5. RF는 그들의 코드가 돌리지만 6개 모델에 들지 않아 수집하지 않는다.
  6. **`constrained_ref` 민감도:** C_nocref로 다시 학습한다. 명세 R-2의 "민감도 분석에서 빼 본다"를 재학습으로 읽었고, 재채점만 한 값도 함께 적는다.
  7. **스레드·병렬:** §2-3과 같다(명세 §12 "병렬은 n_jobs로만"과 다름).
  8. **R-1:** partition 재탐색은 더는 gate가 아니다(명세 R-1). partition만 실패했던 4행은 라벨 A_idx로 바꾼 뒤 외부 결합 감사에서 걸렸고, 사다리 D로도 되살아나지 않았다.
  9. **P-2 튜닝 비교는 이 commit 시점에 진행 중이다.** 우리 파이프라인 단위(OUR-O, ESP-O; 그들의 코드를 쓰지 않음)는 이 commit 뒤 바로 학습한다. Espley 프로토콜 arm(ESP-C, OUR-C, ESP-C_all, ESP-C_nocref)은 P-2 튜닝 비교가 통과한 뒤에만 돈다. 실패하면 STOP 하고 묻는다.
  10. **HPC 규칙 위반 (자진 보고):** 로그인 노드에서 빈 `python3 -c 1`을 한 번 실행했다. 로그인 노드 명령 하나에 `sleep 2`가 들어갔다. 둘 다 계산은 없다.
