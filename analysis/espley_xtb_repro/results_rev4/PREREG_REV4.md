# PREREG_REV4 — rev 4 사전 등록 (2026-09-30)

이 문서는 Phase 3(본실험 재학습)과 Phase 4(Espley 비교, 기하 절제)의 **학습 결과를 보기 전에** 작성해 commit 한다
(REV4_XTB_GEOMETRY.md §1, §3-1). commit 이후 아래 규칙은 바꾸지 않는다. 바꿔야 하면 바꾸지 않고, 결과 보고에
"사전 등록과 다름"으로 따로 적는다. 결과를 본 뒤 arm, 모델, 행을 고르지 않는다.

이 시점까지 본 것은 학습과 무관한 입력 점검뿐이다: Phase 1(G1 기하 생성 성공률·RMSD), Phase 2(feature 재계산,
`--geom dft` 재현 검증, G0–G1 feature 분포 차이), Phase 4-4(Espley AM1 TS 출발 구조). 어떤 모델도 학습하지 않았다.

## 1. 입력

| 항목 | 값 |
|---|---|
| 라벨 | repo 루트 `labels_all.json` (sha256 `62c0e7845045d0810627e88f1f52a3f353059bd78ac12ed89f8cd0f053a08753`), status ok 5,260 |
| G0 feature | `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features_g0.parquet` (sha256 `f3e298052245ba7a20d0beaf2cc87194cf9d1a1c3493b555669edf346310f0f4`) — Coley DFT TS/참조 기하 = rev 3 기하. `--geom dft` 재계산이 rev 3 parquet과 비트 단위로 같음 (`phase2_verify_geom_dft.json`, max\|diff\| 0) |
| G1 feature | `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features_g1.parquet` (sha256 `64ee5da7dcc7e3b0b2c055fe5030135f255e8d30e487cc6d1ea456d64af4b1d8`) — ORCA 6.1.1 `! XTB2 ALPB(water) OptTS Freq` (xtb 6.7.1 edcfbbe) TS + xtb 6.7.1 `--opt tight --alpb water` 참조, 출발점 = 라벨의 DFT 구조 (`g1_geom.py`) |
| feature 엔진 | xtb 6.7.1 (edcfbbe), GFN2, ALPB(water) — 두 기하 공통 |

## 2. 행 (Phase 3)

`results_rev4/rows_rev4.csv` (rxn_id 오름차순, sha256 `5c72dc7d686e1e0a16c470b383d85e33978696447766b0f43c7b0bb00ef8d4ab`), **n = 4,839**.
규칙: G1 `xtb_status` ok ∩ G0 `xtb_status` ok ∩ 두 parquet 모두에서 ESPLEY73 feature와 9 타깃에 NaN 없음 ∩ rev 3
hygiene (`dft_d1_kcal` ≥ 0, `dft_d2_kcal` ≥ 0, `dft_d2_kcal` ≤ 50). 단계별 건수와 제외 id: `rows_rev4.json`
(G1 ok 4,860, G0 ok 5,260, 둘 다 ok 4,860, NaN 제외 0, hygiene 제외 21).
G0와 G1은 **이 행을 이 순서로** 똑같이 쓴다 → seed마다 train/test 분할이 두 기하에서 동일하다.

## 3. Protocol A (rev 3과 동일, `train_ml_single.py`)

- 분할 `split_80_10_10`: `train_test_split(test_size=0.2, random_state=seed)` 후 나머지 20 %를
  `train_test_split(test_size=0.5, random_state=seed)`로 나눈 **두 번째** 출력이 test (rev 3 그대로). seed 22 / 23 / 14 / 1 / 2.
- seed마다 그 seed의 train에서 `GridSearchCV(cv=KFold(5, shuffle=True, random_state=seed), scoring=neg MAE)` (nested).
- 파이프라인: `TransformedTargetRegressor(regressor=StandardScaler(X) → est, transformer=StandardScaler(y))`.
- grid (rev 3 `GRIDS` 그대로): Ridge alpha {1e-4 … 1e3}; KRR(RBF) alpha {1e-5 … 1} × gamma {3e-4 … 1};
  SVR(RBF) C {1, 10, 100, 1000} × gamma {scale, 1e-3, 1e-2, 1e-1} × epsilon {0.05, 0.1, 0.5};
  XGB n_estimators {200, 500} × max_depth {3, 5, 7} × learning_rate {0.03, 0.1}.
- arm: ESPLEY46, ESPLEY54, ESPLEY73. 모델: Ridge, KRR(RBF), SVR(RBF), XGB. Protocol B는 돌리지 않는다.
- 타깃 9: barrier, d1, d2, elst, pauli, oi, disp, cpcm, cds (`TARGETS_REV4`). e_bond, eint_spe, c_ghost는 학습하지 않는다.
- 지표: seed별 test MAE의 평균 ± seed간 sd, NMAE (= MAE / 평균절대편차), r² (seed 평균).

## 4. 보고 규칙 (Phase 3)

- **헤드라인: G1 · ESPLEY73 · KRR(RBF)** (rev 3과 같은 사전 고정 규칙). 다른 arm·모델은 부록.
- G0는 같은 arm·모델로 같은 행에서 학습해 **"G0 = DFT oracle geometry (상한)"**으로만 싣는다. 헤드라인으로 쓰지 않는다.
- **G0의 disp는 모든 rev 4 표에서 비운다**: G0에서 `b_disp`는 DFT disp 라벨과 해석적으로 같아(MAE 0.000) ML 성과가 아니다.
  G1의 disp는 xTB 기하에서 계산한 진짜 feature이므로 보고한다.
- 표: 타깃 × {G0 상한, G1} × 헤드라인 모델의 MAE, NMAE, r², G1 − G0.

## 5. 하류 분석 (Phase 3-3)

- `analyze_extra.py`(charge_breakdown, group_split)와 `evaluate_pairs.py`(MMP split_metrics, margin_calibration)를
  G1과 G0(상한) 각각의 ESPLEY73 · KRR(RBF) 예측으로 다시 돌린다. 행 = `rows_rev4.csv`.
- rev 3과 다른 점 (코드–문서 불일치 수정, AUDIT_G0 §1-7): group_split은 rev 3 코드가 모든 5,260 행과 y 표준화 없는
  KRR을 썼으나, rev 4는 `rows_rev4.csv`와 튜닝 때와 같은 `_make_pipe`(y 표준화)를 쓴다.
- `evaluate_pairs.py`의 KRR은 rev 3처럼 alpha = gamma = 1e-3 고정(튜닝 안 함).
- MMP dominance에서 G0의 disp는 참값(= b_disp)을 쓰므로 G0 dom_agree는 상한이다. G1의 disp는 out-of-fold 예측값.

## 6. Espley 비교 (Phase 4-1, 4-2; `compare_espley.py`)

- 반응: Espley ds3 ML 세트 3,510 (`reaction_number` = rxn_id), 그들의 행 순서. 타깃: 그들의 DFT 값. hygiene 필터 없음.
- 분할: Espley 방식 (test = 20 % hold-out의 **첫 번째** 절반, seed 22 / 23 / 14 / 1 / 2) — 그들이 저장한 test 타깃과
  일치 확인(assert).
- **행: 라벨 있음 ∩ G0 사용 가능 ∩ G1 사용 가능.** 우리 G0·G1 모델은 이 행에서 학습하고 test도 이 행으로 채점한다.
  Espley 저장 예측은 같은 test 행으로 다시 채점한다. Espley 쪽 재학습(4-2)은 그들의 행(역할 라벨 있는 행)으로 학습한다.
- **주 비교 (사전 고정, best-of 선택 없음): G1 · ESPLEY46 · KRR 대 Espley SVR.** interaction, ΔE‡, ΔG‡는 Espley 저장
  예측(SVR), d1/d2는 **역할 기준**(dipole / dipolarophile)으로: Espley 46 AM1 feature의 `distortion_energy_1/2_am1`를
  같은 swap 벡터로 재배열하고 역할 타깃으로 (a) 그들의 프로토콜(ESI Table S3 grid, seed 23 train에서 한 번 튜닝,
  X만 StandardScaler; SVR과 KRR)과 (b) 우리 파이프라인으로 재학습한다. 주 표의 Espley d1/d2는 (a)의 SVR.
- (a)의 재현 점검: 그들의 index d1/d2에서 재튜닝한 파라미터가 `hps.pkl`과 같은지 본다. 튜닝 열은 47(그들의
  hyp_tuning.py) → 46 순으로 시도해 먼저 재현되는 쪽을 쓴다. **둘 다 재현되지 않아도 멈추지 않는다**: 47열로 튜닝하고
  `espley_protocol_replicated = False`를 표와 그림에 표시한다.
- 부록: 양쪽 best-of-models (G0 = 상한 표기), 인덱스 기준 d1/d2 비교. "양쪽이 같은 핸디캡" 문장은 쓰지 않는다
  (Espley AM1 distortion feature는 타깃과 같은 인덱스를 따른다).

## 7. 기하 정보 절제 (Phase 4-3; `espley_fairness_ablation.py`)

- arm: A (Espley46, 우리 파이프라인), B (Espley46 + G0 거리 11), B_role (B의 역할 기준, KRR만), C (Espley46 역할 기준),
  C0 (Espley46, feature swap 없이 역할 타깃), O (우리 ESPLEY46, G0), **B_g1 (Espley46 + G1 거리 11)**, O_g1 (우리
  ESPLEY46, G1; 추가 arm). 모델 KRR(RBF), SVR(RBF) (B_role은 KRR만). 튜닝: seed 23 train에서 한 번
  (`GridSearchCV`, KFold 5 rs 23), 그 뒤 seed마다 refit. 분할: Espley 방식.
- 행: **주 결과 = 모든 arm이 쓸 수 있는 공통 행** (arm 간 비교를 위해; 원 스크립트와 다른 점),
  **재현 확인 = 원 규칙**(arm마다 G0 xtb ok ∩ arm feature ∩ 타깃, `ABL_ROWS=own`, `_ownrows` 파일).

## 8. 이 사전 등록에 없는 것 (사용자 결정 대기)

- Phase 4-5 (Espley AM1 기하 위 xTB feature): 4-4 결과 보고 후 승인 시.
- Phase 5 (G2, autodE로 SMILES부터 xTB 수준 탐색): D1 venv(autodE 1.4.5)를 써야 하므로 사용자 결정 후.
