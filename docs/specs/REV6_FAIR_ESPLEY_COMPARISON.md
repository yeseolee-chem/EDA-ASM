# REV6 — Espley 2024 모델과 우리 모델의 공정한 비교 + 우리 8채널 모델 결과

(사용자 제공 명세, 2026-10-01. 원문 그대로 보관.)

**대상:** `main` `600e5a1` (rev 5 결과까지 들어간 커밋), `analysis/espley_xtb_repro/`
**새 브랜치:** `espley-rev6`
**결과 폴더:** `analysis/espley_xtb_repro/results_rev6/` · **scratch:** `/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/`
**이 사양서 위치:** `docs/specs/REV6_FAIR_ESPLEY_COMPARISON.md` (사용자 결정 원문을 그대로 보관)
**보고:** Phase마다 보고한다. STOP 조건이 나오면 다음 Phase로 가지 않고 사용자에게 묻는다.

**순서:** Phase 0(준비) → R(실패 반응 되살리기) → P(Espley 코드 재현과 입력 정렬) → S(행·분할 고정과 사전 등록)
→ B(공정 비교 실험) → A(8채널 실험) → E(그림) → F(SUMMARY)

---

## 0. 목적과 사용자 결정 (2026-10-01, 원문 요지)

### 0-1. 목적
1. **주 목적: Espley 2024 모델과 우리 모델의 공정한 비교.**
2. **추가 목적: 우리 8채널 모델의 결과를 따로 보기.** 8채널 = d1, d2, elst, Pauli, OI, disp, CPCM, CDS.
3. **관심 없는 것:** "AM1을 xTB로 바꾸면 어떻게 되나"는 이 실험의 질문이 아니다. 보고서에서 결과를 그렇게 해석하거나 그런 틀로 쓰지 않는다.

### 0-2. 사용자 결정 목록

| # | 결정 | 반영 위치 |
|---|---|---|
| D1 | Espley 모델은 **논문 코드 그대로** 가져온다 | §3 arm ESP-C, Phase P |
| D2 | 우리 모델은 **feature 182개**(rev 5 EXT_ALL = ESPLEY73 + B1…B6)를 쓴다 | §3 arm OUR-O |
| D3 | 우리 모델 종류: **SVR, KRR, Ridge, XGB, NN** 각각 결과를 낸다 | §3-2 |
| D4 | Espley 쪽에도 **XGB를 추가**하고, 모델별 결과를 각각 보인다 | §3-1, Phase B |
| D5 | 두 모델은 **같은 반응을 공유**한다 (반응 id 기준) | §2 |
| D6 | 학습/시험 반응을 **양쪽에 똑같이** 나눈다 | §2-3 |
| D7 | 실험을 둘로 한다: **(B) 비교 실험** = Int, ΔE‡, d1, d2 네 값을 정답으로 맞추기 / **(A) 8채널 실험** = 8채널 각각 맞추기 | Phase B, Phase A |
| D8 | 비교 실험의 정답지는 **양쪽 모두 같은 값(우리 라벨)** 이다 | §1-2 |
| D9 | **계산 못 한(실패한) 반응을 최대한 되살린다** | Phase R |
| D10 | **조각 나누기 실패 4건**은 이미 라벨에 있는 조각 정보(A_idx)를 그대로 쓴다 | R-1 |
| D11 | **feature 계산 실패 1건(rxn 4994)은 다시 넣는다** | R-4 |
| D12 | **튜닝 조건 맞추기** — Espley feature를 우리 파이프라인에도 통과시킨다 | §3 arm ESP-O (+ 대칭 arm OUR-C) |
| D13 | **d1, d2 순서 맞추기** — Espley 변형 feature를 역할(dipole, dipolarophile) 순서로 재배열한다 | P-3 |
| D14 | **돌리기 전에 판정 규칙을 고정한다(사전 등록)** — 지표, 검정, 대표 모델 고르는 규칙 | Phase S |

### 0-3. 배경 — 이번 대화에서 확인한 사실 (rev 5 기준, `results_rev5/SUMMARY.md`)

- **feature 세트**
  - 73개(ESPLEY73)에 새 후보 109개를 의미별 블록 6개(B1 오비탈 겹침 21, B2 접촉 23, B3 정전기 7, B4 반응성 지수 39, B5 거리 스캔 16, B6 TS 성격 3)로 나눠 붙였다.
  - 전진 선별이 고른 세트는 EXT_SEL = 73 + B4 + B1 = 133개다.
  - EXT_ALL = 73 + B1…B6 = 182개다. 133개는 182개 안에 모두 들어 있다(`rev5_common.arm_columns`).
  - lockbox에서 182개의 점추정은 elst·Pauli·OI에서 133개보다 2.5–4.8 % 낮았다. 182 − 133 짝지은 CI는 계산하지 않았다.
  - **사용자 결정: 182개를 쓴다.** 182개는 rev 5에서 결과를 보기 전에 고정한 arm이라 선별 편향이 없다.
- **rev 5 행 수가 헷갈렸던 이유**
  - 4,838 / dev 4,112 / lockbox 726은 feature 실험용 행이다.
  - Espley 비교는 3,322행이었다. Espley ds3 3,510 → 우리 라벨 있음 3,509 → G1·hygiene 통과 3,327 → 확장 feature 성공 3,322.
  - rev 6은 **하나의 공통 반응 집합**만 쓴다(§2).
- **실패의 정체**
  - **G1 실패 400 / 5,260**: xTB 함수가 실패한 게 아니라, xTB 지형에서 **TS(안장점) 찾기**가 실패했다.
    - OptTS 미수렴 112
    - 첫 허수 gate(≤ −40 cm⁻¹) 160, 추가 허수 gate 3
    - 형성 결합 mode share < 0.5 46, 외부 결합 gate 23, 형성 결합 거리 gate 1
    - 참조 1 결합 그래프 변화 49, 참조 2 결합 그래프 변화 2
    - partition 계산 불가 4
    - 출처: `results_rev4/SUMMARY.md` Phase 1-2
  - **hygiene 21**: 라벨 쪽 문제다. 변형 에너지는 물리적으로 0 이상이어야 한다. d1 < 0이 7건, d2 < 0이 5건, d2 > 50 kcal/mol이 9건이다.
  - **확장 feature 실패 1(rxn 4994)**: 계산이 깨진 게 아니라, tblite − xtb 에너지 일치 검사에서 1.9e-6 Eh로 기준 1e-6 Eh를 조금 넘었다.
  - **partition 실패 4**: 라벨의 조각은 이미 정해져 있다. 코드가 xTB TS에서 결합 거리로 조각을 **다시 찾아 확인하는 검사**를 했고, 그 확인이 실패한 것이다(`g1_geom.py` L283–295). 원자 순서는 라벨과 같다(`g1_geom.py` L7).
- **Espley 쪽은 계산하지 않았다.** 그들의 공개 feature(AM1 46개), 저장 예측, 코드를 불러다 썼다. 재학습은 두 경우뿐이었다.
  - d1/d2를 역할 기준으로 다시 배정한 경우
  - 그들의 저장 모델이 우리 시험 반응을 이미 학습한 경우
- **구조 출발점은 양쪽이 같다.** Espley의 AM1 TS도 Coley DFT TS에서 출발했다(프로젝트 문서 `claude/review-espley-pipeline-verified.md`). 우리 G1 TS도 같은 DFT TS에서 출발한다. 따라서 이 점에서 두 쪽은 대등하다. 둘 다 SMILES에서 출발하는 배포 모델은 아니다.
- **Espley 라벨로 한 비교는 rev 5 D-3에 이미 있다.** 참고로만 인용하고 다시 돌리지 않는다.

---

## 1. 정의

### 1-1. 우리 라벨 (`labels_all.json`, 수정 금지)
- **8채널**: `dft_d1_kcal`, `dft_d2_kcal`, `dft_elst_dft`, `dft_pauli_dft`, `dft_oi_dft`, `dft_disp_dft`, `dft_cpcm_dft`, `dft_cds_dft`. 이름은 `rev5_common.TARGETS9`와 같다.
- **Int** = `dft_eint_spe_kcal`. `d1 + d2 + eint_spe = barrier`가 정확히 성립한다(`label_true/SMD_RELABEL.md`, 5265/5265, max 8.6e-10).
- **ΔE‡** = `dft_barrier_kcal`.
- 6채널의 합은 `e_bond`이고 `eint_spe = e_bond + c_ghost`(`bsse_shift_kcal`)다. 그래서 **Int는 6채널 합과 같지 않다.** 이 관계를 Phase S에서 데이터로 확인해 기록한다.
- **ΔG‡**: 우리 라벨에 없으므로 비교에서 뺀다.

### 1-2. 비교 실험(B)의 정답지
- 네 타깃: **d1, d2, Int(eint_spe), ΔE‡(barrier)**. 모든 arm이 같은 값으로 학습하고 채점한다.
- Espley 코드에는 **정답 열만** 우리 값으로 바꿔 넣는다. 다른 코드는 바꾸지 않는다. 바꾼 줄은 diff로 기록한다.
- Espley는 Int를 양수 = 안정화 부호로 저장했다. rev 6은 우리 부호(`eint_spe`) 하나만 쓴다. 두 쪽이 같은 부호이므로 MAE는 영향을 받지 않는다.
- d1 = dipole 변형, d2 = dipolarophile 변형이다(우리 라벨의 역할 기준).

---

## 2. 반응 집합과 분할 (D5, D6)

### 2-1. 행 정의 (모두 rxn_id 기준)
| 이름 | 정의 |
|---|---|
| R_esp | Espley ds3 ML 세트 3,510행 (`manual_tt_solvent.pkl`, `reaction_number` = rxn_id, ds3 행 순서) |
| R_lab | R_esp ∩ `labels_all.json` status ok |
| R_feat | Phase R 이후 우리 182 feature가 모두 있는 행 (G1 gate 통과 ∩ 확장 feature ok) |
| R_hyg | d1 ≥ 0, d2 ≥ 0, d2 ≤ 50 (rev 4와 같은 규칙. 라벨 쪽 규칙이라 두 쪽에 똑같이 적용) |
| **C (주 분석)** | R_lab ∩ R_feat ∩ R_hyg |
| C_all (민감도) | R_lab ∩ R_feat (hygiene 행 포함) |

- 행 흐름표 R_esp → R_lab → R_feat → C를 사유별 개수와 함께 만든다.
- Espley 쪽 feature(AM1 46)는 R_esp 전체에 있다. 따라서 공통 반응의 크기는 우리 쪽 실패로만 줄어든다. 이 점을 보고서에 적는다.

### 2-2. 남는 실패의 처리 (D9의 공개 부분)
- Phase R 이후에도 R_feat에 없는 R_lab 행은 개수와 사유를 공개한다.
- **편향 점검(보고)**: 빠진 행이 우리에게 어려운 반응만 골라낸 것인지 확인한다.
  - ESP-C 모델이 seed별로 시험한 행 가운데 빠진 행 쪽과 C 쪽의 Espley 오차 분포를 비교한다.
  - Espley 모델은 빠진 행도 예측할 수 있다. 이 비교에 한해 빠진 행을 Espley 쪽 시험 행에 넣어 따로 채점한다.
  - 평균 |오차| 차이와 그 bootstrap 95 % CI를 보고한다.
- 주 비교는 C에서만 한다.

### 2-3. 분할 — Espley 코드의 분할을 그대로, 두 쪽 같게
- seed: 22, 23, 14, 1, 2 (`compare_espley.SEEDS`, Espley `ml_analysis.py`).
- 분할 함수: `compare_espley.espley_split(n=|R_esp|, seed)`. R_esp 전체 3,510행에 ds3 행 순서로 적용한다. train = 80 %, test = 20 % hold-out의 **첫 절반**이다(그들의 `_perform_train_test_split`).
- seed마다 **train_s = (그 seed의 train) ∩ C**, **test_s = (그 seed의 test) ∩ C**.
- 모든 arm·모델이 seed마다 **같은 train_s 행으로 학습**하고 **같은 test_s 행으로 채점**한다. 행 집합과 순서가 같은지 assert하고, sha256을 기록한다.
- hold-out의 둘째 절반: 그들의 코드에서 쓰는지(NN 검증 등) 확인해 기록한다. 쓴다면 모든 arm에서 같은 방식(∩ C)으로 쓰고, 안 쓴다면 모든 arm에서 쓰지 않는다.
- rev 5 D-3(a)의 비대칭(Espley는 2,807–2,808행, 우리는 평균 2,653.8행으로 학습)은 rev 6에서 없앤다.

---

## 3. arm과 모델 (D1–D4, D12)

feature 2종 × 학습·튜닝 방식 2종 = arm 4개. 모든 arm은 §2-3의 같은 분할을 쓴다.

| arm | feature | 학습·튜닝 방식 | 역할 |
|---|---|---|---|
| **ESP-C** | Espley AM1 46개 (역할 재배열, P-3) | **Espley 코드 그대로** (+ XGB 추가) | 주 비교의 Espley 쪽 (D1, D4) |
| **OUR-O** | 우리 182개 (EXT_ALL) | 우리 파이프라인 | 주 비교의 우리 쪽 (D2, D3) |
| ESP-O | Espley AM1 46개 (역할 재배열) | 우리 파이프라인 | 튜닝 조건을 맞춘 비교 (D12) |
| OUR-C | 우리 182개 | Espley 프로토콜 | 대칭 확인 (D12 보완) |

- ESP-O와 OUR-O는 튜닝이 같으므로 차이가 feature에서 온다.
- ESP-C와 OUR-C도 튜닝이 같으므로 차이가 feature에서 온다.
- ESP-C와 OUR-O(주 비교)는 사용자가 정의한 "Espley 코드 그대로 대 우리 모델"이다.

### 3-1. Espley 코드 그대로 (ESP-C, OUR-C)
출처는 `the-grayson-group/distortion-interaction_ML`이다. `compare_espley.py` 머리 주석이 줄 번호까지 정리해 두었다. 확인된 그들의 방식은 다음과 같고, **고치지 않는다.**
- **튜닝**
  - 스크립트: `hyperparameter_tuning/hyp_tuning.py` (sha256 `d1640730…`).
  - 튜닝은 seed 23의 train 분할에서 **한 번만** 한다(`random_state=23` 하드코딩).
  - CV: `GridSearchCV(cv=5)`, shuffle 없음, `neg_mean_absolute_error`.
  - 스케일: X만 StandardScaler, y는 원값.
  - 열: 튜닝은 47열(`sum_distortion_energies_am1` 포함), 학습은 46열.
  - grid: ESI Table S3(`compare_espley.THEIR_TUNE`).
  - KRR은 poly 커널로 튜닝하고 rbf로 실행한다.
- **학습·예측**
  - 스크립트: `machine_learning/ml_analysis.py`.
  - seed마다 train 전체로 학습하고, X는 StandardScaler, y는 원값이다.
  - 모델: Ridge, KRR, SVR, 2-layer NN, 4-layer NN(`compare_espley.ESP_MODELS`).
- **튜닝 대상**: 그들의 `hyp_tuning.py`는 KRR과 SVR만 튜닝한다(L303–308). Ridge와 NN은 그들의 코드가 하는 그대로 둔다(기본값이든 고정값이든). 실제 처리는 P-1에서 확인해 기록한다.
- **rev 6에서 바꾸는 것은 두 가지뿐이다.** 둘 다 diff로 기록한다.
  1. 정답 열 = 우리 라벨(§1-2)과, 행 = §2-3 분할
  2. **XGB 추가**(D4)
     - grid: `train_ml_single.GRIDS["XGB"]`의 값(n_estimators 200/500, max_depth 3/5/7, learning_rate 0.03/0.1)
     - 튜닝: 그들의 다른 모델과 똑같이 seed 23 train에서 그들의 CV로 한 번
- **hps.pkl은 쓰지 않는다.** 그들의 hps.pkl은 그들의 타깃으로 튜닝한 값이다. 우리 라벨로 바꿨으므로, **그들의 튜닝 코드(`hyp_tuning.py`)를 우리 라벨로 다시 돌린다.**
- **NN**
  - 그들의 NN 정의와 학습 코드(층 수, 노드, 활성화, optimizer, epoch, 조기 종료, 검증 행)를 그들의 저장소에서 찾아 그대로 쓴다.
  - **NN 학습 코드가 저장소에 없거나 그대로 돌릴 수 없으면 STOP한다.** 논문 서술로 다시 구현하지 말고 사용자에게 묻는다.
- **알려진 그들 방식의 특이점** (그대로 두고 보고서에 적는다)
  - seed 23 train에서 한 번 튜닝하므로, 그 train 행이 다른 seed의 test와 겹친다. 하이퍼파라미터 수준의 정보 누출이고, Espley 쪽에 유리하다.
  - KRR poly 튜닝 / rbf 실행
  - 튜닝 47열 / 학습 46열
- **OUR-C 적용 방식**: Espley 프로토콜을 `espley_protocol()` 함수로 감싸서 쓴다. 먼저 Espley 46 feature에서 그들의 코드 출력과 1e-8 안에서 같은지 확인한다(P-2). 그다음 X만 우리 182개로 바꿔 돌린다. 47/46 열 특이점은 182 feature에는 해당하지 않으므로 182열로 튜닝하고 182열로 학습한다(기록).

### 3-2. 우리 파이프라인 (OUR-O, ESP-O)
- rev 4/5와 같다(`train_ml_single.GRIDS`, `_make_pipe`).
  - seed마다 train_s 안에서 nested `GridSearchCV`(`KFold(5, shuffle=True, random_state=seed)`)를 돌린다.
  - X와 y를 표준화한다(TransformedTargetRegressor).
- **모델: Ridge, KRR(rbf), SVR(rbf), XGB, NN-2, NN-4.** 사용자가 정한 5종(SVR, KRR, Ridge, XGB, NN)에서 NN을 Espley와 같은 2층·4층 두 변형으로 나눈 것이다. 그래서 "모델 6개"다.
- **NN**: 모든 arm에서 그들의 NN 정의(2층, 4층)를 그대로 쓴다. 바뀌는 것은 입력 X(46 또는 182)와 학습 행뿐이다.
  - 그들의 NN에 튜닝 단계가 없으면 모든 arm에서 튜닝 없이 같은 설정을 쓴다.
  - 튜닝 단계가 있으면 각 방식(Espley 프로토콜 / 우리 파이프라인)에 맞춰 같은 grid로 튜닝한다.
  - 어느 쪽이든 NN 처리는 네 arm에서 같아야 한다.

### 3-3. 대표 모델 고르는 규칙 (시험 점수로 고르지 않는다, D14)
- arm·타깃마다 **대표 모델** = 6개 모델 가운데 **train_s 안의 검증 MAE**가 가장 낮은 모델이다.
  - 검증 MAE는 우리 파이프라인에서는 inner CV, Espley 프로토콜에서는 그들의 GridSearchCV `best_score_`다.
  - 5 seed의 평균 검증 MAE로 하나를 정한다.
  - 튜닝 점수가 없는 모델(예: 튜닝 없는 NN)은 같은 방식의 5-fold CV를 train_s에서 한 번 더 돌려 검증 MAE를 만든다.
- **모든 모델의 결과를 다 보고한다.** 대표 모델은 검정할 때만 쓴다.
- **보수적 부록**: Espley 쪽 "시험 점수 최고 모델"(test에서 고른 best-of, Espley에 유리)과도 비교해 보고한다.

---

## 4. Phase 0 — 준비

1. `main` `600e5a1`에서 브랜치 `espley-rev6`를 만든다.
2. 이 사양서를 `docs/specs/REV6_FAIR_ESPLEY_COMPARISON.md`로 commit한다.
3. `results_rev6/`과 scratch 폴더를 만든다.
4. 상수 원본 `rev6_common.py`를 만든다. `rev5_common`을 import하고 상수를 복사하지 않는다. 담을 내용:
   - 경로, seed, arm, 모델 목록, 타깃, 행 규칙, 검정 설정
5. **보존 (수정 금지)**
   - `labels_all.json`, G1 기하(`espley_xtb_g1/`), rev 4·5 feature parquet
   - `PREREG_REV4.md`, `PREREG_REV5*.md`, `results_rev4/`, `results_rev5/`
   - 되살린 결과는 **새 파일**로 쓴다(Phase R).

---

## 5. Phase R — 실패 반응 되살리기 (D9, D10, D11)

**원칙**
- **gate는 느슨하게 하지 않는다.** 되살린 TS도 rev 4 G1 gate(`g1_geom.py`의 `GATES`)를 모두 통과해야 한다.
- gate 값을 바꿔야 한다고 판단되면 STOP하고 묻는다. 예외는 R-4 하나뿐이고, 사용자가 이미 결정했다.
- 되살린 행에는 `recovered_by` 열(방법 이름)을 단다.
- 원래 G1 parquet은 수정하지 않는다. 되살린 기하는 `espley_xtb_g1_rev6/`, 되살린 feature는 새 parquet에 쓴다.
- 대상: R_lab에 있는 실패 행이다. Espley 비교에 필요한 행을 먼저 처리하고, 그다음 나머지 라벨 행(8채널 실험 부록용)을 처리한다.

### R-1. partition 실패 4건 → 라벨의 조각 정보 사용 (D10)
- 원자 순서가 라벨과 같으므로 `input_meta.csv`의 `A_idx`(라벨 조각)를 G1 TS에 그대로 적용한다. `fragmenter.partition`으로 다시 찾지 않는다.
- 그다음 나머지 gate(허수, mode share, 형성 결합, 외부 결합, 참조 그래프)를 그대로 적용한다. 같은 반응이 다른 gate에도 걸리면 R-2의 해당 사다리로 넘긴다.
- 이 방식은 rev 6 전체에 적용한다. 앞으로 partition은 라벨 A_idx를 쓰고, 재탐색은 진단으로만 기록한다.

### R-2. TS 찾기 실패 → 재시도 사다리
각 단계는 직전 단계가 실패한 행에만 돈다. 시도마다 입력, 설정, 결과, gate 값을 `R2_attempts.csv`에 남긴다. 단계가 끝날 때마다 사유별로 되살린 수와 남은 수를 센다.

| 실패 사유 (rev 4) | 건수 | 재시도 (순서대로) |
|---|---:|---|
| OptTS 미수렴 | 112 | (a) 마지막 구조에서 재시작, 최대 cycle 500 / (b) 정확한 Hessian을 매 N걸음 다시 계산 / (c) 형성 결합 두 개를 DFT 거리에 고정해 사전 최적화한 뒤 고정을 풀고 OptTS |
| 첫 허수 gate (≤ −40 cm⁻¹) | 160 | (a) 정확한 Hessian + 더 엄격한 수렴 / (b) 가장 낮은 모드 방향으로 ±변위 후 재최적화 / (c) 위 (c)와 같은 고정 사전 최적화 |
| 추가 허수 gate | 3 | 추가 허수 모드 방향으로 변위 후 재최적화 |
| mode share < 0.5, 외부 결합, 형성 결합 거리 | 46 + 23 + 1 | 엉뚱한 TS에 도착한 경우: 형성 결합 고정 사전 최적화 → 해제 후 OptTS |
| 참조 1·2 결합 그래프 변화 | 49 + 2 | 반응물(rel) 최적화에서 결합이 바뀐 경우: (a) 다른 출발 구조(라벨의 `_alt` 반응물 등 라벨과 같은 선택 규칙 안에서) / (b) 그래도 바뀌면 결합 그래프를 유지하는 제약 최적화. (b)로 살린 행은 `recovered_by = constrained_ref`로 표시하고 민감도 분석에서 빼 본다 |

- 재시도 설정(cycle 수, 변위 크기, 고정 방식)은 **Phase R 시작 전에** `R_plan.md`에 적고 commit한다.
- **xTB 지형에 그 TS가 실제로 없을 수 있다.** 사다리를 다 돌아도 안 되는 행은 실패로 남기고 사유와 함께 공개한다(§2-2).

### R-3. 확장 feature 재계산
- 되살린 G1 행에 rev 5 `ext_features.py`(코드 `r5-ext-3`, B1–B6 109열)를 그대로 돌린다.
- rev 4 G1 feature(ESPLEY73 등)도 같은 코드로 계산한다.

### R-4. rxn 4994 다시 넣기 (D11)
- B1 엔진 일치 gate 1.9e-6 Eh가 기준 1e-6 Eh를 넘은 1건이다.
- **사용자 결정으로 포함한다.** 이 행에만 허용치 1e-5 Eh를 적용하고, `recovered_by = gate_tol_user_decision`으로 표시한다.
- 값 자체는 rev 5 계산값을 그대로 쓴다.

### R-5. 보고
- 사유별 표(원래 실패 / 되살림 / 남음, 단계별)를 만든다.
- 되살린 TS와 DFT TS의 heavy RMSD와 형성 결합 차이 분포를, 원래 성공 행 분포와 함께 보인다.
- 최종 R_feat, C, C_all 크기와 행 흐름표를 만든다.
- **STOP 조건**
  - 되살린 행에서 gate 기록이 하나라도 빠졌을 때
  - R-1의 A_idx 적용 결과가 DFT TS 재탐색과 다를 때(진단으로 확인)
  - 계획(`R_plan.md`)에 없는 설정을 썼을 때

---

## 6. Phase P — Espley 코드 재현과 입력 정렬

### P-1. 그들의 코드가 그대로 재현되는지 (STOP gate)
- **그들의 데이터와 그들의 타깃으로** 그들의 코드를 고치지 않고 돌린다.
- 저장 예측(`espley_stored_preds.parquet`, `ml_results.pkl`)과 `hps.pkl` 값을 재현하는지 확인한다.
- 기준: 예측 최대 차이 1e-6 kcal/mol, hps 동일.
- rev 5 D-4는 SVR 학습 곡선에서 1.8e-12를 확인했다. rev 6은 Ridge, KRR, SVR, NN-2, NN-4 다섯 모델 전부를 확인한다.
- **하나라도 재현되지 않으면 STOP.** NN이 난수 때문에 비트 단위로 재현되지 않으면, 그들의 seed 설정을 확인하고 차이 분포를 보고한 뒤 묻는다.

### P-2. 프로토콜 함수 검증
- `espley_protocol()` 래퍼(OUR-C용)가 Espley 46 feature에서 P-1의 그들 코드 출력과 1e-8 안에서 같은지 확인한다.
- 다르면 STOP.

### P-3. d1, d2 순서 맞추기 (D13)
- 우리 d1/d2는 역할 순서(dipole, dipolarophile)이고, Espley 변형 feature는 그들의 조각 순서(`distortion_energy_1/2_am1`)다.
- rev 4 swap 벡터(`$R4_SCRATCH/compare/espley/prep.pkl`, `compare_espley.espley_X(swap=...)`)로 그들의 두 AM1 변형 feature를 역할 순서로 재배열한다. 이것을 ESP-C·ESP-O 모든 타깃에 쓴다.
- **검사 (STOP gate)**: swap 적용 후 Espley의 역할 배정 DFT 변형값과 우리 d1/d2를 비교한다.
  - 프로젝트 문서 `claude/review-espley-pipeline-verified.md`는 우리 역할 라벨과 MAE 0.05–0.06 kcal/mol이었다고 적었다.
  - 기준: 행별 |차| 중앙값 < 0.5 kcal/mol, d1·d2 각각.
  - 넘으면 STOP.
- swap 비율(rev 4에서 약 31 %)을 다시 계산해 기록한다.

### P-4. 정답 열 교체 diff
- 그들의 코드에서 바꾼 줄(정답 열, 행 선택, XGB 추가, hps 재튜닝 호출)을 `P4_their_code_diff.patch`로 남긴다.
- 그 밖의 변경은 금지한다.

---

## 7. Phase S — 행·분할 고정과 사전 등록 (D14)

1. C, C_all 행 목록과 seed별 train_s/test_s를 만들어 `S_rows.csv`, `S_splits.csv`에 쓰고 sha256을 기록한다.
2. 라벨 관계를 C에서 확인해 기록한다(`S_label_checks.json`).
   - `d1 + d2 + eint_spe − barrier`의 최대 |차|
   - `Σ6채널 − e_bond`
   - `eint_spe − e_bond − c_ghost`
3. **`PREREG_REV6.md` + `prereg_rev6.json`을 쓰고 commit한다.** 어떤 학습도 이 commit 뒤에만 돈다. 학습 스크립트는 commit 여부와 sha256을 먼저 확인한다(rev 5 `final_eval.gate_prereg` 방식). 담을 내용:
   - 행 집합(C, C_all)과 분할 sha256, arm 4개 × 모델 6개 × 타깃 정의
   - **주 비교 (확증적, 4개)**
     - 타깃 d1, d2, Int, ΔE‡마다 OUR-O 대표 모델 대 ESP-C 대표 모델(§3-3 규칙)을 비교한다.
     - 지표: seed별 test MAE.
     - 검정: seed별 차이 d = MAE(ESP-C) − MAE(OUR-O)에 **Nadeau–Bengio 보정 t-검정**(`rev5_common.nadeau_bengio`, J = 5, n_test/n_train = seed 평균). 양측 p와 95 % CI를 보고한다.
     - **4개 타깃에 Holm 보정**을 하고, seed 5개 중 우리 쪽이 낮은 수를 센다.
   - **부 비교 (설명적, 보정 없음)**
     - ESP-O 대 OUR-O (튜닝 같음 → feature 차이)
     - ESP-C 대 OUR-C (Espley 프로토콜 안에서 feature 차이)
     - 모델별(6개) 모든 쌍
     - Espley test best-of 대 OUR-O 대표(보수적)
     - C_all 민감도
     - `constrained_ref`로 되살린 행을 뺀 민감도
     - Σ채널 arm(§8-3)
   - 보조 지표: NMAE(`rev5_common.nmae`), r², 1 kcal/mol 안 비율
   - 반올림 규칙: MAE 소수 둘째(disp, CDS 셋째), NMAE·r² 셋째, p 유효숫자 둘
   - **결과를 본 뒤 arm, 모델, 행, 지표를 바꾸지 않는다.** 바꾸면 이탈로 기록한다.

---

## 8. Phase B — 공정 비교 실험 (D7 비교용)

### 8-1. 단위
- arm 4 × 모델 6 × 타깃 4(d1, d2, Int, ΔE‡) × seed 5 = 480 단위.
- 단위마다 캐시 하나를 둔다(parquet 예측 + JSON 완료 표시, 원자적 쓰기, idempotent).
- 입력 지문(데이터 sha256, 열, grid, seed, 패키지 버전)이 다른 캐시를 만나면 exit 3으로 멈춘다.
- **예측 저장**: test_s의 반응별 예측을 모두 저장한다(`B_predictions.parquet`). rev 5에서 lockbox 예측을 저장하지 않아 짝지은 CI를 다시 계산할 수 없었다. 이번에는 모두 저장한다.

### 8-2. 산출
- `B_per_seed.csv`: arm × 모델 × 타깃 × seed의 MAE, NMAE, r², n_train, n_test, 튜닝값, grid 경계 일치
- `B_summary.csv`: seed 평균 ± SE
- `B_primary_tests.csv`: §7의 주 비교 4개 (d, CI, t, p, Holm p, seed 승수)
- `B_secondary_tests.csv`: 부 비교 전체
- `B_representative.json`: arm·타깃별 대표 모델과 그 검증 MAE (5 seed 평균, 모델별)
- `B_bias_check.csv`: §2-2 편향 점검
- `B.json`: 설정, 행, 분할, 입력 sha256, gate 기록, P-1·P-2·P-3 결과

### 8-3. Σ채널 arm (부 비교, 보고만)
"8채널로 나눠 예측한 뒤 더하면 Int, ΔE‡를 직접 예측하는 것보다 나은가"를 본다.
- Phase A의 OUR-O 채널 예측(같은 분할, 같은 test 행)을 쓴다.
- c_ghost(`dft_c_ghost_kcal`)도 같은 방식으로 예측한다.
- 계산:
  - Int̂ = Σ6̂ + ĉ_ghost
  - ΔE‡̂ = d1̂ + d2̂ + Σ6̂ + ĉ_ghost
- 모델은 채널마다 대표 모델을 쓴다.
- 직접 예측(OUR-O, ESP-C)과 같은 표에 놓고 보고한다. 검정은 설명적 범주다.

---

## 9. Phase A — 우리 8채널 실험 (D7 8채널용, 추가 목적)

### 9-1. 주 결과 — 비교 실험과 같은 행·분할
- 행 C, §2-3 분할, OUR-O(182 feature, 우리 파이프라인), 모델 6개.
- 타깃 8개: d1, d2, elst, Pauli, OI, disp, CPCM, CDS. Σ채널 arm을 위해 c_ghost도 함께 돌린다.
- 산출: `A_per_seed.csv`, `A_summary.csv`(채널 × 모델 MAE ± SE, NMAE, r²), `A_representative.json`(채널별 대표 모델, §3-3 규칙), `A_predictions.parquet`.
- **Espley에는 이 채널을 맞히는 모델이 없다.** 그래서 이 결과는 비교가 아니라 우리 모델의 결과로 보고한다.
- 참고열로 ESP-O(Espley 46 feature × 우리 파이프라인)를 같은 8채널에 돌린 결과를 함께 놓는다. 이것은 "Espley feature로 채널을 맞히면"이고, 부 비교(설명적)다.

### 9-2. 부록 — 우리 라벨 전체 행
- 행: 우리 라벨 ok ∩ R_feat ∩ R_hyg. Espley 반응이 아닌 행도 포함한다.
- 분할: rev 4/5 Protocol A(80/10/10, seed 22/23/14/1/2, `split_80_10_10`).
- OUR-O, 모델 6개, 8채널.
- 같은 분할 안의 모델 간 비교만 해석한다. rev 5 헤드라인과 직접 비교하지 않는다(행이 바뀌면 분할이 통째로 바뀜).

---

## 10. Phase E — 그림

양단 183 mm, 7 pt, PDF + PNG 300 dpi + 그린 값 CSV. Arial이 없으면 DejaVu Sans를 쓰고 기록한다.

| 그림 | 내용 |
|---|---|
| fig1_primary | 타깃 4개 × arm 4개, 대표 모델의 seed 평균 MAE 막대 + seed 점 5개. 주 비교(ESP-C 대 OUR-O)에 Δ, Holm p 주석 |
| fig2_models | 타깃 4개 × 모델 6개 × arm 4개 MAE 히트맵 또는 묶음 막대 (모든 모델 공개) |
| fig3_parity | ESP-C 대표 / OUR-O 대표 × 타깃 4개 패리티 |
| fig4_seed_pairs | 타깃별 seed 5개의 짝지은 선 (ESP-C → OUR-O) |
| fig5_channels | 8채널 × 모델 6개 MAE (Phase A 주 결과), ESP-O 참고 막대 |
| fig6_sumchannels | 직접 예측 대 Σ채널 예측 (Int, ΔE‡) |
| fig7_recovery | Phase R 사유별 되살림 막대 + 편향 점검 |

---

## 11. Phase F — `results_rev6/SUMMARY.md` (한국어)

- **맨 앞 세 문장**: 무엇을 비교했는지, 주 비교 결론(4개 타깃, Holm 보정 후), 8채널 결과 한 줄.
- **절 순서**
  1. 주 비교
  2. 튜닝 맞춘 비교(ESP-O, OUR-C)
  3. 모델별 전체
  4. Σ채널
  5. 8채널(Phase A)
  6. 되살리기와 편향 점검
  7. 이탈 사항
  8. 한계
  9. 산출물
  10. 재현
- **수치**: 모두 `results_rev6/` 파일에서 옮긴다. 폴더 밖 출처는 그 자리에 경로를 적는다.
- **반드시 적을 한계**
  - 남은 실패 행과 편향 점검 결과
  - Espley 프로토콜의 seed 23 튜닝 누출(그들에게 유리)
  - 구조 출발점이 양쪽 모두 DFT TS라서 배포 성능이 아니라는 점
  - NN 재현 정도
  - grid 경계 일치
  - `constrained_ref` 행
- **"AM1을 xTB로 바꿔서 좋아졌다"는 식의 해석 문장을 쓰지 않는다.** 이 실험은 그 질문을 분리하지 않는다. 비교 대상은 "Espley 모델(그들의 입력·코드)"과 "우리 모델(우리 입력·코드)"이다.
- **이탈과 HPC 규칙 위반**은 자진 보고한다.

---

## 12. 규칙 (rev 4/5와 같음)

- **HPC (저장소 `CLAUDE.md`)**
  - 모든 계산은 `sbatch`로 하고 `#SBATCH --time=48:00:00`을 쓴다.
  - 로그인 노드에서 python, xtb, 루프, `nohup`, 폴러를 쓰지 않는다.
  - 제출 전에 `squeue -u $USER -h | wc -l`을 확인한다. MaxJobs 10, MaxSubmit 20이고, array 원소는 하나씩 센다.
  - 20을 넘는 fan-out은 `--dependency=afterany` 사슬로 한다.
  - CPU 파티션은 `sinfo`로 빈 쪽을 고른다.
- **idempotent**: 모든 단위는 출력이 있으면 건너뛰고, 결과는 원자적으로 쓴다(`rev5_common.write_atomic`).
- **스레드**: BLAS 스레드를 1로 둔다(`OMP/OPENBLAS/MKL_NUM_THREADS=1`). 병렬은 GridSearchCV `n_jobs`로만 한다.
- **시험 행 보호**: test_s 행은 튜닝, 대표 모델 선택, Phase R 설정 결정에 쓰지 않는다. 학습 스크립트는 첫 실행 때 튜닝 캐시에 test id가 없는지 감사한다(rev 5 `final_eval.audit_select` 방식). 예외는 Espley 프로토콜의 seed 23 튜닝 특이점 하나뿐이고 보고한다.
- **수치**: 파일에서 직접 읽는다. 추정이나 기억으로 쓰지 않는다.
- **환경**: rev 4/5 venv(xtb 6.7.1 `edcfbbe`, tblite 0.7.0)를 쓰고, 버전을 기록한다.

---

## 13. 하지 말 것

- 정답 열과 §3-1의 두 변경 말고는 Espley 코드를 고치지 않는다.
- hps.pkl(그들의 타깃으로 튜닝한 값)을 우리 라벨에 쓰지 않는다.
- gate를 느슨하게 해서 반응을 되살리지 않는다(R-4 사용자 결정 1건 제외).
- 시험 점수로 대표 모델이나 feature 세트를 고르지 않는다. feature는 182개로 고정이다.
- 결과를 본 뒤 행, 모델, 지표를 바꾸지 않는다.
- rev 4·5 결과 파일, 사전 등록 문서, 라벨, 원래 기하·feature 파일을 덮어쓰지 않는다.
- Espley 라벨 비교(rev 5 D-3)를 다시 돌리지 않는다. 필요하면 인용만 한다.

---

## 14. 논의 → 사양 대조표 (빠진 항목이 없는지 확인용)

| 대화에서 나온 내용 | 사양 위치 |
|---|---|
| 목적 = Espley와 공정 비교, 추가로 8채널 결과 | §0-1 |
| AM1→xTB 효과는 관심 없음 | §0-1, §11 |
| 182 feature 사용 (133 ⊂ 182, 182−133 CI 없음) | §0-3, §3 |
| 4,838 / 4,112 / 3,322가 헷갈림 → 공통 반응 하나 | §0-3, §2-1 |
| G1 실패 = TS 찾기 실패 (사유 9종, 400건) | §0-3, R-2 |
| hygiene 21 = 라벨 문제, 두 쪽에 같게 적용, 민감도에서 포함 | §2-1, §7 |
| 조각 나누기 실패 4건 → 라벨 A_idx 사용 | R-1 |
| feature 실패 1건(rxn 4994) 다시 넣기 | R-4 |
| 실패 반응 최대한 되살리기, 남은 실패는 공개·편향 점검 | Phase R, §2-2 |
| Espley는 계산하지 않고 그들의 코드·feature 그대로 | §0-3, §3-1, P-1 |
| Espley 저장 모델은 우리 시험 행을 이미 학습 → 같은 분할로 재학습 | §2-3, §3-1 |
| 정답지 통일(우리 라벨), 비교 타깃 Int·ΔE‡·d1·d2, ΔG‡ 제외 | §1 |
| Int 정의(eint_spe)와 6채널 합(e_bond)의 차이 | §1-1, §7, §8-3 |
| 8채널 실험과 4타깃 비교 실험을 따로 | Phase A, Phase B |
| 같은 학습/시험 분할, 같은 학습 행 수 | §2-3 |
| Espley에 XGB 추가, 모델별 결과 공개 | §3-1, §8-2, fig2 |
| 우리 모델 5종(SVR, KRR, Ridge, XGB, NN) | §3-2 (NN은 2층·4층) |
| 튜닝 조건 맞추기(5-1) | ESP-O, OUR-C |
| d1·d2 순서 맞추기(5-2) | P-3 |
| 사전 등록(5-3): 지표, 검정, 대표 모델 규칙 | Phase S, §3-3 |
| 대표 모델은 검증 점수로, 시험 점수로 고르지 않음 | §3-3 |
| 구조 출발점은 양쪽 모두 DFT TS (대등, 배포 성능 아님) | §0-3, §11 |
| Σ채널로 Int·ΔE‡ 만들기 (부 비교) | §8-3 |
| Espley 라벨 비교는 rev 5 D-3에 이미 있음 | §0-3, §13 |
| 짝지은 CI를 다시 계산할 수 있도록 예측 저장 | §8-1 |
