# GNN3_CLEAN — GNN3(3채널 짝 모델) 코드를 처음 보는 사람도 이해할 수 있게 정리하기 — Claude Code 실행 지시서

**목표:** GNN3 결과를 만든 코드를 하나의 폴더 `gnn3/`로 정리한다. 처음 보는 검토자가 README와 코드만 읽고 다음을 이해할 수 있어야 한다.
1. 무엇을 예측하는지
2. 입력이 어떻게 만들어지는지
3. 모델이 어떻게 생겼는지
4. 어떻게 학습 · 평가 · 예측하는지

GNN3 결과는 CH3M 결과(`analysis/ch3m/results/`, 단위 `D__lf3__S__GNN__f{0..4}`)이며, 3채널 strain · inter · solv와 barrier의 ΔΔ를 예측한다.

**제일 중요한 규칙: 숫자는 하나도 바뀌면 안 된다.** 이름 · 구조 · 주석 · 문서만 바꾸고 계산은 그대로 둔다. 같은 입력, 같은 GPU(RTX 3090)에서 기존 결과와 비트 단위로 같아야 한다(5절).

**저장소:** `abs3` 브랜치 HEAD **`04642c1c5f9dc100bb4d3bc599f8c315ab506659`**에서 새 브랜치 **`gnn3-clean`**을 만든다.
- worktree: `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-gnn3`
- 이 파일은 `docs/specs/GNN3_CLEAN.md`로 commit한다.
- 새 폴더 `gnn3/`(저장소 최상위)와 scratch `$GNN3=/gpfs/tmp_cpu2/yeseo1ee/gnn3`를 쓴다.
- **기존 코드는 고치거나 지우지 않는다.** 기존 결과의 출처라서 그대로 둔다. 새 코드는 기존 코드를 읽어 옮긴 사본이다.

**순서:** 1 시작 조건 → 2 조사(코드 지도) → 3 새 구조와 이름 → 4 옮기기 · 주석 → 5 동일성 검사 → 6 문서 → 7 처음 보는 독자 시험 → 8 commit · push → **멈춤**
- 10절 STOP 조건에서만 멈춘다. 옮기다 버그를 찾으면 **고치지 말고** 원래 동작을 그대로 옮긴 뒤 `CODE_MAP.md`의 "발견한 문제" 절에 적고 보고한다.
- 계산은 저장소 `CLAUDE.md`대로 `sbatch`와 `--time=48:00:00`을 쓴다.

---

## 1. 시작 조건 (아니면 STOP)

| # | 조건 |
|---|---|
| 1-1 | `origin/abs3` = `04642c1c5f9dc100bb4d3bc599f8c315ab506659` |
| 1-2 | GNN3 단위 파일이 있다: `/gpfs/tmp_cpu2/yeseo1ee/ch3m/units/D__lf3__S__GNN__f{0..4}.{json,parquet}`와 `__s{0..4}.pt`. 예측 parquet sha256이 `analysis/ch3m/results/KM_report.json`의 기록과 같다 |
| 1-3 | 입력이 ch3m과 같다: `$CH3/lf/X182_lf.parquet` sha256 `cd5ec799…f199`, 토큰 manifest sha256 `d9c91438…cf72`, `$LAD/F_pairs_cr2.csv` sha256 `f2fe5cab…a6ce`, T1 분할 5개(`prereg_ch3m.json` 값) |
| 1-4 | `env_d1fu2`가 동작한다(GPU, RTX 3090) |

결과: `gnn3/results/S0_start.json`

---

## 2. 조사 — GNN3 결과가 실제로 거친 코드 지도 (`gnn3/CODE_MAP.md` 초안)

다음 실행 경로에서 **실제로 불리는 함수만** 추적한다(import 추적 + 실행 추적). 쓰이지 않는 함수는 옮기지 않는다.

| 경로 | 시작점 |
|---|---|
| 학습(교차검증) | `analysis/ch3m/train_m.py run GNN 3` |
| 최종 모델 | `analysis/ch3m/final_m.py`(GNN 3모드로 부를 때) |
| 예측 | `analysis/ch3m/predict_m.py` |
| 데이터 | `analysis/ch3/data_ch3.py`, `analysis/deploy/data_lf.py`, `analysis/d1_pair/fu2/{train,models,fu2_common}.py`, `analysis/ch3/ch3_common.py` |
| 입력 만들기 | `analysis/d1_g2/g2_autode.py`(g2 TS 탐색), `analysis/deploy/finalize_lf.py`(라벨 없는 마무리), `analysis/deploy/feat_lf.py` · `meta_lf.py` · `tokens_lf.py`, `analysis/d1_pair/fu2/{atoms,tokens}.py`, `analysis/espley_xtb_repro/{xtb_slice,ext_features,rev5_common,rev6_common}.py` |
| 평가 | `analysis/ch3/eval_ch3.py`와 `analysis/ch3m/eval_m.py` 중 GNN3 지표(MAE, 부호, 순위, 원인 일치)에 쓰인 함수 |

**산출:** `CODE_MAP.md`

| 항목 | 내용 |
|---|---|
| 함수 표 | 원래 파일:함수(행) → 하는 일 한 줄 → 새 위치 · 새 이름 |
| 상수 표 | 원래 이름 → 값 → 단위 → 새 이름 (예: `I_BAR`, `LAMBDA`, grid, seed, patience) |
| 옮기지 않는 것과 이유 | 예: 사전 등록 · 커밋 도우미, 다른 모델(TF · AW · XGB), 사후 진단 |
| 바깥 의존 | 저장소 밖 코드(`$PILOT`의 `orca_direct.py`, `run_ts.py`, `pilot_common.py` 등 D1 실행 환경)와 버전(autodE 1.4.5, xtb 6.7.1, ORCA 6.1.1) |

---

## 3. 새 구조와 이름 규칙

```
gnn3/
  README.md              # 영어. 검토자용 첫 문서 (6절)
  README_ko.md           # 같은 내용 한국어
  MODEL_CARD.md          # 성능 · 쓰임 · 한계 (6절)
  CODE_MAP.md            # 2절 + 발견한 문제
  gnn3/                  # 파이썬 패키지
    constants.py         # 모든 상수: 이름, 값, 단위, 출처를 한 곳에
    channels.py          # 10 라벨 → 3채널(+barrier) 묶기 행렬과 설명
    data.py              # 토큰 · 전역 182 · 라벨 · 짝 · 분할 읽기, 표준화
    model.py             # GNN 인코더 + 반대칭 짝 머리
    training.py          # 손실, 한 epoch, 조기 종료, 튜닝, 교차검증 단위, 최종 모델
    predict.py           # 저장된 모델로 짝 ΔΔ 예측, seed 평균 · 표준편차
    metrics.py           # 논문에 쓰는 지표 (MAE, 부호, 계열 안 순위, 원인 일치)
    inputs/              # 입력 만들기: 기존 코드를 부르는 얇은 감싸개 + 설명 (아래)
  scripts/               # 명령줄: prepare_inputs.py, train_cv.py, train_final.py, predict.py, evaluate.py
  jobs/                  # sbatch 스크립트 (RTX 3090, 48 h)
  tests/                 # pytest (CPU) + 동일성 검사 (5절)
  results/               # 검사 결과 json
```

**이름 · 주석 규칙:**

| 항목 | 규칙 |
|---|---|
| 언어 | 코드 · 주석 · docstring은 영어(검토자용). README_ko만 한국어 |
| 이름 | 줄임말 대신 뜻이 보이는 이름. 예: `I_BAR` → `BARRIER_INDEX`, `fit` → `train_one_fit`, `ra, rb` → `rows_a, rows_b`. 화학 용어 줄임말(ΔΔ, TS, EDA)은 README 용어집에 있을 때만 |
| 단위 | 에너지 변수에는 `_kcal`, 거리에는 `_angstrom` 붙이기. 표준화된 값에는 `_std` |
| 배열 모양 | 모든 공개 함수 docstring에 입력 · 출력 배열 모양을 적음. 예: `[n_pairs, 4]` |
| docstring | 공개 함수 · 클래스마다 NumPy 형식: 하는 일 한 문장 → Parameters(모양 · 단위) → Returns → Raises → 화학적 이유가 있으면 Notes |
| 주석 | "무엇"이 아니라 "왜"를 쓴다. 예: 반대칭 머리를 쓰는 이유, 표준화 통계를 학습 반응으로만 내는 이유 |
| 타입 힌트 | 모든 공개 함수 |
| 지우는 것 | 쓰이지 않는 코드, 주석 처리된 코드, 사후 진단용 분기 |
| 형식 검사 | 환경에 `ruff`가 있으면 `ruff check`와 `ruff format`. 없으면 건너뛰고 기록(설치하려고 네트워크를 쓰지 않음) |

**숫자를 지키기 위한 규칙** (GPU에서 비트 단위로 같으려면 꼭 지킬 것):

| 항목 | 규칙 |
|---|---|
| 모듈 생성 순서 | 원래 GNN과 같은 순서로 하위 모듈을 만든다(같은 seed → 같은 초기 가중치) |
| 난수 사용 순서 | 섞기, 순서 증강, dropout, seed 설정을 원래와 같은 순서 · 횟수로 |
| 계산 순서 | 연산 순서 · dtype · 합치는 축을 바꾸지 않는다 |
| cuDNN · 결정성 | 원래 코드와 같은 설정 |
| 가중치 키 | `state_dict` 키 이름이 바뀌면 원래 키 → 새 키 대응표를 `model.py`에 두고, 기존 `.pt`를 그대로 읽을 수 있게 한다 |

**입력 만들기(`inputs/`)는 다시 쓰지 않고 감싸기만 한다.**
- g2 TS 탐색(autodE + xTB), 라벨 없는 마무리, xTB 특성 계산은 저장소 밖 D1 실행 환경 코드에 기대고 있다.
- 그래서 `prepare_inputs.py`가 기존 함수들을 **정해진 순서로 부르는** 얇은 감싸개를 만들고, 단계마다 무엇을 하고 무엇을 내는지 docstring으로 설명한다.
- 단계: SMILES → g2 TS → 라벨 없는 마무리 → xTB 단일점 → 원자 토큰 29 · 전역 182.

---

## 4. 옮기기 · 주석 달기

- 2절 표의 함수만 옮긴다. 옮길 때마다 `CODE_MAP.md`의 그 행을 채운다.
- 하나의 공개 진입점마다 짧은 사용 예를 docstring 맨 끝에 둔다.
- 순환 import가 없게 한다(`constants` → `channels` → `data` → `model` → `training` → `predict` · `metrics`).

---

## 5. 동일성 검사 — 하나라도 실패하면 STOP

| # | 검사 | 방법 | 기준 |
|---|---|---|---|
| E-1 데이터 | 새 `data.py`가 만드는 배열과 목록 | 토큰 텐서, 전역 182, y4, 짝 목록, fold별 train · test · inner 목록, 표준화 통계를 ch3 `Ch3Data`(`get_data('ch3')`)와 비교 | sha256 같음 |
| E-2 순전파 | 기존 가중치를 새 모델에 읽기 | `D__lf3__S__GNN__f{0..4}__s{0..4}.pt` 25개를 새 `model.py`로 읽어 각 fold 시험 짝을 예측하고, seed 평균을 KM_predictions(GNN, lf3)와 비교 | 같은 GPU에서 최대 차 ≤ 1e-6 kcal/mol |
| E-3 학습 재현 | 새 `training.py`로 fold 0을 처음부터 | 튜닝 4칸 + final 5 seed를 RTX 3090에서 돌려 `D__lf3__S__GNN__f0`와 비교 | 같은 칸, 같은 최적 epoch, 검증값 차 ≤ 1e-6, 시험 yhat 최대 차 ≤ 1e-4 |
| E-4 지표 | 새 `metrics.py`로 GNN3 지표 계산 | `EM_metrics.csv`(P_main para MAE), `EM_rank.csv`, `EA_cause.csv`의 GNN3 행 | 소수 넷째 자리까지 같음 |
| E-5 입력 감싸개 | `prepare_inputs.py`를 **기존 g2 폴더에서 출발**하는 단계부터 D1 10반응(dipole마다 2개, seed 20261111)에 실행 | 라벨 없는 마무리 → 특성 → 토큰 결과를 `$CH3/lf`의 같은 반응과 비교 | 토큰 · 182열 최대 차 ≤ 1e-8 |
| E-6 단위 시험 (CPU, pytest) | 모델 · 학습의 성질 | 반대칭 f(a,b) = −f(b,a), f(a,a) = 0, 출력 모양, 항등식 벌점 계산, 표준화 통계가 학습 반응만 씀, 묶기 행렬 M | 모두 통과 |

- E-5에서 g2 TS 탐색 자체(autodE 형태 탐색)는 다시 돌리지 않는다. 형태 탐색이 무작위라 같은 TS가 나온다는 보장이 없다. README에 이 점을 적는다.
- E-3이 비트 단위로 같지 않은데 E-2는 같으면, 학습 루프의 난수 · 연산 순서가 바뀐 것이다. 고친 뒤 다시 돌린다(두 번까지). 그래도 다르면 STOP.

결과: `gnn3/results/E_equivalence.json`(`all_pass`, 검사별 수치)

---

## 6. 문서

**`README.md`** (영어, 같은 내용의 `README_ko.md`):

| 절 | 내용 |
|---|---|
| 1. What it predicts | 한 문단 + 에너지 다이어그램 설명. ΔΔ = (치환 반응) − (같은 계열 H 기준 반응), 3채널 정의, 단위 kcal/mol, 물(SMD), 전자 에너지 |
| 2. Pipeline | mermaid 그림: SMILES → g2 TS(autodE + GFN2-xTB, ALPB water) → label-free finalize → xTB features(atom tokens 29, global 182) → GNN encoder(shared) → antisymmetric pair head → ΔΔ ×4 (seed 5개 평균). 단계마다 코드 위치 |
| 3. Glossary | pair, family, H reference, anchor pair, channel, strain/inter/solv, cr2 reference structures, g2, lf, T1 fold(core 단위), P_main, para/meta/ortho/direct, seed spread |
| 4. Data contracts | 파일마다 열 · 모양 · 단위 · 출처: 라벨(`F_pairs_cr2.csv`), 토큰 npz, `X182_lf.parquet`, 분할 json, 예측 parquet |
| 5. Model | 층 · 폭 · 풀링 · 전역 투영 · 머리 · 반대칭 이유 · 손실(표준화 L1 + 항등식 벌점 λ) · 하이퍼파라미터 표 |
| 6. Training & evaluation | 교차검증 절차(core 단위 5 fold, inner 튜닝, 조기 종료, final, seed 5개) · 실행 명령 · 예상 시간 |
| 7. Predicting new reactions | H 기준 반응도 g2로 만들어야 함, 적용 범위 문턱(seed 표준편차), g2 실패 처리 |
| 8. Reproducibility | 버전(Python · torch · autodE · xtb · ORCA), GPU 종류, 결정성 설정, 동일성 검사 결과 요약 |

**`MODEL_CARD.md`:**
- 성능: CH3M · ABS3 결과에서 GNN3 행만 가져온다.
  - P_main para · all MAE
  - 부호 정답률
  - 계열 안 순위(순서 정답률, 상위 3, Spearman)
  - 원인 일치율(칸 평균)
  - 위치별(direct · o · m · p)과 골격별 MAE · R²
- 쓰임: 같은 계열 안 작용기 효과의 3채널 ΔΔ.
- 한계:
  - 골격이 다른 반응끼리는 비교할 수 없다(학습 짝에 없음).
  - cyclooctyne(D)은 정확도가 낮다(R² 0.13).
  - ortho · meta 개별 값은 약하다.
  - DFT 대비 정확도이지 실험 대비가 아니다.
  - 물(SMD) 전자 에너지만 다룬다.
  - g2 실패율은 약 7 %다.

---

## 7. 처음 보는 독자 시험

**새 하위 에이전트**를 하나 띄운다. 이 작업 내용을 모르고, `gnn3/` 폴더만 읽을 수 있게 한다. 그 에이전트가 다음 10문항에 답하게 한다.

1. 모델이 예측하는 양 4개와 단위는?
2. ΔΔ의 기준 반응은 무엇이고, 새 반응을 예측할 때 무엇을 더 계산해야 하나?
3. 입력 두 종류(원자 토큰 · 전역 특성)의 모양과 만드는 단계는?
4. DFT는 어디에 쓰이고 어디에 쓰이지 않나?
5. 모델의 출력이 a, b 순서를 바꾸면 어떻게 되며, 왜 그런가?
6. 손실 함수의 두 항은?
7. 교차검증은 무엇 단위로 나누며, 왜?
8. 최종 모델의 칸 · epoch는 어떻게 정하나?
9. 새 SMILES 짝 하나를 예측하는 명령은?
10. 이 모델을 쓰면 안 되는 경우 두 가지는?

**채점과 반영:**
- 채점은 이 지시서와 원래 결과를 기준으로 한다.
- 틀리거나 "문서에 없음"이라고 답한 문항은 문서 · docstring을 고친다.
- 고친 뒤 **새 에이전트로** 한 번 더 시험한다.

결과: `gnn3/results/R_reader_test.json`(문항별 답, 정답 여부, 고친 곳)

---

## 8. commit · push

- commit 대상: `docs/specs/GNN3_CLEAN.md`, `gnn3/` 전체(코드 · 문서 · 시험 · 결과 json).
- commit하지 않음: `$GNN3/` 아래(재학습 산출, 로그).
- 단계마다 commit한다(S0 · 지도 · 옮기기 · 동일성 · 문서 · 독자 시험).
- 마지막에 `git push -u origin gnn3-clean`를 하고, HEAD SHA를 `gnn3/results/SUMMARY.md` 끝에 적는다.

**`SUMMARY.md` (한국어):**
- 새 구조 한눈에 보기
- 동일성 검사 표
- 이름을 바꾼 주요 항목 20개
- 옮기지 않은 것
- 발견한 문제
- 독자 시험 결과
- 사용한 GPU-h

---

## 9. 예상 비용

| 단계 | 예상 |
|---|---|
| E-2 | < 0.1 GPU-h |
| E-3 | 약 1 GPU-h (CH3M GNN3 fold 하나) |
| E-5 | < 1 core-h |
| 나머지 | CPU 몇 분 |

## 10. STOP 조건

- 시작 조건이 실패한다.
- 동일성 검사가 실패한다(E-3은 5절 예외대로).
- 옮기다 기존 결과를 바꿔야만 하는 상황이 생긴다.

버그를 찾은 것은 STOP이 아니다(기록하고 보고).

## 11. 범위 밖

- 계산 · 하이퍼파라미터 · 특성의 어떤 변경도 하지 않는다.
- TF · AW · XGB · 절대값 모델은 옮기지 않는다.
- GNN3 최종 모델 학습과 14만 반응 배포는 하지 않는다.
- 기존 `analysis/` 코드를 수정 · 삭제하지 않는다.
