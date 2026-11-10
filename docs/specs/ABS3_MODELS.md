# ABS3 — 반응 하나의 3채널 절대값을 예측하는 모델(TF · GNN · XGB)을 만들어 짝(ΔΔ) 모델과 비교하기 — Claude Code 실행 지시서

**용어:** **D1** = 기존 D1 라벨(재라벨링 안 함, 반응물 기준 구조는 LADDER cr2판). **3채널** = strain(d1 + d2) ·
inter(elst + pauli + oi + disp + c_ghost) · solv(cpcm + cds) + barrier, 즉 타깃 4개.
**짝 모델** = 지금까지의 ΔΔ 모델(TF3 = ch3, GNN3 · XGB3 = CH3M). **절대값 모델** = 이번에 새로 만드는 모델.

**저장소:** `ch3m` 브랜치의 마지막 commit **`36b8e17b4098288a99a9c87c0f69b8e89aa9a0bb`**에서 새 브랜치 **`abs3`**을 만든다.
worktree는 `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3`이다. 이 파일은 `docs/specs/ABS3_MODELS.md`로 commit한다.

| 구분 | 위치 |
|---|---|
| 새 코드 | `analysis/abs3/` |
| 결과 | `analysis/abs3/results/` |
| scratch | `$ABS3=/gpfs/tmp_cpu2/yeseo1ee/abs3` |
| 읽기만 | `analysis/ch3`, `analysis/ch3m`, `analysis/deploy`, `analysis/d1_pair`, `analysis/d1_ladder`, `analysis/geo2`의 코드와 결과 · `$CH3/lf`(재생성 lf 입력) · `$LAD` · `$D1PAIR_SCRATCH` · `$CH3M/xgb_pack` |

**기존 코드는 고치지 않는다.** import해서 쓰고, 새 동작은 `analysis/abs3/`에 둔다.

**순서:** 1 시작 조건 → 2 브랜치 → 3 새 코드 → 4 라벨 · 분할(LS) → 5 코드 검사(C) → 6 사전 등록(R) → 7 교차검증 학습(K) → 8 평가(E) → 9 요약(S) → 10 commit · push → **멈춤**
- 12절 STOP 조건에서만 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다.
- 모든 계산은 `sbatch`로, `--time=48:00:00`을 쓴다(저장소 `CLAUDE.md`).

---

## 0. 무엇을, 왜

### 0-1. 목표 (사용자 결정 2026-11-11)

짝 모델은 두 반응(치환 반응 a와 같은 계열의 H 기준 반응 b)을 함께 넣고 **차이 ΔΔ(a − b)**를 낸다.
절대값 모델은 **반응 하나**를 넣고 그 반응의 3채널 값을 낸다. 같은 입력 · 라벨 · 분할에서 세 모델(TF · GNN · XGB)로 두 방식을 비교한다.

| 질문 | 내용 |
|---|---|
| **Q-A** | 절대값 모델은 반응 하나의 3채널을 얼마나 맞히는가? 짝 모델은 이것을 할 수 없다 |
| **Q-D** | 절대값 모델의 예측을 빼서 만든 ΔΔ = ŷ(a) − ŷ(b)가 짝 모델의 ΔΔ보다 나은가, 같은가, 나쁜가? 모델 종류마다 답한다 |
| **Q-C** | Q-D의 결과를 오차 상쇄로 설명할 수 있는가? 같은 계열 두 반응의 절대값 오차가 얼마나 같이 움직이는가를 본다 |

- **사전 가설 H2:** 짝 모델이 ΔΔ에서 더 낫다. 짝으로 묶으면 계통 오차가 상쇄되기 때문이다.
  - 근거: DeepDelta(Fralish et al., *J. Cheminform.* 2023, doi:10.1186/s13321-023-00769-x)는 두 분자를 함께 넣는 짝 모델이 분자 하나 모델(ChemProp, Random Forest)보다 성질 차이를 더 잘 맞혔다고 보고했다. 10개 ADMET 벤치마크 중 Pearson r 기준 70 %, MAE 기준 60 %에서 그랬다. 이유로 계통 오차 상쇄를 들었고, 작은 데이터(< 1,500)에 가장 맞는다고 했다.
  - 이번 데이터는 약 8,000반응으로 더 크므로, 그 이점이 줄어들 수 있다.
- 이 실험은 **비교**만 한다. 최종 모델(W)과 배포 모델 교체는 범위 밖이다(13절).

### 0-2. 실험 칸

| 모델 | 절대값 (새로) | 짝 ΔΔ (재사용) |
|---|---|---|
| TF | `A__lf3__S__TF__f{0..4}` | TF3 = `analysis/ch3/results/K3_predictions.parquet` |
| GNN | `A__lf3__S__GNN__f{0..4}` | GNN3 = `analysis/ch3m/results/KM_predictions.parquet`(rung lf3, GNN) |
| XGB | `A__lf3__S__XGB__f{0..4}` | XGB3 = 같은 파일(rung lf3, XGB) |

입력은 짝 모델과 같은 재생성 lf 입력 `$CH3/lf`(g2 구조, 라벨 없는 마무리, xTB 특성, DFT · HF 없음)이다.

### 0-3. 짝 모델과 다른 점 (이것만 다르다)

| 항목 | 짝 모델 | 절대값 모델 |
|---|---|---|
| 학습 단위 | 짝 (a, b) | 반응 하나 |
| 출력 | ΔΔ 4개 | 절대값 4개 |
| 정답 | `F_pairs_cr2.csv`의 `dd_<t>` @ M | 반응별 cr2 라벨 @ M (4절) |
| 머리 | `½[f(r_a, r_b) − f(r_b, r_a)]`, 입력 2·d_r | `f(r)`, 입력 d_r. 은닉층은 같게(2층, 폭 2 × width) |
| 표준화 | ΔΔ의 표준편차(평균 0) | 학습 반응의 평균과 표준편차 |
| 증강 | (a, b) ↔ (b, a) 순서 바꾸기 | 없음 |
| XGB 특성 | [Δx / s, (x̄ − μ) / s], 양방향, 반대칭 | (x − μ) / s, 한 방향 |
| TF 교차 주의 | 있음: `cross` 블록에서 a의 토큰이 b를, b의 토큰이 a를 본다 | **없음**: 반응이 하나라 볼 상대가 없다 |

인코더는 `fu2/models.py`의 것을 그대로 쓴다(토큰 29 + 전역 182, "A+G").
- GNN: `readout` = 원자 합 + 주의 풀링 + 전역 투영.
- TF: `tokens` → `pooled`(평균 + 주의 풀링). 짝 TF의 `cross` 블록은 빠진다. 그래서 TF 비교에는 "교차 주의가 있느냐"라는 구조 차이가 함께 들어간다. 요약에 이 점을 적는다.

### 0-4. 공통 설계

| 항목 | 값 |
|---|---|
| 반응 집합 | 짝 모델 분할의 각 목록(train · test · inner_train · inner_val)에 들어 있는 사용 가능 짝(`DeployData('lf')`, P_lf)의 반응 a와 b를 모두 모은 것. T1과 inner 분할은 core 단위로 묶여 있으므로 train과 test 반응 집합이 겹치지 않아야 한다(4절 확인). 그래서 절대값 모델은 짝 모델과 **같은 반응**을 보고 같은 반응에서 시험받는다 |
| TF · GNN 학습 | 짝 모델(DEPLOY 11-2 = FU2 M-1, ch3 3모드)과 같은 값: grid(TF 2/3층 × 64/128, GNN 4/6층 × 64/128), AdamW 1e-3 / wd 1e-4, cosine 300 epoch, dropout 0.1, patience 30, seed 20261019 + 0..4 평균. 튜닝은 inner_train / inner_val, 검증 지표 = 4타깃 표준화 MAE 평균(ch3 3모드와 같은 정의). final = 바깥 학습 반응, 튜닝 최적 epoch |
| 배치 | 반응 64개(짝 모델의 "64짝"과 같은 숫자; 기록) |
| 손실 | FU2 `loss_fn`과 같은 형태: 표준화 L1 + λ · \|ŷ_barrier − (ŷ_strain + ŷ_inter + ŷ_solv)\| / s_barrier. 여기서 ŷ는 kcal(= μ + s · 출력)이고 λ = `fu2_common.LAMBDA` |
| 토큰 수치 표준화 | FU2와 같은 함수를 쓰고, 통계는 그 fit의 **학습 반응만**으로 낸다 |
| XGB | `train_ml_single.GRIDS["XGB"][0]`를 clone하고 `d1_train.grid_cells("XGB")` 12칸으로 바꾼다(짝 XGB와 같은 추정기와 grid). 타깃마다 따로: inner 4 fold(짝 XGB와 같은 규칙인 거르기 전 바깥 학습 짝 목록 → core GroupKFold(4) → 사용 가능 짝 → 그 짝들의 반응)에서 평균 MAE로 칸을 고르고, 바깥 학습 반응 전부로 final fit. X = 182열(`$CH3/lf/X182_lf.parquet`)을 학습 반응 평균 · 표준편차로 z-score한 것. y = 표준화 |
| 관문 | C-5 누출(시험 반응이 학습 반응 · 표준화 통계 · inner 집합에 없음). 반대칭 관문은 해당 없음 |
| GPU | RTX 3090(`-p gpu1 --gres=gpu:rtx3090:1`), 짝 모델과 같은 종류 |

---

## 1. 시작 조건 (하나라도 아니면 STOP)

| # | 조건 |
|---|---|
| 1-1 | `origin/ch3m` = `36b8e17b4098288a99a9c87c0f69b8e89aa9a0bb` |
| 1-2 | `$CH3/lf/X182_lf.parquet` sha256 = `cd5ec799f088a72aa31a2682240f0657469d07db5f5fc56050aa92765fb7f199`. `analysis/ch3/results/lf_inputs/fu2_lf/T_manifest.csv` sha256 = `d9c914380a235aae919d503bd5eed2034cfba5c20073c519476dfd0be5bacf72`. 토큰 파일 8,502개 |
| 1-3 | `$LAD/F_pairs_cr2.csv` sha256 = `f2fe5cab1c61f082b7cf929ca1146d96f8e451d37ed7d4ee2b5b127b4a7fa6ce`. `$LAD/labels_d1_cr2.json` sha256 = `7e5882bce4d47dcd8164b61320a390530463bfa14731029c122d7f09920b6755`. `analysis/d1_ladder/results/labels_d1_cr2.parquet` sha256 = `696d2ec6ee1b75b5f33bad91fcf9368c528644eecdde4c3d226be3200cd447df`(모두 `analysis/d1_ladder/results/V_report.json` outputs 값) |
| 1-4 | 짝 모델 예측: `K3_predictions.parquet` sha256 = `4faefc9c5e38c1a8e6b1592c1c3a476cc31c9edfa1d1fea9e8ce388cde0fa24e`, `KM_predictions.parquet` sha256 = `da01baf40b125f75b03cf5f8e79b37468ddd87cca36c0196334952d63ea6ea6e` |
| 1-5 | FU2 T1 분할 5개와 `S_folds.csv`의 sha256 = `analysis/ch3m/results/prereg_ch3m.json` inputs_sha256 값 |
| 1-6 | `env_d1fu2`(`.d1fu2_env_ok`), `env_espley`(xgboost 3.2.0 · scikit-learn 1.3.1, CH3M S0와 같음) |
| 1-7 | 디스크 여유 10 GB 이상 |

결과: `analysis/abs3/results/S0_start.json`

---

## 2. 브랜치 · 폴더

```
git fetch origin ch3m
git worktree add -b abs3 /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3 36b8e17b4098288a99a9c87c0f69b8e89aa9a0bb
```
- 이 지시서를 commit한다.
- `sbatch --time=48:00:00 --wrap "mkdir -p $ABS3/{logs,units,smoke,pack}"`

---

## 3. 새 코드 (`analysis/abs3/`)

| 파일 | 내용 |
|---|---|
| `abs_common.py` | 경로(`ABS3`, `RES`), 쓰기 가드(`$ABS3`와 `analysis/abs3/results` 밖이면 STOP), 단위 이름, 모델 라벨(`A-TF`, `A-GNN`, `A-XGB`, `TF3`, `GNN3`, `XGB3`), 원자적 쓰기, sha256, STOP(exit 3) · not ready(exit 2) |
| `abs_data.py` | `analysis/ch3/data_ch3.py`를 import한다(DEP_SCRATCH 묶기 그대로). `DeployData('lf')`의 토큰 · 사용 가능 규칙 · 분할 목록 위에 **반응 단위 데이터**를 만든다: 반응 목록, 반응별 절대값 y4(4절), fold별 반응 집합(train · test · inner_train · inner_val), 짝 → 반응 대응표 |
| `abs_model.py` | `AbsNet`: `Mo.build(model, cell, d_in=29, n_out=4, d_glob=182, "A+G", dropout)`를 인코더로 쓰고, 그 짝 머리(`head`)와 TF의 `cross` 블록은 쓰지 않는다. 반응 벡터 r: GNN = `enc.readout(R)`(d_r = 3·width), TF = `enc.pooled(*enc.tokens(R))`(d_r = 2·width). 새 머리 = `Mo.mlp(d_r, 2*width, 4, dropout, n_hidden=2)`. d_r은 더미 배치로 한 번 돌려 확인한다. forward: `R, ia, _ = FT.batch(T, rows, rows)` → `head(r(R))[ia]`. 옵티마이저에는 실제로 쓰는 파라미터만 넣는다(짝 머리 · cross 제외) |
| `train_abs.py` | TF · GNN. FU2 `fit`의 반응 단위판이다: 0-4의 학습 규칙, 조기 종료, 튜닝 → final, seed 5개. 단위 기록 형식은 ch3m `train_m.py`와 같다(칸, 최적 epoch, 검증값, 시간, GPU 이름, 지문). 명령: `list`, `smoke`, `overfit <TF\|GNN>`, `run <TF\|GNN> <i\|@I> <n>`, `collect` |
| `abs_pack.py` | env_d1fu2 CPU. XGB용 묶음 `$ABS3/pack/lf_abs.npz` + `.json`: 반응 id(토큰 순서), X = `X182_lf.parquet` 행, y4 절대값, fold별 반응 집합, 거르기 전 바깥 학습 짝 목록과 짝별 core · 반응 a · b(XGB inner fold용), 원천 sha256 |
| `xgb_abs.py` | env_espley. 0-4의 XGB 규칙. `train_ml_single.GRIDS`와 `d1_train.grid_cells`는 import해서 쓴다. inner 4 fold는 ch3m `xgb_m.py`(= LADDER `XData.split`)의 짝 단위 규칙을 그대로 쓴 뒤 반응으로 바꾼다. 명령: `list`, `smoke`, `overfit`, `run <i\|@I> <n>`, `collect` |
| `eval_abs.py` | env_espley. 8절. ch3 `eval_ch3.py`의 정의 · 함수(`boot_rel`, `metrics`, `rank_table`, `cause`, `identity_residual`)를 import해서 쓴다 |
| `prereg_abs.py`, `tests_abs.py` | 6절, 5절 |
| `jobs/gpu_abs.slurm`, `jobs/cpu_abs.slurm` | ch3m `jobs/gpu_m.slurm` · `cpu_m.slurm` 복사본. 로그는 `$ABS3/logs` |

---

## 4. Phase LS — 반응별 절대값 라벨과 반응 단위 분할 (`abs_data.py summary`, `abs_pack.py`; CPU)

**라벨:** `$LAD/labels_d1_cr2.json`의 반응별 값. 키 대응은 `V_report.json` pairs.code.d1_pairs를 따른다.

| 타깃 | 키 |
|---|---|
| d1 | `d1_kcal` |
| d2 | `d2_kcal` |
| elst | `elst_dft` |
| pauli | `pauli_dft` |
| oi | `oi_dft` |
| disp | `disp_dft` |
| cpcm | `cpcm_dft` |
| cds | `cds_dft` |
| c_ghost | `bsse_shift_kcal` |
| barrier | `barrier_kcal` |

y4 = y10 @ M(`ch3_common.M`)으로 만든다. 키가 없으면 STOP한다.

**확인 (다르면 STOP):**

| # | 검사 | 기준 |
|---|---|---|
| LS-1 | 짝 재구성: `F_pairs_cr2.csv`의 D1 짝 전부에서 `dd_<t>` = y_t(a) − y_t(b), 10타깃 | 최대 차 ≤ 1e-6 kcal/mol |
| LS-2 | 반응 단위 항등식: \|barrier − (strain + inter + solv)\| 분포 | 최대 ≤ 0.12(라벨 QC의 Σ6 − Bond 잔차 최대 0.11 + 반올림). 분포를 보고한다 |
| LS-3 | fold k마다 train 반응 ∩ test 반응 = ∅, inner_train ∩ inner_val = ∅. fold k 시험 짝의 a와 b가 모두 test 반응에 있음 | 모두 참 |
| LS-4 | 반응 수: fold별 train / test / inner 반응 수, 전체 반응 수(사용 가능 짝에 나오는 반응) | 보고 |
| LS-5 | XGB 묶음의 반응 · X · y4 · 집합이 `abs_data`와 같음 | sha256 같음 |

결과: `results/LS_report.json`

---

## 5. Phase C — 코드 검사 (`tests_abs.py`) — 하나라도 실패하면 STOP

| # | 검사 | 기준 |
|---|---|---|
| C-E | 인코더 동일성: 같은 seed로 만든 `AbsNet.enc`와 `Mo.build(...)`가 같은 배치에서 같은 반응 벡터를 냄(GNN `readout`, TF `pooled(tokens)`). 옵티마이저 파라미터에 짝 머리 · `cross`가 없음 | 최대 차 0, assert |
| C-O | 과적합 시험: fold 0 학습 반응 64개, 가장 큰 칸, 300 epoch, dropout 0, 1 seed. TF · GNN · XGB 각각 | 그 64개의 학습 MAE ≤ 각 타깃 표준편차의 10 % (학습 경로가 실제로 배우는지) |
| C-T | 배선: 출력 4개, 표준화 평균 · 표준편차가 학습 반응만으로 계산됨, 손실 항등식 벌점이 kcal로 계산됨, 검증 지표 = 4타깃 평균, 단위 기록의 타깃 · 열 | assert |
| C-5 | 누출: 시험 반응이 학습 반응 · 표준화 통계 · inner 집합에 없음(모든 fit) | assert |
| C-X | XGB 결정성: fold 0 한 타깃 한 칸을 두 번 fit | 예측 차 0 |
| C-P | ΔΔ 조립: 반응 예측표에서 만든 짝 ΔΔ가 시험 짝의 `ŷ(a) − ŷ(b)`와 같고, 정답 ΔΔ가 `F_pairs_cr2 dd @ M`과 같음 | 최대 차 ≤ 1e-9 |
| C-8 | smoke: TF · GNN fold 0, 칸 최소 · 최대, 2 epoch, 1 seed. XGB fold 0, 칸 첫 · 끝 | 끝까지 돌고 C-T · C-5 통과. 산출은 `$ABS3/smoke` |

결과: `results/C_tests.json`(`all_pass`), `C_overfit.json`, `C8_smoke.json`

---

## 6. Phase R — 사전 등록 (`results/PREREG_ABS3.md` + `prereg_abs3.json`)

K 학습 전에 commit한다. `run`은 두 파일이 git에 있고 바뀌지 않았을 때만 돈다(ch3m과 같은 규칙).
md에 적을 것은 0절 전체(질문, 가설 H2, 칸, 다른 점, 설계), 8절 지표 정의와 해석 규칙, 모든 입력 sha256이다.

---

## 7. Phase K — 교차검증 학습

| 단위 | 장비 | 명령 | 예상 |
|---|---|---|---|
| `A__lf3__S__TF__f{0..4}` | RTX 3090 | `train_abs.py run TF @I 5` | 약 0.7 GPU-h |
| `A__lf3__S__GNN__f{0..4}` | RTX 3090 | `train_abs.py run GNN @I 5` | 약 3 GPU-h |
| `A__lf3__S__XGB__f{0..4}` | CPU 32코어 | `xgb_abs.py run @I 5` | 약 10 CPU-h |

- 단위 산출: `$ABS3/units/<unit>.{json,parquet}`(TF · GNN은 `__s{0..4}.pt`도). 지문이 같으면 건너뛰고, 다르면 STOP.
- `--array=0-4%5`. 대기 20 상한을 지킨다.
- **collect → `results/KA_predictions.parquet`** (반응 단위)
  - 열: `rxn_id, fold, model, target, y, yhat, yhat_seed_sd(XGB는 NaN), dipole, skeleton, series, attach`
  - 함께: `KA_val.csv`, `KA_report.json`(단위별 칸 · epoch · 시간, 합계 GPU-h · CPU-h, 반응 수)
- **짝 ΔΔ 조립 → `results/KA_pairs.parquet`**
  - fold k의 시험 짝마다 ŷ(a) − ŷ(b)를 만든다.
  - 열은 짝 모델 `K_predictions`와 같다(`pair_id, ref_kind, pos, fold, rung, arm, model, target, y, yhat, base, beta, yhat_seed_sd`). `rung = abs3`, `model ∈ {A-TF, A-GNN, A-XGB}`.
  - yhat_seed_sd는 seed별 ΔΔ의 표준편차다.
  - C-P 규칙으로 확인한다.

---

## 8. Phase E — 평가 (`eval_abs.py run`)

### 8-1. 공통

정답 확인과 집합 · 부트스트랩 규칙은 CH3M 8-1과 같다.
- 정답 확인: 짝 정답은 `F_pairs_cr2 @ M`, 반응 정답은 4절 y4. 다르면 STOP한다.
- 집합: P_main · para 공통 짝이 주 집합이고, all은 보조다.
- 부트스트랩: core 단위 2,000회, seed 20261019.

### 8-2. Q-A — 절대값 정확도 → `EA_abs.csv`

- **대상:** 시험 반응 전부(fold마다 그 fold의 시험 반응). 모델 3개 × 타깃 4개.
- **지표:** MAE, RMSE, r, R², MAE의 core 부트스트랩 95 % CI.
- **기준값:**
  - 학습 반응 평균으로 예측했을 때의 MAE(fold마다 계산)
  - 정답 표준편차
- **나눠 보기:** dipole별, 부착(direct · o · m · p)별, 치환 반응 대 H 기준 반응.
- **주의:** T1은 core(dipole × 골격) 단위로 떼어 놓은 분할이라, 시험 반응은 **처음 보는 core**의 반응이다.

### 8-3. Q-D — ΔΔ에서 절대값 모델 대 짝 모델 → `ED_compare.csv`

모델 종류마다 A-x와 x3을 비교한다(TF: A-TF 대 TF3, GNN: A-GNN 대 GNN3, XGB: A-XGB 대 XGB3).

- **지표:** P_main para(주)와 all(보조)에서 타깃 4개마다 상대 차 = (MAE(A-x) − MAE(x3)) / MAE(x3), 95 % CI.
- **해석 규칙 (CH3M 8-2와 같음):**
  - 흐름 기준 noise_t = CH3M `EM_noise.csv`의 같은 위치 · 타깃 \|TF10 대 dep\| 값이다.
  - 문턱은 max(noise_t, 2 %)이다.

| 표시 | 조건 |
|---|---|
| 절대값 나음 | 95 % 상한 < 0 이고 점추정 < −문턱 |
| 짝 나음 | 95 % 하한 > 0 이고 점추정 > +문턱 |
| 같음 | 그 밖 |

- **추가 한 줄:** 가장 좋은 절대값 모델(4타깃 para MAE 평균 최소) 대 GNN3(현재 가장 좋은 짝 모델)을 같은 형식으로 적는다.

### 8-4. Q-C — 오차 상쇄 → `EC_cancel.csv`

절대값 모델마다 P_main para · all 시험 짝에서 타깃마다 다음을 계산한다.
- corr(e(a), e(b)): 같은 짝 두 반응의 절대값 오차 상관
- 상쇄 비율 = 평균 \|e(a) − e(b)\| / 평균(\|e(a)\| + \|e(b)\|). 낮을수록 차이를 낼 때 오차가 많이 지워진다는 뜻이다.
- 8-3의 상대 차를 함께 적는다.

H2에 맞는지는 표로만 보고하고, 검정하지 않는다.

### 8-5. 보조 지표 (판정 아님, 6개 모델: A-TF · A-GNN · A-XGB · TF3 · GNN3 · XGB3)

정의는 `eval_ch3.py` 그대로 쓴다(P_sens 기준 공통 짝, 예측 자신의 barrier 부호).

| 산출 | 내용 |
|---|---|
| `EA_metrics.csv` | ΔΔ: 타깃 × 위치별 MAE, RMSE, r, 부호 정답률(\|y\| ≥ 1) |
| `EA_rank.csv` | 계열 안 장벽 순위: 두 작용기 순서 정답률(0.41–1 / 1–2 / > 2), 실제 최저가 예측 1등 · 상위 3, Spearman |
| `EA_cause.csv` | 주도 채널 · 유형(S / SI / I) 일치율. 반응 하나와 칸 평균, 기준값 함께 |
| `EA_identity.csv` | 반응 단위 \|ŷ_barrier − (ŷ_strain + ŷ_inter + ŷ_solv)\|(절대값 모델)와 짝 단위 같은 값(6개 모델) |

### 8-6. selftest (`eval_abs.py selftest`, K 전에 실행)

같은 코드로 CH3M `EM_metrics.csv`의 GNN3 · TF3 · XGB3 P_main para MAE 행과 `EM_rank.csv`의 세 행을 소수 넷째 자리까지 재현한다. 다르면 정의 차이를 보고한다(STOP 아님).

산출: `results/EA_selftest.json`, `EA_report.json`(8-2 · 8-3 · 8-4 요약, 표시, 입력 sha256)

---

## 9. Phase S — 요약 `results/SUMMARY_ABS3.md` (한국어)

1. 한 줄 결론: Q-A · Q-D · Q-C
2. C 검사 결과(과적합 시험 수치 포함)
3. Q-A 표: 모델 × 타깃 MAE · R² · CI, 평균 예측 기준값, dipole · 부착별 요약
4. Q-D 표: 모델별 A-x 대 x3 para MAE와 상대 차 · CI · 표시(all 보조), 그리고 가장 좋은 절대값 모델 대 GNN3
5. Q-C 표와 H2 해석
6. 보조 지표 요약(순위 1등 · 상위 3 · Spearman, 원인 일치율 칸 평균, 항등식 잔차)
7. 단위별 칸 · epoch, 실제 GPU-h · CPU-h
8. 편차 목록

---

## 10. commit · push

- commit 대상: `docs/specs/ABS3_MODELS.md`, `analysis/abs3/` 코드 · job, `analysis/abs3/results/`. 50 MB가 넘는 파일은 경로만 적는다.
- commit하지 않음: `$ABS3/` 아래 전부(`CLAUDE.md`).
- 단계마다 commit한다(S0 · LS · C · PREREG · K · E · S). PREREG commit은 K 제출 전에 끝나야 한다.
- 마지막에 `git push -u origin abs3`를 하고 HEAD SHA를 요약 끝에 적는다.

---

## 11. 예상 비용

| 단계 | 예상 |
|---|---|
| LS | < 0.5 core-h |
| C | < 0.3 GPU-h, < 1 CPU-h |
| K | 약 4 GPU-h(RTX 3090), 약 10 CPU-h |
| E | < 1 CPU-h |

근거: 반응 단위 학습은 한 epoch에 반응을 한 번씩만 인코딩한다. 짝 학습은 짝마다 두 번 인코딩하므로 epoch당 계산이 약 절반이다(CH3M GNN3 5.0, TF10 1.25 GPU-h 기준). XGB는 양방향 복제가 없고 데이터가 반응 수만큼이다(CH3M XGB3 18.8 CPU-h 기준).

---

## 12. STOP 조건

| 단계 | 조건 |
|---|---|
| 1 | 시작 조건 실패 |
| 4 | LS-1 · LS-3 · LS-5 실패, 라벨 키 없음 |
| 5 | C 검사 실패 |
| 6 | PREREG가 git에 없거나 바뀜 |
| 7 | 지문이 다른 캐시, C-5 · C-T · C-P 실패 |
| 8 | 정답이 라벨과 다름 |

결과가 H2와 반대여도 STOP이 아니다.

## 13. 실행 규칙과 범위 밖

- 실행 규칙은 CH3M 14절과 같다. sbatch만 쓰고 48 h. 로그인 노드에 상주 프로세스를 두지 않는다. 실행 10 / 대기 20(배열 원소마다 1개). `--dependency=afterany` 체인. GPU는 RTX 3090만 쓰고, CPU는 빈 파티션을 쓴다. 모든 단계는 멱등이고 원자적으로 쓴다.
- 환경: LS(`abs_data`, `abs_pack`), TF · GNN의 C · K · collect는 `env_d1fu2`. XGB의 C · K · collect와 E는 `env_espley`.
- **범위 밖:**
  - 최종 모델(W) · 추론기 · 배포 모델 교체
  - 10타깃 절대값 모델, AW
  - LODO · LOSO 분할
  - 짝 모델 재학습
  - "절대값 모델 + 짝 모델" 결합
