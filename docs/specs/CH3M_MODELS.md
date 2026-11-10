# CH3M — 3채널 직접 예측 대 8채널 예측의 합: TF · GNN · XGB 세 모델로 비교하고 배포 모델 다시 고르기 — Claude Code 실행 지시서

**용어:** 이 문서에서 **D1** = 기존 D1 라벨이다(재라벨링 안 함). 라벨 파일은 배포 모델 lf · ch3과 같은
`$LAD/F_pairs_cr2.csv`다.

**저장소:** `ch3` 브랜치의 마지막 commit **`6a4f59c26a3c4db0302feed3d159d51597b330e9`**에서 새 브랜치 **`ch3m`**을 만든다(2절).
새 worktree는 `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3m`이다.
**이 파일:** `docs/specs/CH3M_MODELS.md`로 commit한다.
**새 코드:** `analysis/ch3m/` · **결과:** `analysis/ch3m/results/` · **scratch:** `$CH3M=/gpfs/tmp_cpu2/yeseo1ee/ch3m`

**읽기만:**
- 코드와 commit된 결과: `analysis/ch3/`, `analysis/deploy/`, `analysis/d1_pair/`, `analysis/d1_ladder/`, `analysis/geo2/`
- `$CH3=/gpfs/tmp_cpu2/yeseo1ee/ch3`: 재생성한 lf 입력 `$CH3/lf`, ch3 단위 · 모델
- `$LAD=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder`, `$G2=/gpfs/tmp_cpu2/yeseo1ee/d1_g2`, `$D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair`

**기존 코드는 고치지 않는다.** import해서 부르고, 바꿀 동작은 `analysis/ch3m/`에 둔다.

**순서:** 1 시작 조건 → 2 브랜치 → 3 새 코드 → 4 XGB 입력 묶음(XP) → 5 코드 검사(C) → 6 사전 등록(R) →
7 교차검증 학습(K) → 8 평가(E) → 9 (교체할 때만) 최종 모델 · 추론기(W · I) → 10 요약(S) → 11 commit · push → **멈춤**
- 13절 STOP 조건에서만 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다.
- 모든 계산은 `sbatch`, `--time=48:00:00`(저장소 `CLAUDE.md`). 로그인 노드에서는 one-shot 제출과 git 외에 아무것도 하지 않는다.

---

## 0. 무엇을, 왜

### 0-1. 목표 (사용자 결정 2026-11-10)

- ch3(TF, 3채널 직접 예측)은 lf 합(TF, 8채널 예측을 더한 값)에 비해 사전 등록 채택 규칙을 넘지 못했다.
  - P_main para에서 strain +8.1 %, barrier +5.7 %였다(`analysis/ch3/results/SUMMARY_CH3.md`).
  - 그런데 비교한 모델은 TF 하나뿐이었다.
- 이번에는 **GNN과 XGB까지** 같은 입력 · 라벨 · 분할에서 "3채널 직접" 대 "10타깃 예측의 합"을 비교한다. 답할 질문은 둘이다.
  - **Q1** 3채널 직접 학습이 10타깃 합보다 나은가? 모델 종류마다 답한다.
  - **Q2** 배포 모델(현재: deploy lf TF 10타깃 합)을 바꿀 후보가 있는가? 사전 등록 규칙으로 정한다.
- ML 쪽 새로움은 노리지 않는다. 모델 구조 · 학습 규칙은 모두 기존 LADDER · DEPLOY 것을 그대로 쓴다.

### 0-2. 실험 칸 (2 × 3)

| 모델 | 10타깃 학습 → 3채널로 더함 | 3채널 직접 (strain · inter · solv · barrier) |
|---|---|---|
| TF | **새로** `D__lf10__S__TF__f{0..4}` | 기존 ch3 재사용: `analysis/ch3/results/K3_predictions.parquet` (`D__lf3__S__TF`) |
| GNN | **새로** `D__lf10__S__GNN__f{0..4}` | **새로** `D__lf3__S__GNN__f{0..4}` |
| XGB | **새로** `D__lf10__S__XGB__f{0..4}` | **새로** `D__lf3__S__XGB__f{0..4}` |

- 모든 새 단위는 ch3과 같은 **재생성 lf 입력** `$CH3/lf`를 쓴다. 그래서 6칸이 모두 같은 입력 위에 있다.
- **기준(현재 배포 모델) `dep`:** deploy commit의 `analysis/deploy/results/K_predictions.parquet`(lf TF 10타깃, 원래 입력)을 더한 값.
- **TF 10타깃을 다시 학습하는 이유:**
  - 6칸을 같은 입력으로 맞춘다.
  - 같은 모델을 다시 학습했을 때 생기는 차이(TF10 대 dep)를 **흔들림 기준**으로 쓴다. ch3 C-3′ 진단에서 이 흔들림이 수 %였다.
- **AW는 하지 않는다.** LADDER g2에서 이미 확실히 나빴다(0-3 표).

### 0-3. 이미 아는 것 (참고값, 판정 아님)

LADDER g2 · S 10타깃 교차검증 예측(`analysis/d1_ladder/results/L_predictions.parquet`)을 3채널로 더한 값이다.
P_main para 공통 1,658짝에서 계산했고, 입력은 g2(lf와 1e-6 수준 차이)다.

| 모델 | strain | inter | solv | barrier |
|---|---|---|---|---|
| g2 TF 10합 | 0.360 | 0.373 | 0.294 | 0.327 |
| g2 GNN 10합 | 0.362 | 0.368 | 0.288 | 0.325 |
| g2 AW 10합 | 0.427 | 0.404 | 0.317 | 0.385 |
| g2 XGB 10합 | 0.436 | **0.772** | 0.309 | 0.392 |
| dep (lf TF 10합) | 0.361 | 0.383 | 0.310 | 0.330 |
| ch3 (lf TF 3직접) | 0.390 | 0.382 | 0.306 | 0.349 |

para에서 본 채널 오차의 상쇄:

| | TF | GNN | XGB |
|---|---|---|---|
| corr(오차 pauli, 오차 elst + oi) | −0.99 | −0.99 | −0.96 |
| elst · pauli · oi 오차를 따로 더한 크기 → 합의 오차 | 3.64 → 0.34 | 3.74 → 0.34 | 4.02 → 0.72 |
| corr(오차 d1, 오차 d2) | −0.54 | −0.64 | −0.51 |

**사전 가설 H1:**
- 10타깃 합에서 채널 오차가 거의 완전히 상쇄되는 모델(TF · GNN)은 3채널 직접 학습의 이득이 없다.
- 상쇄가 덜 되는 모델(XGB: 타깃마다 따로 학습)은 inter에서 이득이 있다.

8-4절에서 설명용으로 확인한다.

### 0-4. 공통 설계

| 항목 | 값 |
|---|---|
| 라벨 | `$LAD/F_pairs_cr2.csv`의 `dd_<t>` 10개. 3채널 = `y10 @ M` (`analysis/ch3/ch3_common.M`: strain = d1 + d2, inter = elst + pauli + oi + disp + c_ghost, solv = cpcm + cds, barrier) |
| 입력 | lf · S arm. TF · GNN = 원자 토큰(d_in 29) + 전역 182열. XGB = 전역 182열만(LADDER · rev 9 arm S와 같음) |
| 분할 | FU2 T1 5 fold (`$D1PAIR_SCRATCH/fu2/splits/T1__f{k}.json`). 사용 가능 짝 P_lf 7,388, 시험 짝 7,279 (ch3 K3와 같음) |
| TF · GNN 학습 | DEPLOY 11-2 = FU2 M-1 / LADDER arm S. `Mo.build(model, cell, d_in=29, n_out, d_glob=182, "A+G")`. grid: TF 2/3층 × 64/128, **GNN 4/6층 × 64/128** (`fu2/models.py GRID`). AdamW 1e-3 / wd 1e-4, cosine 300 epoch, dropout 0.1, batch 64짝, patience 30, (a,b)/(b,a) 순서 무작위 증강. 튜닝은 inner_train에서 하고 칸 선택은 inner_val로 한다. final = 바깥 학습 짝, 튜닝 최적 epoch, seed 20261019 + 0..4 평균 |
| TF · GNN 타깃 모드 | 10모드 = deploy 값(항등 벌점 Σ₉, 검증 지표 = 주 9타깃 평균). 3모드 = ch3 값(벌점 Σ₃, 검증 지표 = 4타깃 평균). `analysis/ch3/data_ch3.use_targets`로 바꾼다 |
| XGB 학습 | LADDER `train_xgb.py` arm S 규칙 그대로 (= rev 9 arm C). `d1_train.Scale`, `_xgb_fit`(양방향), `_xgb_g`(반대칭), grid 12칸. 타깃마다 따로: inner GroupKFold(4, core) 평균 MAE로 칸을 고르고, 바깥 학습 짝 전부로 final fit. seed 없음(한 번 fit) |
| XGB 3모드 | 타깃 strain · inter · solv · barrier를 각각 따로 튜닝한다. barrier는 10모드와 같은 데이터 · 규칙이라 **같은 값이 나와야 한다**(C-XB) |
| 관문 | 모든 fit에서 C-4 반대칭(\|g(a,b)+g(b,a)\|, \|g(a,a)\| ≤ 1e-4 kcal/mol, 무작위 시험 짝 200개), C-5 누출 검사 |
| GPU | RTX 3090 (`-p gpu1 --gres=gpu:rtx3090:1`). deploy · LADDER · ch3과 같은 종류. 단위 지문에 GPU 이름을 적는다 |

---

## 1. 시작 조건 (하나라도 아니면 STOP)

| # | 조건 |
|---|---|
| 1-1 | `git fetch origin ch3` 후 `origin/ch3` = `6a4f59c26a3c4db0302feed3d159d51597b330e9` (아니면 그 SHA로 직접 fetch) |
| 1-2 | `$CH3/lf/X182_lf.parquet` sha256 = `cd5ec799f088a72aa31a2682240f0657469d07db5f5fc56050aa92765fb7f199`. 토큰 manifest(`analysis/ch3/results/lf_inputs/fu2_lf/T_manifest.csv`) sha256 = `d9c914380a235aae919d503bd5eed2034cfba5c20073c519476dfd0be5bacf72` (= ch3 `V_inputs.json` V-4 · V-5 값). `$CH3/lf/fu2_lf/tokens`의 토큰 파일 8,502개가 manifest와 맞음 |
| 1-3 | `analysis/ch3/results/K3_predictions.parquet` sha256 = `4faefc9c5e38c1a8e6b1592c1c3a476cc31c9edfa1d1fea9e8ce388cde0fa24e`. `analysis/deploy/results/K_predictions.parquet` sha256 = `a7c30339e46b5455972ba4bb93493d3f50f33db4b77d8bc6f3a83a5b53f58616` |
| 1-4 | `analysis/d1_ladder/results/L_predictions.parquet` sha256 = `1de11e77ea82571b139d6070ad556386e2a74b482af5c357498614bee37b3e4d` (= `L_report.json` predictions_sha256) |
| 1-5 | `$LAD/F_pairs_cr2.csv`, FU2 T1 분할 5개, `$G2/fu2/tokens`, `$G2/X182_d1_g2.parquet`, `$LAD/inputs/g2.parquet`이 있다. DeployData('g2')가 만들어진다(5절 C-R 회귀용) |
| 1-6 | 환경: `env_d1fu2`(`.d1fu2_env_ok`), `env_espley`. env_espley 버전이 LADDER XGB 기록(`L_xgb_report.json` versions)과 같다: xgboost 3.2.0, scikit-learn 1.3.1, numpy 1.26.0, pandas 2.1.1, python 3.11.16 |
| 1-7 | 디스크 여유 20 GB 이상 |
| 1-8 | (기록만, STOP 아님) 현재 배포 모델 파일 `/gpfs/tmp_cpu2/yeseo1ee/deploy/model/D__lf__S__TF__all__s{0..4}.pt`가 있는지. 11-09 정리에서 deploy scratch가 지워졌으면 없을 수 있다(9절에서 씀) |

**1-2 실패 시:** `$CH3/lf`가 지워졌거나 바뀐 경우다. STOP하지 않고 `docs/specs/CH3_MODEL.md` 3–4절(입력 재생성 L · M · F · T + V)을
같은 경로 `$CH3/lf`에 다시 실행한다. 그 뒤 새 sha256을 1-2 대신 기록하고 계속한다. 이 경우 ch3 K3(TF 3직접)는 다른 입력에서 학습된 것이므로 6칸을 같은 입력에 두기 위해 TF 3모드도 새로 학습한다(`D__lf3__S__TF__f{0..4}`를 `$CH3M/units`에).

결과: `analysis/ch3m/results/S0_start.json`.

---

## 2. 브랜치 · 폴더

```
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction          # 주 저장소 (실제 경로 확인)
git fetch origin ch3
git worktree add -b ch3m /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3m 6a4f59c26a3c4db0302feed3d159d51597b330e9
```
- 이 지시서를 `docs/specs/CH3M_MODELS.md`로 commit한다.
- 폴더: `sbatch --time=48:00:00 --wrap "mkdir -p $CH3M/{logs,units,smoke,cr,xgb_pack,model}"`.
- `analysis/ch3/results/lf_inputs/`(재생성 입력의 보고서)는 ch3 commit에 있으므로 이 worktree에서 그대로 읽는다.

---

## 3. 새 코드 (`analysis/ch3m/`)

| 파일 | 내용 |
|---|---|
| `ch3m_common.py` | 경로(`CH3M`, `RES = analysis/ch3m/results`), 쓰기 가드(`$CH3M`과 `analysis/ch3m/results` 밖이면 STOP), 단위 이름, 모델 라벨(`dep`, `TF10`, `TF3`, `GNN10`, `GNN3`, `XGB10`, `XGB3`), 원자적 쓰기, sha256, STOP(exit 3) · not ready(exit 2) |
| `train_m.py` | TF · GNN. `analysis/ch3/train_ch3.py`의 `run_unit` · `collect` · `final`을 **모델 이름을 인자로** 받게 옮긴 사본(GRID = `Mo.GRID[model]`, `Mo.build(model, …)`, `FT.fit(D, model, …)`). 데이터는 `analysis/ch3/data_ch3.py`(`get_data('lf')` = 10모드, `get_data('ch3')` = 3모드, `use_targets`)를 import해서 그대로 쓴다. 명령: `list`, `smoke`, `cr <TF\|GNN>`, `run <TF\|GNN> <10\|3> <i\|@I> <n>`, `collect`, `final <후보>` |
| `xgb_pack.py` | env_d1fu2, CPU. 4절 |
| `xgb_m.py` | env_espley, CPU. LADDER `train_xgb.py`의 `run_unit`을 **타깃 목록을 인자로** 받게 옮긴 사본이다. `prep_fit` · `leak_check` · `run_parallel` · `_task` · `gate_eval`은 `analysis/d1_ladder/train_xgb.py`에서 import해서 그대로 부른다. fit 전에 `train_xgb.TARGETS` · `NT` · `H_IDX`를 그 실행의 타깃 목록(arm S이므로 `H_IDX = []`)으로 바꾸는 `use_targets`를 둔다(data_ch3.use_targets와 같은 방식). inner 4 fold는 `XData.split`의 규칙을 그대로 옮긴다: 거르기 전 바깥 학습 목록에서 core로 GroupKFold(4) → 사용 가능 짝만 남김 → fold 0 = 분할 파일 inner_train / inner_val 단언. 명령: `list`, `smoke`, `cr`, `run <10\|3> <i\|@I> <n>`, `collect` |
| `eval_m.py` | env_espley. 8절. `analysis/ch3/eval_ch3.py`의 정의 · 함수(`boot_rel`, `compare`, `metrics`, `rank_table`, `cause`, `identity_residual`, `ad_thresholds`)를 import해서 7개 모델에 같은 코드로 적용한다 |
| `prereg_m.py` | 6절 사전 등록 쓰기 · 확인 (`analysis/ch3/prereg_ch3.py`와 같은 규칙) |
| `tests_m.py` | 5절 코드 검사 |
| `predict_m.py` | 9절에서만. `analysis/ch3/predict_ch3.py`와 같은 인터페이스이고 `.pt`의 `model`로 TF · GNN을 만든다 |
| `jobs/gpu_m.slurm` | `analysis/ch3/jobs/gpu_ch3.slurm` 복사. 로그 `$CH3M/logs`로 바꾸고, `DEP_SCRATCH=$CH3/lf`, `DEP_RESULTS=<worktree>/analysis/ch3/results/lf_inputs`는 그대로 둔다 |
| `jobs/cpu_m.slurm` | `analysis/ch3/jobs/cpu_ch3.slurm` 복사. `CH3MENV=d1fu2 \| espley`. XGB는 `--cpus-per-task=32`, `LAD_XGB_NJOBS=$SLURM_CPUS_PER_TASK` (worker 수는 결과를 바꾸지 않음, train_xgb `run_parallel`) |

---

## 4. Phase XP — XGB 입력 묶음 (`xgb_pack.py`, env_d1fu2 CPU)

XGB는 torch 없는 env_espley에서 돈다. DeployData(torch import)를 거기서 만들 수 없으므로, TF · GNN과 **같은 짝 · 분할 · 라벨**을
env_d1fu2에서 꺼내 묶음 파일로 넘긴다.

| 묶음 | 원천 | 쓰임 |
|---|---|---|
| `$CH3M/xgb_pack/lf.npz` + `lf.json` | `DeployData('lf')` (`data_ch3.get_data('lf')`) | K (XGB10 · XGB3) |
| `$CH3M/xgb_pack/g2.npz` + `g2.json` | `DeployData('g2')` (LADDER 사용 가능 규칙, deploy C-2에서 `LadData('g2','S')`와 같음이 확인됨) | C-RX 회귀 |

**내용:**
- 짝 표 행 순서: pair_id, ref_kind, pos, a_rxn, b_rxn, core(`S_folds.csv`), y10, y4 = y10 @ M
- 토큰 행 순서의 rxn_id, ra, rb, usable_pair
- 5 fold의 분할 파일 원래 id 목록(train · test · inner_train · inner_val)
- **X = 원천의 182열 parquet 값**(토큰 행 순서, 열 = `T_report glob_cols`). 토큰 glob이 아니라 LADDER XData와 같은 표 값이다.

**json:**
- 원천 파일 sha256(토큰 manifest, X182, 라벨, 분할)
- `pair_list_sha`. lf는 `7de6e33ac550656a26c6b2cbfdf63069cfe3ab996d4a65a9811e39a15116d146`(= ch3 V-6)이어야 한다.
- 짝 수
- npz sha256

**확인 (다르면 STOP):**
- y10이 라벨 파일 `dd_<t>`와 비트 단위로 같다.
- 분할 목록에 사용 가능 규칙을 적용한 결과가 `DeployData.split(k)`의 train · test · inner_train · inner_val과 같다.
- X 행이 토큰 glob과 1e-12(상대) 안에서 같다.

---

## 5. Phase C — 코드 검사 (`tests_m.py`) — 하나라도 실패하면 STOP

회귀 검사는 **LADDER가 학습에 쓴 g2 입력 그대로**에서 돌린다. 입력이 비트 단위로 같으므로 결과도 같아야 한다.
deploy C-3과 ch3 진단 D1에서 TF는 RTX 3090에서 튜닝 검증값까지 그대로 재현됐다. ch3 C-3′이 실패한 이유는 입력 재생성의 1e-7 차이였고, 여기서는 그 차이가 없다.

| # | 검사 | 기준 |
|---|---|---|
| C-RT | `train_m.py cr TF`: 10모드, `DeployData('g2')`, fold 0 전체 규칙(4칸 + seed 5개), RTX 3090 → LADDER `L__g2__S__TF__f0` (`L_report.json` units: 칸 3 × 128, best_epoch 70, val_kcal_mae; `L_predictions.parquet` g2 · S · TF fold 0) | 같은 칸, 같은 best epoch, val_kcal_mae 차 ≤ 1e-6, 시험 yhat 최대 차 ≤ 1e-4 |
| C-RG | `train_m.py cr GNN`: 같은 조건 → `L__g2__S__GNN__f0` (칸 6 × 128, best_epoch 47) | 같은 기준 |
| C-RX | `xgb_m.py cr`: g2 묶음, 10모드, fold 0 전체 grid → `L__g2__S__XGB__f0` (`L_xgb_report.json` units chosen; `L_predictions.parquet` g2 · S · XGB fold 0) | 타깃마다 같은 칸, yhat 최대 차 ≤ 1e-6 |
| C-T | 3모드 배선: GNN 출력 차원 4, 항등 벌점 `[0,1,2]` 합 대 `3`, 검증 지표 4타깃 평균, `.pt`의 `model = GNN` · `TARGETS = TARGETS3`, grid = `Mo.GRID["GNN"]`. XGB는 `train_xgb.TARGETS` · `NT`가 실행 타깃과 같고, 단위 기록의 타깃 · 예측 열이 4개 | 모두 assert |
| C-D | 데이터 동일성: train_m의 lf 데이터 · 분할 = ch3 `Ch3Data` / `DeployData('lf')`, `xgb_pack/lf` 짝 · 분할 = 같은 목록 | 배열 · 목록 sha256 같음 |
| C-4 · C-5 | 모든 fit | 0-4 관문 |
| C-8 | smoke: GNN 10 · 3모드 fold 0, 칸 최소 · 최대, 2 epoch, seed 1개. XGB 3모드 fold 0, 칸 첫 · 끝 | 끝까지 돌고 C-4 · C-T 통과. 산출은 `$CH3M/smoke`(K가 읽지 않음) |

**C-RG만 실패할 때 (C-RT는 통과):**
- GNN의 GPU 비결정성일 수 있으므로 같은 명령을 한 번 더 돌린다(C-RG2, `$CH3M/cr/rg2`).
- C-RG2 = C-RG(yhat 최대 차 ≤ 1e-4)인데 LADDER와 다르면 → 코드 차이 → **STOP**.
- C-RG2 ≠ C-RG이면 → GNN 학습 자체가 비결정적이다. 차이 수치를 `C_deviation.json`에 기록하고 진행한다(STOP 아님). 이 경우 GNN 결과의 흔들림이 TF보다 클 수 있다고 요약에 적는다.

결과: `results/C_tests.json`(`all_pass`), `CR_tf.json`, `CR_gnn.json`, `CR_xgb.json`, `C8_smoke.json`.

---

## 6. Phase R — 사전 등록 (`results/PREREG_CH3M.md` + `prereg_ch3m.json`)

K 학습 전에 commit한다. `train_m.py run` · `xgb_m.py run` · `final`은 두 파일이 git에 있고, HEAD 이후 바뀌지 않았고,
json의 `md_sha256` = md의 sha256일 때만 돈다.

**md에 적을 것:**
- 0-2 실험 칸, 0-4 설계, 0-3 참고값과 가설 H1
- 8절 지표 정의 전부, 8-2 해석 규칙, 8-3 교체 규칙
- 기준 파일(deploy `K_predictions`, ch3 `K3_predictions`), 라벨 · 분할 · 입력 · XGB 묶음의 sha256

---

## 7. Phase K — 교차검증 학습

| 단위 | 장비 | 명령 | 예상 |
|---|---|---|---|
| `D__lf10__S__TF__f{0..4}` | RTX 3090 | `train_m.py run TF 10 @I 5` | 약 1 GPU-h |
| `D__lf10__S__GNN__f{0..4}` | RTX 3090 | `train_m.py run GNN 10 @I 5` | 약 4–5 GPU-h (LADDER g2 GNN 4.2) |
| `D__lf3__S__GNN__f{0..4}` | RTX 3090 | `train_m.py run GNN 3 @I 5` | 약 4–5 GPU-h |
| `D__lf10__S__XGB__f{0..4}` | CPU 32코어 | `xgb_m.py run 10 @I 5` | 약 30 CPU-h (LADDER g2 XGB 29) |
| `D__lf3__S__XGB__f{0..4}` | CPU 32코어 | `xgb_m.py run 3 @I 5` | 약 12 CPU-h |

- 산출: `$CH3M/units/<unit>.{json,parquet}`(TF · GNN은 `__s{0..4}.pt`도). 단위 json이 있으면 지문(라벨 · 입력 · 분할 · 코드 sha256, GPU 이름)을 비교하고 건너뛴다. 다르면 STOP.
- 배열은 `--array=0-4%5`. 대기 20 상한(배열 원소마다 1개)을 넘지 않게 GPU 세 묶음을 `--dependency=afterany`로 잇는다.
- **collect:**
  - `train_m.py collect`와 `xgb_m.py collect` → `results/KM_predictions.parquet`.
    - 열은 deploy `K_predictions`와 같다(`pair_id, ref_kind, pos, fold, rung, arm, model, target, y, yhat, base, beta, yhat_seed_sd`).
    - `rung ∈ {lf10, lf3}`, `model ∈ {TF, GNN, XGB}`.
  - 함께 `KM_val.csv`, `KM_report.json`(단위별 칸 · epoch · 시간, 합계 GPU-h · CPU-h)을 쓴다.
- **C-XB (collect에서):** `D__lf3__S__XGB`의 barrier 예측 = `D__lf10__S__XGB`의 barrier 예측(최대 차 ≤ 1e-9, 같은 칸). 다르면 STOP.

---

## 8. Phase E — 평가 (`eval_m.py run`)

### 8-1. 공통

**모델 7개 (3채널 예측):**
- 10타깃 모델(`dep`, `TF10`, `GNN10`, `XGB10`)은 yhat을 `M`으로 더한다. barrier는 직접 예측을 쓴다.
- 3채널 모델(`TF3` = ch3 K3, `GNN3`, `XGB3`)은 그대로 쓴다.

**정답 확인:** 정답 y는 모두 `F_pairs_cr2.csv @ M`이다. 각 예측 파일의 y가 이 값과 같은지 먼저 확인한다(다르면 STOP).

**나머지 정의는 `eval_ch3.py`와 같다:**
- 짝 속성(`S_folds.csv` core · fam · in_P_main · in_P_sens, 라벨 파일 dipole · cfg · series · pos · Xb)
- 집합 P_main = reference & in_P_main
- **7개 모델이 모두 예측한 공통 짝**만 쓴다.
- 부트스트랩은 core 단위 2,000회, seed 20261019 (`analysis/geo2/eval_geo2.py boot_diff`)

**주 집합 = P_main · para 공통 짝.** 0-3과 같으면 1,658짝이다.

### 8-2. Q1 — 3채널 직접 대 10타깃 합, 모델마다 → `EM_3v10.csv`

TF · GNN · XGB 각각, 타깃 4개(strain · inter · solv · barrier)마다 다음을 계산한다.
- 상대 차 = (MAE(x3) − MAE(x10합)) / MAE(x10합)과 95 % CI
- 대상은 P_main para(주)와 P_main all(보조)이다.

**흔들림 기준 noise_t:**
- noise_t = \|상대 차(TF10 대 dep)\|. 같은 모델을 다시 학습했을 때의 차이다.
- `EM_noise.csv`에 TF10 대 dep의 상대 차 · CI도 적는다.

**해석 규칙 (사전 등록, 판정용 아님):**

| 표시 | 조건 |
|---|---|
| 3 나음 | 95 % 상한 < 0 이고 점추정 < −max(noise_t, 2 %) |
| 10합 나음 | 95 % 하한 > 0 이고 점추정 > +max(noise_t, 2 %) |
| 같음 | 그 밖 |

### 8-3. Q2 — 배포 모델 교체 규칙 → `EM_choice.csv`

**후보:** `TF3`, `GNN10`, `GNN3`, `XGB10`, `XGB3`. `TF10`은 dep과 같은 모델이라 후보가 아니다.
**기준:** `dep`.
**지표:** P_main para 공통 짝에서 타깃 4개마다 상대 차 r_t = (MAE_후보 − MAE_dep) / MAE_dep, 95 % CI.

**교체 조건 (둘 다 만족):**
- (a) 4타깃 모두 95 % 상한 ≤ +5 %
- (b) 4타깃 중 2개 이상에서 점추정 ≤ −5 % **그리고** 95 % 상한 < 0

**판정:**
- 여러 후보가 통과하면 r_t 4개의 평균이 가장 낮은 후보를 고른다. 동률이면 TF3 → GNN10 → GNN3 → XGB10 → XGB3 순서로 고른다.
- 아무도 통과하지 못하면 **배포 모델은 그대로**(deploy lf TF 10합)다. STOP하지 않는다.
- −5 %의 근거: 같은 deploy 모델을 seed만 바꿔 다시 학습해도 barrier MAE가 +3.6 % 달라졌다(deploy `NULL_gc.json`). 그보다 큰 개선만 교체 사유로 본다.

### 8-4. 오차 상쇄 진단 (H1, 설명용) → `EM_cancel.csv`

10타깃 모델 4개(`dep`, `TF10`, `GNN10`, `XGB10`)에 대해 P_main para · all에서 계산한다.
- corr(오차 pauli, 오차 elst + oi), corr(오차 d1, 오차 d2), corr(오차 cpcm, 오차 cds)
- 묶음마다 상쇄 비율 = 평균 \|Σ 오차\| / 평균 Σ \|오차\| (낮을수록 많이 상쇄됨)
- 8-2의 상대 차를 함께 적는다.

H1에 맞는지는 표로만 보고한다(모델 3개라 검정하지 않는다).

### 8-5. 보조 지표 (7개 모델 모두, 판정 아님)

`eval_ch3.py` 정의 그대로 쓴다(P_sens 기준 공통 짝, 예측 자신의 barrier 부호).

| 산출 | 내용 |
|---|---|
| `EM_metrics.csv` | 타깃 × 위치: MAE, RMSE, r, 부호 정답률(\|y\| ≥ 1) |
| `EM_rank.csv` | 계열 안 장벽 순위: 두 작용기 순서 정답률(0.41–1 / 1–2 / > 2), 실제 최저가 예측 1등 · 상위 3, Spearman 중앙값 · 25 % |
| `EM_cause.csv` | 주도 채널 · 유형(S / SI / I) 일치율. 반응 하나와 칸 평균, 기준값 함께. 칸 평균 r · MAE |
| `EM_ad.json` | 적용 범위 문턱(시험 짝 `yhat_seed_sd` 95 백분위). XGB는 seed가 하나라 기록하지 않는다 |
| `EM_identity.csv` | \|ŷ_barrier − (ŷ_strain + ŷ_inter + ŷ_solv)\| 평균 · p95 |

### 8-6. selftest (`eval_m.py selftest`, K 전에 실행)

- (i) `TF3` 대 `dep`이 ch3 `E3_compare.csv`의 P_main para 4줄을 소수 넷째 자리까지 재현한다.
- (ii) LADDER `L_predictions.parquet` g2 · S의 TF · GNN · XGB · AW를 같은 코드로 더하면 0-3 표를 소수 셋째 자리까지 재현한다.
- 다르면 정의 차이를 보고한다(STOP 아님).

산출: `results/EM_selftest.json`, `EM_report.json`(8-2 표시, 8-3 판정, 요약, 입력 sha256).

---

## 9. Phase W · I — 최종 모델과 추론기 (8-3에서 교체될 때만)

| 고른 후보 | 할 일 |
|---|---|
| 없음 | 배포 모델 = lf TF 10합 그대로. 1-8에서 deploy 최종 모델 파일이 있으면 그것을 쓰고 W · I는 하지 않는다. **없으면** 같은 모델을 재생성 입력에서 다시 만든다: `train_m.py final TF10`(아래 GNN 행과 같은 DEPLOY 13 규칙, `D__lf10__S__TF__all__s{0..4}.pt`) → I-1 · I-2. 약 0.15 GPU-h |
| `TF3` | ch3 최종 모델(`$CH3/model/D__lf3__S__TF__all__s{0..4}.pt`, `analysis/ch3/results/MODEL_CARD_ch3.json`, `I3_selftest.json`)이 이미 있다. 기록만 한다 |
| `GNN10` · `GNN3` | `train_m.py final <후보>`: DEPLOY 13 규칙(학습 짝 `split("all")`, 칸 = fold 선택 최빈값(동률: 튜닝 검증 평균 낮은 칸 → grid 순서), epoch = 최적 epoch 중앙값, seed 20261019 + 0..4, C-4 · C-T) → `$CH3M/model/D__<rung>__S__GNN__all__s{0..4}.pt`, `MODEL_CARD.json`(사본 `results/MODEL_CARD_ch3m.json`), `results/WM_report.json`. 그다음 `predict_m.py` I-1(`--model cv`로 KM 시험 짝 재예측 ≤ 1e-4) · I-2(`--model final`로 학습 짝 1,000개, 같은 GPU에서 ≤ 1e-6) → `results/IM_selftest.json` |
| `XGB10` · `XGB3` | **STOP하고 보고**한다. XGB 최종 모델 저장 · 추론기는 아직 없어서 별도 명령서가 필요하다. 0-3으로 보아 일어날 가능성은 낮다 |

---

## 10. Phase S — 요약 `results/SUMMARY_CH3M.md` (한국어)

포함할 것:
1. 한 줄 결론: Q1(모델마다 3 대 10합)과 Q2(배포 모델 교체 여부)
2. C 검사 결과(C-RT · C-RG · C-RX 수치, 편차가 있으면 그 내용)
3. **2 × 3 표:** P_main para MAE(strain · inter · solv · barrier)와 dep 행, 8-2 상대 차 · CI · 표시
4. 흔들림 기준(TF10 대 dep)
5. 8-4 오차 상쇄 표와 H1 해석
6. 8-3 교체 판정 표
7. 보조 지표 요약(순위 1등 · 상위 3, 원인 일치율 칸 평균)
8. 단위별 칸 · epoch, 실제 GPU-h · CPU-h
9. 편차 목록

---

## 11. commit · push

| 무엇 | commit 여부 |
|---|---|
| `docs/specs/CH3M_MODELS.md`, `analysis/ch3m/` 코드 · job 스크립트 | commit |
| `analysis/ch3m/results/` (json · csv · md, `KM_predictions.parquet`, 9절 모델 카드) | commit. 파일당 50 MB가 넘으면 commit하지 않고 요약에 경로만 적음 |
| `$CH3M/` 안의 것 (XGB 묶음, 단위, `.pt`, 로그) | **commit하지 않음** (`CLAUDE.md`: `/gpfs/tmp_cpu2/...` 출력 금지) |

- 단계마다 commit한다(S0 · XP · C · PREREG · K · E · (W · I) · S). PREREG commit은 K 제출 전에 끝나야 한다.
- 마지막에 `git push -u origin ch3m`를 하고, push된 HEAD SHA를 요약 끝에 적는다.

---

## 12. 예상 비용

| 단계 | 예상 | 근거 |
|---|---|---|
| XP | < 0.5 core-h | DeployData 로딩 2회 |
| C-RT / C-RG | 약 0.2 / 1.1 GPU-h | LADDER g2 TF · GNN fold 0 (`L_report.json` wall) |
| C-RX | 약 7 CPU-h | LADDER g2 XGB fold 0 (`L_xgb_report.json` cpu_seconds 25,397) |
| C-8 | < 0.2 GPU-h, < 1 CPU-h | |
| K | 약 10 GPU-h, 약 42 CPU-h | 7절 |
| E | < 1 CPU-h | |
| W · I (GNN일 때만) | 약 1 GPU-h | |

합계는 약 12 GPU-h(RTX 3090), 약 50 CPU-h다.

---

## 13. STOP 조건

| 단계 | 조건 |
|---|---|
| 1 | 시작 조건 실패(1-2는 재생성으로 대체, 1절) |
| 4 | XP 확인 실패 |
| 5 | C 검사 실패(C-RG는 5절 예외 규칙대로) |
| 6 | PREREG가 git에 없거나 바뀜 |
| 7 | 지문이 다른 캐시, C-4 · C-5 · C-T 실패, C-XB 실패 |
| 8 | 예측 파일의 y가 `F_pairs_cr2.csv @ M`과 다름 |
| 9 | XGB 후보가 교체 규칙을 통과함. I-1 · I-2 실패 |

8-3 교체 규칙을 아무도 못 넘는 것은 STOP이 아니다(보고만 한다).

---

## 14. 실행 규칙 (저장소 `CLAUDE.md`)

- `sbatch`만 쓰고 `--time=48:00:00`. 로그인 노드에 상주 프로세스를 두지 않는다(nohup, 셸 반복문, 감시 프로세스 금지).
- 실행 10 / 대기 20 상한(배열 원소마다 1개). 제출 전 `squeue -u $USER -h -r | wc -l`.
- `--dependency=afterany` 체인: S0 → XP → C(CPU: C-D, C-RX, XGB smoke) · C(GPU: C-RT, C-RG, GNN smoke) → C 병합 → PREREG commit
  → K(GPU 세 묶음 · CPU 두 묶음) → collect → E → (W · I) → S.
- CPU는 `sinfo -p cpu1,cpu2`에서 빈 쪽을 쓴다. GPU는 RTX 3090(gpu1)만 쓴다. 비어 있지 않으면 기다린다(A10으로 옮기지 않는다).
- 모든 단계는 멱등이고 원자적으로 쓴다. 48 h에 잘리면 같은 스크립트를 다시 제출한다.

| 단계 | 환경 |
|---|---|
| S0, XP, C-D, collect(TF · GNN) | `env_d1fu2` (CPU) |
| C-RT, C-RG, C-8(GNN), K(TF · GNN), W, I | `env_d1fu2` (GPU, RTX 3090) |
| C-RX, C-8(XGB), K(XGB), collect(XGB), E | `env_espley` (CPU) |

---

## 15. 범위 밖

- AW 모델(0-2)
- seed만 바꾼 반복 학습(흔들림은 TF10 대 dep 하나로만 본다)
- 새 특성 · 새 모델 구조 · grid 확장
- 14만 반응 배포 실행, 새 칸 표본 DFT, 숨은 꺾임 분석(D1 DFT로만)
