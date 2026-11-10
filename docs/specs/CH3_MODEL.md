# CH3-MODEL — 3채널(변형·상호작용·용매) 예측 모델: D1 라벨 · lf 입력으로 만들고 학습하고 결과 push — Claude Code 실행 지시서

**용어:** 이 문서에서 **D1** = 기존 D1 라벨이다(재라벨링 안 함). 라벨 파일은 배포 모델 lf와 같은 판인
`$LAD/F_pairs_cr2.csv`(기준 conformer를 통일한 cr2)를 쓴다.

**저장소:** GitHub에서 지워진 `deploy` 브랜치의 마지막 commit **`2f9ffa35511ca59dc74f02744edc2a9d5ff31afd`**에서
새 브랜치 **`ch3`**를 만든다(2절). 새 worktree `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3`.
**이 파일:** `docs/specs/CH3_MODEL.md`로 commit한다.
**새 코드:** `analysis/ch3/` · **결과:** `analysis/ch3/results/` · **scratch:** `$CH3=/gpfs/tmp_cpu2/yeseo1ee/ch3`
**읽기만:** `analysis/deploy/`, `analysis/d1_pair/`, `analysis/d1_ladder/`, `analysis/geo2/` 코드와 commit된 결과,
`$G2=/gpfs/tmp_cpu2/yeseo1ee/d1_g2`, `$LAD=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder`, `$D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair`,
D1 run(`/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run`). **기존 코드는 고치지 않는다.** import해서 부르고, 바꿀 동작은 `analysis/ch3/`에 둔다.

**순서:** 1 시작 조건 → 2 브랜치 → 3 입력 재생성(L·M·F·T) → 4 재생성 확인(V) → 5 새 코드 → 6 코드 검사(C) →
7 사전 등록(R) → 8 교차검증 학습(K) → 9 평가(E) → 10 최종 모델(W) → 11 추론기 검사(I) → 12 요약(S) → 13 commit · push → **멈춤**
- 15절 STOP 조건에서만 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다.
- 모든 계산은 `sbatch`, `--time=48:00:00`(저장소 `CLAUDE.md`). 로그인 노드에서는 one-shot 제출과 git 외에 아무것도 하지 않는다.

---

## 0. 무엇을, 왜

### 0-1. 목표 (사용자 결정 2026-11-10)

8채널 대신 **3채널**을 직접 예측하는 배포용 모델을 만든다. ML 쪽 새로움은 노리지 않는다. 입력 · 모델 구조 · 학습 규칙 ·
분할은 배포 모델 lf(`docs/specs/DEPLOY_MODEL.md`, deploy commit `2f9ffa3`)와 같고, **타깃만 바꾼다**.
그리고 "8채널 모델 예측을 더해서 만든 3채널(lf 합)"보다 나은지 같은 짝에서 비교한다.

### 0-2. 확정 설계

| 항목 | 값 |
|---|---|
| 타깃 (4개) | `strain` = d1 + d2 · `inter` = elst + pauli + oi + disp + c_ghost · `solv` = cpcm + cds · `barrier` |
| 쉬운 말 | 비트는 비용 · 끌고 미는 힘 · 물의 효과 · 장벽(검산과 순위용) |
| 라벨 | `$LAD/F_pairs_cr2.csv`의 `dd_<t>`를 위 식으로 더한 값. strain + inter + solv − barrier 잔차 최대 ≈ 0.1 kcal/mol (라벨 자체의 수치 잡음) |
| 입력 | lf · S arm: 원자 토큰(d_in 29) + 전역 182열. DFT 값 없음. deploy 코드로 다시 만든다(3절) |
| 모델 | `Mo.build("TF", cell, d_in=29, n_out=4, d_glob=182, "A+G")` |
| 학습 규칙 | DEPLOY 11-2와 같음. 바뀌는 곳 3개뿐: 출력 10 → 4, 항등 벌점 Σ₉ → Σ₃ (0.1 × 평균 \|ŷ_barrier − (ŷ_strain + ŷ_inter + ŷ_solv)\| / s_barrier), 튜닝 검증 지표 = 4 타깃 표준화 MAE 평균 |
| 분할 | FU2 T1 5 fold (DEPLOY와 같은 파일) |
| 비교 기준 | **lf 합**: deploy commit의 `analysis/deploy/results/K_predictions.parquet`(lf 8채널 교차검증 예측)을 위 식으로 더한 값. barrier는 lf 직접 예측 |
| 최종 모델 | 사용 가능한 D1 짝 전부, 칸 = fold 선택 최빈값, epoch = 최적 epoch 중앙값, seed 20261019 + 0..4 (DEPLOY 13과 같은 규칙) |
| 단위 이름 | 교차검증 `D__lf3__S__TF__f{0..4}`, 최종 `D__lf3__S__TF__all__s{0..4}` |

### 0-3. 입력을 다시 만드는 이유

11-09 정리(R2-DEL)에서 deploy scratch(`/gpfs/tmp_cpu2/yeseo1ee/deploy`: lf 토큰 · 182열 · 메타데이터)와 deploy 브랜치가
지워졌다. 코드는 GitHub에서 commit SHA로 아직 받을 수 있다(2절). 입력은 같은 코드로 `$CH3/lf`에 다시 만들고, deploy commit의
보고서에 적힌 sha256과 맞춰 같은 입력인지 확인한다(4절). 비용은 deploy 실측으로 약 28 core-h다.

---

## 1. 시작 조건 (하나라도 아니면 STOP)

| # | 조건 |
|---|---|
| 1-1 | `git fetch origin 2f9ffa35511ca59dc74f02744edc2a9d5ff31afd` 성공. 그 commit에 `analysis/deploy/`, `analysis/d1_pair/fu2/`, `analysis/d1_ladder/`, `analysis/geo2/eval_geo2.py`, `analysis/deploy/results/K_predictions.parquet`가 있다 |
| 1-2 | `K_predictions.parquet`의 sha256 = 같은 commit `results/K_report.json`의 `predictions_sha256` |
| 1-3 | `$G2/jobs/<job_id>/`가 D1 대상 9,266개 모두에 있고, `analysis/d1_g2/results/G_status.csv`의 ok 8,616개는 `ts.xyz`, `ts_result.json`, `ref_dipole.xyz`, `ref_dipolarophile.xyz`를 가진다 (DEPLOY 0-2) |
| 1-4 | `$G2/g2_final/`, `$G2/X182_d1_g2.parquet`, `$G2/fu2/tokens`가 있다 (deploy의 L-2 · L-3 · L-4 관문이 읽는다) |
| 1-5 | `$LAD/F_pairs_cr2.csv`가 있고, 그 sha256이 deploy `results/K_report.json`의 `inputs_sha256`에 기록된 라벨 sha256과 같다 (키 이름은 파일에서 확인) |
| 1-6 | FU2 T1 분할 `$D1PAIR_SCRATCH/fu2/splits/T1__f{0..4}.json`의 sha256 = `analysis/d1_pair/results/fu2/S_splits.json` |
| 1-7 | 환경: D1 run env(`d1_run/nurion/env.sh`), `analysis/d1_pair/jobs/g.slurm`의 `PYENV=venv5`, `env_d1fu2`(`.d1fu2_env_ok`), `env_espley` |
| 1-8 | D1 manifest sha256 = `g2_common.MANIFEST_SHA`. 디스크 여유 30 GB 이상 |

결과: `analysis/ch3/results/S0_start.json` (항목별 통과 여부, 경로, sha256).

---

## 2. 브랜치 · 폴더

```
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction          # 주 저장소 (실제 경로 확인)
git fetch origin 2f9ffa35511ca59dc74f02744edc2a9d5ff31afd
git worktree add -b ch3 /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3 2f9ffa35511ca59dc74f02744edc2a9d5ff31afd
```
- 이 지시서를 `docs/specs/CH3_MODEL.md`로 commit한다.
- 폴더: `sbatch --time=48:00:00 --wrap "mkdir -p $CH3/{logs,lf,units,model,smoke,c3r}"`.
- deploy job 스크립트의 `#SBATCH --output`은 지워진 `/gpfs/tmp_cpu2/yeseo1ee/deploy/logs`를 가리킨다. 제출할 때 항상
  `--output=$CH3/logs/%x_%j.out`(배열은 `%x_%A_%a.out`)으로 덮어쓴다.

---

## 3. 입력 재생성 — deploy 코드 그대로 (Phase L · M · F · T)

deploy 코드를 **수정 없이** 환경 변수로 경로만 바꿔 실행한다. `analysis/deploy/`에서 제출한다.

| 변수 | 값 |
|---|---|
| `DEP_SCRATCH` | `$CH3/lf` |
| `DEP_RESULTS` | `<worktree>/analysis/ch3/results/lf_inputs` |
| 나머지 | deploy 기본값 (`DEP_G2`, `DEP_LAD`, `DEP_D1_RUN`) |

`--export=ALL,DEP_SCRATCH=$CH3/lf,DEP_RESULTS=<위 경로>`로 넘긴다. `$CH3/lf/prof_empty`는 빈 폴더로 둔다(DEPLOY 1절).

| 단계 | 명령 (DEPLOY 3–6절 그대로) | 환경 · 스크립트 | 관문 (DEPLOY와 같음) |
|---|---|---|---|
| L | `finalize_lf.py run @I 8 <cores>` (배열 0-7%8) → `finalize_lf.py report` | `jobs/cpu_d1.slurm` | L-1 · L-2 · L-4 실패 → STOP |
| M | `meta_lf.py run` | `jobs/cpu_d1.slurm` | 전하 ≠ 0 → STOP |
| F | `feat_lf.py slice @I 2` (배열 0-1) → `feat_lf.py merge` | `analysis/d1_pair/jobs/g.slurm`, `PYENV=venv5` | F-1 · F-3 실패 → STOP |
| T | `tokens_lf.py atoms @I <n>` → `tokens_lf.py gate` → `tokens_lf.py tokens` | 같은 환경 | 원소 · 열 목록, Q-a, T-a 실패 → STOP |

`--dependency=afterany` 체인으로 L → L report → M → F → F merge → T → T gate · tokens 순서로 잇는다.
Q(parity)는 하지 않는다. 4절 V가 deploy 입력과 같은지를 직접 확인한다.

---

## 4. Phase V — 재생성 확인 (`analysis/ch3/verify_inputs.py`)

deploy commit의 보고서에 적힌 값과 비교한다.

| # | 대상 | 기준값 (deploy commit) | 통과 |
|---|---|---|---|
| V-1 | `.done` 반응 수, `labels_lf.json` 레코드 수 | 8,616 (`L_report`, `M_report`) | 같음 |
| V-2 | `labels_lf.json` sha256 | `e7fdf9b57371f7c8e2851cd7fef4ef131604b6257047bd5e75f55ee326b5f720` (`F_report.labels_sha256`) | 같음, 또는 아래 내용 비교 통과 |
| V-3 | `input_meta_lf.csv` sha256 | `104ab323de72fb7e4a999744e5957914e3d4dfedefbeddafa8c1bf4941451a09` (`F_report.meta_sha256`) | 같음, 또는 내용 비교 통과 |
| V-4 | `X182_lf.parquet` sha256, 특성 성공 반응 수 | `8917a728c231a48baff5bafe697dc88f745e9c0fc8d807690805c47dbf42bd18` (`fu2_lf/T_report.x182_sha256`), 8,504 (`F_report.n_feat`) | 같음, 또는 내용 비교 통과 |
| V-5 | 토큰 manifest sha256, 토큰 반응 수 | `72e0523c2ac14f2312431f86109191858de314eed875f74486d6eefeb5db2b92` (`T_report.manifest_sha256`), 8,502 | 같음, 또는 내용 비교 통과 |
| V-6 | `DeployData('lf')`의 사용 가능 짝 목록 sha256 | deploy `K_report.pair_list_sha` | 같음 |

**내용 비교** (sha256이 다를 때만): 행 목록이 같고, 182열의 최대 \|차\| ≤ 1e-6, 토큰 배열의 최대 \|차\| ≤ 1e-6.
타임스탬프처럼 값이 아닌 필드만 다른 경우를 걸러 내기 위한 것이다. 넘으면 STOP.
결과: `results/V_inputs.json`.

---

## 5. 새 코드 (`analysis/ch3/`)

| 파일 | 내용 |
|---|---|
| `ch3_common.py` | 경로(`CH3`, `RES`), `TARGETS3 = ("strain", "inter", "solv", "barrier")`, 10 → 4 변환 행렬 `M` (아래), 쓰기 가드(`$CH3`과 `analysis/ch3/results` 밖이면 STOP), 원자적 쓰기, sha256, STOP(exit 3) · not ready(exit 2) |
| `data_ch3.py` | `Ch3Data(DeployData)`: DeployData('lf')와 입력 배열 · 짝 · 분할이 같고, `y`만 `y10 @ M`. 원래 10열은 `y10`으로 남긴다 |
| `train_ch3.py` | `smoke` · `c3r` · `run` · `collect` · `final` (6–10절). DEPLOY `train_tf.py`와 같은 동작에서 타깃 목록만 인자 |
| `eval_ch3.py` | 9절 평가. ch3과 lf 합에 같은 코드를 쓴다 |
| `predict_ch3.py` | DEPLOY `predict_tf.py`와 같은 인터페이스, 출력 타깃 4개 |
| `prereg_ch3.py` | 7절 사전 등록 쓰기 · 확인 (DEPLOY `prereg_deploy.py`와 같은 규칙) |
| `tests_ch3.py` | 6절 코드 검사 |
| `verify_inputs.py` | 4절 |
| `jobs/gpu_ch3.slurm`, `jobs/cpu_ch3.slurm` | deploy `jobs/gpu_fu2.slurm` · `cpu_py.slurm`을 복사하고 로그 경로만 `$CH3/logs`로, `DEP_SCRATCH=$CH3/lf` export 추가 |

**변환 행렬** (`TARGETS` 순서 d1 d2 elst pauli oi disp cpcm cds c_ghost barrier → strain inter solv barrier):

| | strain | inter | solv | barrier |
|---|---|---|---|---|
| d1, d2 | 1 | 0 | 0 | 0 |
| elst, pauli, oi, disp, c_ghost | 0 | 1 | 0 | 0 |
| cpcm, cds | 0 | 0 | 1 | 0 |
| barrier | 0 | 0 | 0 | 1 |

**타깃 수 바꾸기:** `fu2/train.py`의 `fit` · `loss_fn` · `predict`는 `fu2_common.TARGETS` · `I_BAR` · `I_CH9` · `I_PRIM`을
모듈 상수로 읽는다. deploy · fu2 파일은 고치지 않는다. 방법은 둘 중 하나다.
(a) 새 프로세스 시작 때 데이터 · 모델을 만들기 전에 이 상수들을 ch3 값(`TARGETS3`, `I_BAR=3`, `I_CH9=[0,1,2]`,
`I_PRIM=[0,1,2,3]`)으로 바꾸는 얇은 래퍼.
(b) 타깃에 의존하는 부분만 인자로 받는 사본을 `analysis/ch3/`에 두기.
어느 쪽이든 6절 C-3′(원래 10타깃으로 deploy 재현)와 C-T(ch3 모드 확인)를 통과해야 한다. `dep_common` · `data_lf`의
import 때 TARGETS 일치 단언은 상수를 바꾸기 **전에** import해서 그대로 통과시킨다.

---

## 6. Phase C — 코드 검사 (`tests_ch3.py`) — 하나라도 실패하면 STOP

| # | 검사 | 기준 |
|---|---|---|
| C-Y | 라벨 변환: `Ch3Data.y` = `y10 @ M` | 비트 단위 같음. \|strain + inter + solv − barrier\| 최대값과 분포 기록 (≈ 0.1 이하 예상, 0.2 넘으면 STOP) |
| C-D | 데이터 동일성: Ch3Data의 입력 배열 · 짝 목록 · 분할 목록 = `DeployData('lf')` | `np.array_equal`, 목록 sha256 같음 |
| C-3′ | **학습 회귀 (10타깃 모드)**: `train_ch3.py c3r` = 타깃 목록을 원래 10개로 두고 fold 0 전체 규칙(튜닝 4칸 + seed 5개)을 RTX 3090(gpu1, deploy K와 같은 GPU 종류)에서 실행 → deploy `K_report` fold 0 단위와 `K_predictions` fold 0과 비교 | (i) 선택 칸 같음 (ii) 최적 epoch 차 ≤ 2 (iii) fold 0 시험 para MAE가 타깃마다 ±2 % 안. 최대 \|Δyhat\| 기록. 산출 `$CH3/c3r/` |
| C-T | ch3 모드 확인: 모델 출력 차원 4, 손실의 항등 벌점이 `[0,1,2]`의 합 대 `3`, 검증 지표가 4 타깃 평균, 저장된 `.pt`의 `TARGETS` = `TARGETS3` | 모두 assert |
| C-4 | 반대칭 (DEPLOY C-4): 모든 fit, 무작위 시험 짝 200개 | \|g(a,b)+g(b,a)\| ≤ 1e-4, \|g(a,a)\| ≤ 1e-4 kcal/mol |
| C-5 | 표준화 누출 (DEPLOY C-5) | 시험 쪽 0개 |
| C-8 | smoke: ch3 모드 fold 0, 칸 최소 · 최대, 2 epoch, seed 1개 | 끝까지 돌고 C-4 · C-T 통과 |

결과: `results/C_tests.json` (`all_pass`).

---

## 7. Phase R — 사전 등록 (`results/PREREG_CH3.md` + `prereg_ch3.json`)

K 학습 전에 commit한다. `train_ch3.py run` · `final`은 두 파일이 git에 있고, HEAD 이후 바뀌지 않았고, json의 `md_sha256`
= md의 sha256일 때만 돈다.

**md에 적을 것:** 0-2 설계, 9절 지표 정의 전부, 9-2 채택 규칙, 비교 기준 파일(deploy `K_predictions.parquet`)과 라벨 ·
분할 · 입력의 sha256.

---

## 8. Phase K — 교차검증 학습 (`train_ch3.py run` / `collect`)

| 항목 | 값 |
|---|---|
| 단위 | `D__lf3__S__TF__f{0..4}` → `$CH3/units/<unit>.{json,parquet}`, `__s{0..4}.pt` |
| 학습 규칙 | DEPLOY 11-2 그대로 (grid 2/3층 × 64/128폭, AdamW lr 1e-3 wd 1e-4, cosine 300 epoch, dropout 0.1, batch 64짝, patience 30, (a,b)/(b,a) 순서 무작위 증강, seed 20261019 + 0..4 평균). 타깃 · 벌점 · 검증 지표만 0-2대로 |
| 멱등 | 단위 json이 있으면 지문(라벨 · 입력 · 분할 · 코드 sha256, GPU 이름) 비교 후 건너뜀. 다르면 STOP |
| GPU | RTX 3090 (`-p gpu1 --gres=gpu:rtx3090:1`), 배열 `--array=0-4%5`. 예상 약 1 GPU-h |
| collect | `results/K3_predictions.parquet`. 열은 deploy `K_predictions`와 같고 `rung = "lf3"`, `target ∈ TARGETS3`. `K3_val.csv`, `K3_report.json` |

---

## 9. Phase E — 평가 (`eval_ch3.py run`)

### 9-1. 공통

- **두 모델:** `ch3` = K3 예측. `lf_sum` = deploy `K_predictions`의 yhat을 0-2 식으로 더한 값(barrier는 직접 예측).
  정답 `y`는 둘 다 `$LAD/F_pairs_cr2.csv`. deploy `K_predictions`의 `y`가 이 파일과 같은지 먼저 확인한다(다르면 STOP).
- **짝 속성:** D1PAIR `S_folds.csv`(core, fam, sigma, in_P_main, in_P_sens)와 `F_pairs_cr2.csv`의 `dipole`, `cfg`(빈 값 → `-`),
  `series`, `pos`, `Xb`.
- **집합:** P_main = reference & in_P_main (주 비교). 두 모델이 모두 예측한 짝만 쓴다(공통 집합).
- **부트스트랩:** core 단위 2,000회, seed 20261019, `analysis/geo2/eval_geo2.py`의 `boot_diff`.

### 9-2. 주 비교와 채택 규칙 → `E3_compare.csv`

P_main · para 공통 짝에서 타깃마다 d = MAE(ch3) − MAE(lf_sum), 상대 차 d / MAE(lf_sum), 95 % CI.

**채택 규칙 (사전 등록):**
- strain · inter · solv 각각 상대 차 점추정 ≤ 0 **그리고** 95 % 상한 ≤ +5 %
- barrier는 95 % 상한 ≤ +5 %

모두 만족하면 **ch3을 배포 모델로 채택**한다. 아니면 STOP하지 않고 두 모델을 나란히 보고하며, 결정은 사용자에게 넘긴다.

### 9-3. 보조 지표 (두 모델 모두, 판정 아님)

| 산출 | 내용 |
|---|---|
| `E3_metrics.csv` | 타깃 × 위치(all, p, m, o, direct): n, MAE, RMSE, 상관 r, 부호 정답률(\|y\| ≥ 1인 짝), para MAE의 95 % CI |
| `E3_rank.csv` | 계열 안 장벽 순위. 계열 = fam × pos, P_sens reference 짝에 H(0, 0)를 더해 점 ≥ 8. 두 작용기 순서 정답률(실제 \|Δ\| 구간 0.41–1, 1–2, > 2), 실제 최저 작용기가 예측 1등인 비율 · 상위 3 안인 비율(무작위 기대값 3/n 함께), Spearman 중앙값 · 25 % 분위. para와 all |
| `E3_cause.csv` | **반응 하나:** \|y_barrier\| ≥ 1인 짝에서 (i) 주도 채널 = argmax(strain, inter, solv) × sign(barrier)의 일치율, (ii) 유형 S(strain이 barrier와 같은 방향, 나머지 inter + solv는 반대) · SI(둘 다 같은 방향) · I(나머지만 같은 방향)의 일치율, 각각 가장 흔한 범주만 찍었을 때의 기준값. **칸 평균:** 칸 = dipole × cfg × series × pos × Xb, 짝 ≥ 4. \|평균 y_barrier\| ≥ 1인 칸에서 같은 두 일치율과 기준값. 모든 칸에서 타깃별 칸 평균의 r · MAE |
| `E3_ad.json` | 적용 범위 문턱: 타깃별 시험 짝 `yhat_seed_sd`의 95 백분위 (ch3) |
| `E3_report.json` | 9-2 판정, 위 요약, 입력 sha256 |

참고값 (이번 대화에서 lf 합으로 계산, 같은 정의):

| 지표 | 값 |
|---|---|
| 짝 단위 MAE (all) | strain 0.81, inter 0.65, solv 0.52, barrier 0.75 |
| 부호 정답률 | strain 0.90, inter 0.93, solv 0.84 |
| 주도 채널 일치율 | 반응 하나 0.72 (기준 0.63), 칸 평균 0.84 (기준 0.71) |
| 유형 일치율 | 반응 하나 0.69 (기준 0.45), 칸 평균 0.83 (기준 0.53) |

`eval_ch3.py selftest`는 lf_sum으로 이 값을 재현해야 한다(소수 둘째 자리). 다르면 정의 차이를 보고한다(STOP 아님).

---

## 10. Phase W — 최종 모델 (`train_ch3.py final`)

9절 결과와 상관없이 실행한다(약 0.2 GPU-h).

| 항목 | 규칙 (DEPLOY 13과 같음) |
|---|---|
| 학습 짝 | `Ch3Data.split("all")` |
| 칸 · epoch | fold 선택 칸의 최빈값(동률: 튜닝 검증 평균이 낮은 칸 → grid 순서), epoch = 5 fold 최적 epoch 중앙값 |
| seed | 20261019 + 0..4 |
| 관문 | C-4 (무작위 학습 짝 200개), C-T |
| 산출 | `$CH3/model/D__lf3__S__TF__all__s{0..4}.pt`, `$CH3/model/MODEL_CARD.json`(복사본을 `results/MODEL_CARD_ch3.json`에), `results/W3_report.json` |

`MODEL_CARD`는 DEPLOY 13의 항목에 더해 `targets = TARGETS3`, 변환 행렬 `M`, 9-2 채택 여부를 적는다.

---

## 11. Phase I — 추론기 검사 (`predict_ch3.py`)

| # | 검사 | 기준 |
|---|---|---|
| I-1 | `--model cv`로 K3 시험 짝을 다시 예측 → `K3_predictions`와 비교 (fold별 5 seed 평균끼리) | ≤ 1e-4 kcal/mol |
| I-2 | `--model final`로 학습 짝 1,000개를 예측 → W 직후 기록과 비교 (같은 GPU) | ≤ 1e-6 |

결과: `results/I3_selftest.json`. 실패하면 STOP.

---

## 12. Phase S — 요약 `results/SUMMARY_CH3.md` (한국어)

포함할 것:
- V: 입력 재생성 결과
- C 검사 결과, C-3′ 수치
- 최종 칸 · epoch · 학습 짝 수
- **E3_compare 표**와 채택 판정
- E3_rank · E3_cause 요약 (ch3 대 lf 합, 기준값 함께)
- 적용 범위 문턱
- 실제 사용한 core-h · GPU-h
- 편차 목록

---

## 13. commit · push

| 무엇 | commit 여부 |
|---|---|
| `docs/specs/CH3_MODEL.md`, `analysis/ch3/` 코드 · job 스크립트 | commit |
| `analysis/ch3/results/` (json · csv · md, `K3_predictions.parquet`, `MODEL_CARD_ch3.json`, `lf_inputs/` 보고서) | commit. 파일당 50 MB 넘으면 commit하지 않고 요약에 경로만 적음 |
| `$CH3/` 안의 것 (토큰, 182열, `.pt` 모델, 로그) | **commit하지 않음** (`CLAUDE.md`: `/gpfs/tmp_cpu2/...` 출력 금지) |

단계마다 commit한다(S0 · V · C · PREREG · K · E · W · I · S). PREREG commit은 K 제출 전에 끝나야 한다.
마지막에 `git push -u origin ch3`를 하고, push된 HEAD SHA를 요약 끝에 적는다.

---

## 14. 예상 비용

| 단계 | 예상 | 근거 (deploy 실측) |
|---|---|---|
| L | 약 18 core-h | DEPLOY 요약 7절 |
| F | 약 6 core-h | 같음 |
| T | 약 4 core-h | 같음 |
| C-3′ | 약 0.25 GPU-h | DEPLOY C-3 |
| K | 약 1 GPU-h | DEPLOY K |
| W | 약 0.2 GPU-h | DEPLOY W |

---

## 15. STOP 조건

| 단계 | 조건 |
|---|---|
| 1 | 시작 조건 하나라도 실패 (특히 1-1: commit을 받을 수 없음) |
| 3 | deploy 관문 L-1 · L-2 · L-4, M, F-1 · F-3, T 실패 |
| 4 | V 실패 (내용 비교까지 넘음) |
| 6 | C 검사 하나라도 실패 |
| 7 | PREREG가 git에 없거나 바뀜 |
| 8 | 지문이 다른 캐시, C-4 · C-T 실패 |
| 9 | deploy `K_predictions`의 `y`가 `F_pairs_cr2.csv`와 다름 |
| 11 | I-1 · I-2 실패 |

9-2 채택 규칙을 못 넘는 것은 STOP이 아니다(보고만).

---

## 16. 실행 규칙 (저장소 `CLAUDE.md`)

- `sbatch`만, `--time=48:00:00`. 로그인 노드에 상주 프로세스 없음(nohup, 셸 반복문, 감시 프로세스 금지).
- 실행 10 / 대기 20 상한(배열 원소마다 1개). 제출 전 `squeue -u $USER -h -r | wc -l`.
- 긴 흐름은 `--dependency=afterany` 체인으로 잇는다: L → M → F → T → V → C(CPU) → C-3′ · C-8(GPU) → PREREG commit
  → K → collect → E → W → I → S.
- CPU 단계는 `sinfo -p cpu1,cpu2`에서 빈 쪽을 쓴다. GPU 단계(C-3′, K, W, I)는 RTX 3090(gpu1)을 쓴다. 비어 있지 않으면
  대기열에서 기다린다.
- 모든 단계는 멱등이고 원자적으로 쓴다. 48 h에 잘리면 같은 스크립트를 다시 제출한다.

| 단계 | 환경 |
|---|---|
| L, M | D1 run env (`jobs/cpu_d1.slurm`) |
| F, T | `analysis/d1_pair/jobs/g.slurm`, `PYENV=venv5` |
| V, C-Y, C-D | `env_d1fu2` (CPU) |
| C-3′, C-8, K, W, I | `env_d1fu2` (GPU) |
| E | `env_espley` |

---

## 17. 범위 밖

- 14만 반응 배포 실행(새 반응 열거, TS 탐색, 예측) — 다음 명령서
- 새 칸의 표본 DFT 확인
- 숨은 꺾임 분석 — D1 DFT로만 한다
