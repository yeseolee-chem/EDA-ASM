# ESPB — Espley-type baseline을 D1·짝 ΔΔ task에 맞춰 학습하고 GNN3R2와 비교 — Claude Code 실행 지시서

**저장소:** `origin/gnn3r2` HEAD(`f6d9ee3` 또는 그 뒤)에서 새 브랜치 **`espb`**를 만든다. 새 worktree는 `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-espb`(이하 `$WT`)다.
**이 파일:** `docs/specs/ESPB.md`로 commit한다.
**새 코드:** `analysis/espb/` · **결과:** `analysis/espb/results/` · **scratch:** `$EB=/gpfs/tmp_cpu2/yeseo1ee/espb`

**읽기만 하는 것:**
- GNN3R2 결과: `analysis/gnn3r2/results/`의 짝, 분할, CV 예측, 평가
- g2r 입력: `$G2R=/gpfs/tmp_cpu2/yeseo1ee/g2r`
- cr2 라벨: `$LAD=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder`
- 코드: `analysis/d1_pair/`, `analysis/espley_xtb_repro/`, `gnn3/` 패키지

**기존 코드는 고치지 않는다.** 필요한 코드는 `analysis/espb/`로 복사한 뒤, 2-3절 표에 적은 곳만 바꾼다.

**순서:** 1 시작 조건 → 2 브랜치·복사 → P(데이터 표) → G(사전 등록 commit) → A·B(학습, CPU) → C(모으기·대표 선택) → E(평가·비교) → F(그림) → R(보고·commit·push) → **멈춤**
- 11절 STOP 조건에서만 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다.
- 모든 계산은 `sbatch`로, `--time=48:00:00`를 쓴다(저장소 `CLAUDE.md`). 로그인 노드에서는 one-shot 제출, git, `squeue`·`scancel`만 한다.
- 제출 전에 `squeue -u $USER -h | wc -l`을 확인한다. MaxSubmit 20, MaxJobs 10이며 array 원소 하나가 한 칸을 쓴다.
- **test 점수를 보고 grid·모델·규칙을 바꾸지 않는다.** Espley 쪽이 이기거나 비슷해도 그것은 결과이지 STOP 사유가 아니다.

---

## 0. 무엇을, 왜

### 0-1. 목표

- **비유:** 같은 시험장(D1), 같은 시험지(test 짝 8,175개), 같은 채점자(core 분할, 같은 지표 코드)를 두고 선수만 바꾼다.
  - Espley 선수의 원래 시험은 "건물 하나의 높이 맞히기"(반응별 절대 장벽)였다.
  - 우리 시험은 "두 건물의 높이 차이 맞히기"(짝 ΔΔ)다.
  - 그래서 두 방식으로 시험을 보게 한다. E-A는 높이를 각각 재고 뺀다. E-B는 차이를 직접 잰다.
- **baseline:** Espley et al., *Digital Discovery* 2024, 3, 2479–2486 (doi:10.1039/D4DD00224E)의 SQM descriptor + Ridge/KRR/SVR/NN-2/NN-4 회귀. 교수님 지시(2026-11-16)에 따라 dataset과 task를 GNN3R2와 같게 맞춘다.
- **두 방식의 근거:** "절대값을 예측한 뒤 빼기"(E-A)와 "차이를 직접 학습"(E-B)의 구분은 Fralish et al., *J. Cheminform.* 2023, 15, 101 (doi:10.1186/s13321-023-00769-x) Fig. 1A/B와 같다.
- **산출:** 타깃(strain, inter, solv, barrier)마다 Espley 쪽 대표 모델 하나를 정한다. 그 대표와 GNN3R2를 **같은 test 짝, 같은 지표 코드(`gnn3.metrics`)**로 비교한다.

### 0-2. Espley 원본에서 바꾸는 점 (보고서와 논문 SI에 이 목록을 그대로 싣는다)

| # | Espley 2024 원본 (ds3) | ESPB | 이유 |
|---|---|---|---|
| X-1 | Coley [3+2], ωB97X-D/def2-TZVP 라벨 | GNN3R2 데이터셋 8,594 반응 / 8,344 짝, cr2 B3LYP-D3(BJ)/def2-TZVP SMD(water) 라벨 | 데이터 동일 |
| X-2 | AM1(Gaussian) descriptor 46개 | GFN2-xTB/ALPB(water) **ESPLEY46** (`train_ml_single.ESPLEY46` = DIST11 + MULLIKEN15 + VALENCE15 + E5). APT 전하 대신 Wiberg valence, q_barrier 없음 | 사용자 결정(2026-11-16). rev 4에서 ds3의 같은 반응에 대해 xTB 판이 Espley AM1 판보다 5개 타깃 모두 낮았다 |
| X-3 | AM1로 최적화한 TS·GS | g2r xTB TS (`$G2R/lf/X182_lf.parquet`, GNN3와 같은 파일) | 입력 정보 수준 동일 |
| X-4 | 무작위 80/10/10, seed 22/23/14/1/2 | GNN3R2 core 단위 outer fold 5개 (`results/splits/T1__f<k>.json`) | 분할 동일 |
| X-5 | seed 23 train에서 KFold(5) GridSearchCV 한 번 | outer fold마다 GroupKFold(4, group = core) — GNN3 inner split과 같은 core 묶음 | 튜닝 조건 동일, baseline에 유리한 쪽 |
| X-6 | ESI Table S3 grid. KRR은 표에서 polynomial, 본문에서 RBF | ESI Table S3 ∪ rev 4 grid(`train_ml_single.GRIDS`). KRR은 RBF로 튜닝하고 실행한다. 효과 없는 값(Ridge `tol`, RBF SVR `coef0`·`degree`)은 뺀다(P-9로 확인) | 원본보다 좁지 않은 grid, 커널 불일치 해소 |
| X-7 | y 원값 | y 표준화(rev 4 `_make_pipe`), NN은 `r6_nn` 그대로 | rev 4에서 Espley 원본을 이긴 설정 |
| X-8 | 타깃 5개(dipole·dipolarophile distortion, Interaction, ΔE‡, ΔG‡) | E-A: d1, d2, inter, solv, barrier. Interaction을 inter와 solv로 나눈다. ΔG‡는 D1에 라벨이 없어 뺀다 | 우리 3채널 |
| X-9 | 반응별 절대값 | 짝 ΔΔ: E-A는 f(a) − f(b), E-B는 반대칭 짝 모델(D1PAIR arm C 구현) | task 동일 |
| X-10 | seed별 분할의 평균 MAE | NN 최종 예측 = seed 22, 23, 14, 1, 2 예측의 평균 | GNN3의 5-seed 평균과 맞춘다 |

### 0-3. 하지 않는 것

| 항목 | 이유 |
|---|---|
| AM1 원본 descriptor 계산 | 사용자 결정 2026-11-16: xTB 46개로 한다 |
| feature 효과와 구조 효과를 나누는 arm (Espley 회귀기 + X182 전체) | 사용자 결정 2026-11-16: 나누지 않는다 |
| XGB | Espley 모델이 아니다 |
| GNN3 재학습 | GNN3R2 CV 예측(`analysis/gnn3r2/results/cv/gnn3_fold*.parquet`, commit되어 있음)을 그대로 쓴다 |
| 이전 결과 삭제 | 이 실험이 대체하는 이전 판이 없다. D1PAIR, LADDER, GNN3R2는 그대로 둔다 |

---

## 1. 시작 조건 (하나라도 아니면 STOP) → `results/S0_start.json`

| # | 조건 |
|---|---|
| 1-1 | `origin/gnn3r2` = `f6d9ee3` 또는 그 뒤. `analysis/gnn3r2/results/RUN_REPORT_GNN3R2.md`의 모든 단계가 통과 |
| 1-2 | `analysis/gnn3r2/results/`의 sha256이 다음과 같다. `F_pairs_r.csv` `9d75fae8b7ba39ab7d83112342a7ddd299017ad7eb3b05f97a4409deec4f3455`, `S_folds_r.csv` `0f0895611d42c276bcf28e0346027aac33c7c0c8454b4e64a4366e7cd7ce278a`, `S_splits_r.json` `8b660f9fb8f3e3d794b166d18c0e93bce50a1c3ae93f0f4de3fe57fbfc260ef5`, `B_label_report.json` `a3e30bf438d21aa817f317903135e06cc7ef9ff81ae464d4c8c361e2405d5007`, `eval/metrics.csv` `9d176e94eac49bc12912ffa2c3f1d27f2d61352bce7a159e3727dacabbe30cc1` |
| 1-3 | split 파일 sha256: `splits/T1__f0.json` `b4353087b159dd1e8ef92d570c6907288729c42f4701c4e0df85979b8aada835`, `f1` `3de69f5ad84174beb3ee444f2d119fec41ccea4c3cec50f8f1dbc2e08dbd0d0e`, `f2` `92d9ddac1c40883a9ee8ceacf70c25378e3ebe3d416f1690040b506d61d6a84d`, `f3` `887fb7a50e127f7a14d336a818b05b54f3cdf327067f9b32113bdc424f4ad17d`, `f4` `deeb6cb4fcce2afb4005a98eac2cd2690d5e4df04fcf31fc101c1cbf0c6f6452` |
| 1-4 | GNN3 CV 예측 sha256: `cv/gnn3_fold0.parquet` `5fcbdad85b7b96ce202e2412ecb2ba45069e82c85504c434671ad4a73333db29`, `fold1` `7c40ec36a26a9887887f86ca0e547ea2aedc53a81953858199cc7d6620d830a3`, `fold2` `843a44eace18ead6d4556d1866b62899af2f56226389a67d08e1606c4b765543`, `fold3` `c852ccdd4b288938a94db59d7529f99ce7fcddf0486a72edee7137b4669c482d`, `fold4` `3f9949c9b391e1c09a01799455c100d9cabc33556d254aba076b97441cb1aabe` |
| 1-5 | `$G2R/lf/X182_lf.parquet` sha256 = `66c40316319870214e4773b5a838af380d90220f1318536d5b6ac7524955754d` (GNN3R2 `cv/gnn3_fold0.json` `settings.inputs_sha256.x182`) |
| 1-6 | `$LAD/labels_d1_cr2.json` sha256 = `7e5882bce4d47dcd8164b61320a390530463bfa14731029c122d7f09920b6755` (GNN3R2 `B_report.json`의 입력 기록) |
| 1-7 | 복사 원본 코드(`origin/gnn3r2` 기준) sha256: `analysis/d1_pair/d1_train.py` `414b5cc66762391691b252235c9133f0d193312a8039d35369299a8dfe30027c`, `analysis/d1_pair/d1_f0.py` `6cc918984cd63feb1a9ec262c73ce2e9272cada95f1cf6130ea8d8ccb89f152b`, `analysis/d1_pair/d1pair_common.py` `65a6e2da81e4dedf20c51423cbfe9c97e8b8f2272b24454a739a09e8552aeedc`, `analysis/espley_xtb_repro/r6_nn.py` `9824c153e835f3c977f027bfac578f6253c30f113e938ce3b7b1c110dedf0846`, `analysis/espley_xtb_repro/train_ml_single.py` `05c6b38cdf7803f4a6c6b2d643555db95b5e366ea02714049232add38a55da4c`, `gnn3/gnn3/metrics.py` `86bb77c685255f1da07bc99dee33cfd79902c499f647ade31d041d04644a6d88`, `gnn3/scripts/evaluate.py` `e8b089b81e9ec5cedc898a3769ef27e365da7fabc5f53ccafb3324c281f1adc1` |
| 1-8 | 환경. `env_espley`(`/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley/bin/python`)의 판: sklearn 1.3.1, tensorflow 2.16.1, keras 3.3.3, keras_tuner. `env_d1fu2`(`gnn3/jobs/run.slurm`)에서 `gnn3.metrics`가 import된다. 판을 기록한다 |
| 1-9 | `$EB`가 없거나 비어 있다. 디스크 여유 20 GB 이상 |

sha256이 다르면 STOP한다.

---

## 2. 브랜치 · 복사 · 경로

### 2-1. 브랜치

```
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction        # 주 저장소 (실제 경로 확인)
git fetch origin gnn3r2:refs/remotes/origin/gnn3r2
git worktree add -b espb /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-espb origin/gnn3r2
sbatch -p cpu1,cpu2 --time=48:00:00 --wrap "mkdir -p $EB/logs $EB/P $EB/units/EA $EB/units/EB $EB/nn"
```
- 이 지시서를 commit한다.
- 제출할 때마다 `--output=$EB/logs/%x_%j.out`(array는 `%x_%A_%a.out`)으로 덮어쓴다.

### 2-2. 새 폴더 `analysis/espb/`

| 파일 | 내용 |
|---|---|
| `eb_common.py` | 경로, 상수, grid, 원자적 쓰기. `gr_common.py`의 helper(`sha256`, `write_atomic`, `write_json`, `read_json`)를 복사한다 |
| `s0_start.py` | 1절 |
| `prep.py` | Phase P |
| `prereg.py` | Phase G: 사전 등록 파일 생성과 확인 |
| `ea_train.py` | Phase A (E-A). `analysis/d1_pair/d1_f0.py`를 복사하고 2-3절 표대로 바꾼다 |
| `eb_train.py` | Phase B (E-B). `analysis/d1_pair/d1_train.py`를 복사하고 2-3절 표대로 바꾼다 |
| `collect.py` | Phase C |
| `compare.py` | Phase E (`env_d1fu2`, `gnn3.metrics` 사용) |
| `figures.py` | Phase F |
| `report.py` | Phase R (보고서와 manifest `results/espb.json`) |
| `jobs/env.sh`, `jobs/py.slurm` | `analysis/d1_pair/jobs/`의 사본. 바꾸는 것은 scratch 변수(`ESPB_SCRATCH`, 기본 `$EB`)와 로그 경로뿐 |
| `jobs/submit_train.sh` | A·B 학습 → C → E → F를 의존성 사슬로 한 번에 제출(12절) |

### 2-3. 사본에서 바꿀 곳 (이 밖의 수정은 실행 전에 `results/DECISIONS.md`에 적고 보고)

| 사본 | 원본 | 바꿀 곳 | 그대로 둘 곳 |
|---|---|---|---|
| `ea_train.py` | `d1_f0.py` | 행 = outer fold k의 학습 반응. feature = ESPLEY46(`X46.parquet`). 타깃 = 5절 E-A의 5개. CV = P-6의 core → inner fold 지도(GroupKFold(4)), `KFold(5)` 아님. grid = 5-1절 표. NN의 validation_data = inner fold 1의 val 반응(Hyperband 목표와 fit history에만 쓰고 학습 행은 줄이지 않음). 최종 fit = seed 5개. 출력 = test 반응 예측 + inner OOF 예측. 경로·캐시·지문·코드 버전 `espb-ea-1` | `_make_pipe` + `GridSearchCV(scoring="neg_mean_absolute_error")`, `r6_nn.tune` / `r6_nn.fit_predict` 호출 방식, NN 단계마다 새 프로세스, 지문 검사 |
| `eb_train.py` | `d1_train.py` | `data()` = Phase P 표(짝 8,344, ESPLEY46, GNN3R2 분할). `unit_list` = arm EB × 모델 5개(XGB 제외) × 타깃 4개 × fold 0–4. `target_y` = 채널 ΔΔ(f0를 빼지 않음). NN `fit` 단계를 seed 5개로 돌려 평균한다. 경로·캐시·지문·코드 버전 `espb-eb-1`. 사전 등록 gate = `prereg.py` | `Scale`, `sqd`·`dist3`·`kern`, `_krr_cell`·`_svr_cell`·`_svr_one`, `grid_cells`(Ridge·KRR·SVR), `_tune_grid`, `fit_final`, `nn_model`·`nn_builder`·`nn_stage`의 tune·cv, gate M-1(반대칭), `d1pair_common`의 상수 값(`GAMMAS`, `SVR_C`, `SVR_EPS`, `N_INNER`, `SCALE_EPS`, `ANTISYM_TOL`, `ANTISYM_N`, `SEED`, `CV_SEED`, `TUNE_SEED`, `NN_SIZE`) |

- 원본과 사본의 diff를 `results/COPY_DIFF.patch`에, 원본 sha256을 `results/COPY.json`에 기록한다.
- 상수는 `eb_common.py`에 같은 값으로 다시 적는다. `s0_start.py`는 그 값이 `d1pair_common`의 값과 같은지 import로 확인한다. `d1pair_common`은 읽기만 한다.

---

## 3. Phase P — 데이터 표 (`prep.py`, `env_espley`, CPU) → `results/P_report.json`

### 3-1. 만드는 것

| 파일 | 내용 |
|---|---|
| `$EB/P/X46.parquet` | rxn_id + ESPLEY46 46열(`train_ml_single.ESPLEY46` 순서). `X182_lf.parquet`에서 그대로 뽑는다 |
| `results/P_reactions.csv` | rxn_id, job_id, core, fold_T1, 반응별 라벨 10개(`d1pair_common.D1_TCOL` 열 이름, `labels_d1_cr2.json`), 반응별 채널 4개 |
| `results/P_inner_folds.csv` | outer fold k, core, inner fold j (0–3) |
| `results/P_report.json` | gate와 개수 |

**반응별 채널**(짝 채널과 같은 정의, `gnn3.channels.labels_to_channels`):
- strain = d1 + d2
- inter = elst + pauli + oi + disp + c_ghost
- solv = cpcm + cds
- barrier = barrier

### 3-2. gate

| # | 검사 | 실패 시 |
|---|---|---|
| P-1 | 짝 8,344개. fold별 test 짝 수 = {−1: 169, 0: 1,682, 1: 1,705, 2: 1,414, 3: 1,344, 4: 2,030}. test 합 8,175 | STOP |
| P-2 | 반응 = a_rxn ∪ b_rxn = 8,594개. 모든 짝에서 a와 b의 core가 같다. 반응마다 core가 하나다 | STOP |
| P-3 | ESPLEY46 46열이 `X182_lf.parquet`에 모두 있고, 8,594 반응 모두 유한하다 | STOP |
| P-4 | 반응별 라벨로 다시 만든 `dd_<10개>`가 `F_pairs_r.csv`와 같다(max \|차\| ≤ 1e-9). 반응별 채널의 짝 차이가 `gnn3.metrics.channel_truth(F_pairs_r.csv)`와 같다(≤ 1e-9) | STOP |
| P-5 | outer fold k마다 train·test 짝 목록이 `T1__f<k>.json`과 정확히 같다. 학습 반응 ∩ test 반응 = ∅ | STOP |
| P-6 | outer fold k마다 학습 짝(split json 순서)에 core로 GroupKFold(4)를 건다(= `d1_train.inner_folds`). **첫 split의 val 짝 = `T1__f<k>.json`의 `inner_val`**(정확히 같음). core → inner fold 지도를 만들고, 반응은 자기 core의 inner fold를 받는다 | STOP |
| P-7 | 반응별 닫힘 \|barrier − (d1 + d2 + elst + pauli + oi + disp + cpcm + cds + c_ghost)\|의 최댓값 | 보고만 |
| P-8 | xTB 에너지 규약: \|xtb_e_barrier − (xtb_sum_distortion + xtb_interaction)\|의 최댓값, xtb_sum_distortion = xtb_dist_dipole + xtb_dist_dipolarophile인지 | 보고만(pre-ML 표의 부호 규약으로 기록) |
| P-9 | 효과 없는 hyperparameter 확인. fold 0 학습 반응에서 무작위 500개(seed 20261019), 타깃 barrier. (a) Ridge `tol` 0.1과 1.0의 예측이 같다, (b) RBF SVR `coef0` {0, 1} × `degree` {1, 3}의 예측이 같다. 둘 다 max \|차\| ≤ 1e-10 | STOP (grid 축소 근거가 무너짐) |

---

## 4. Phase G — 사전 등록 (`prereg.py`) → `results/PREREG_ESPB.md`, `results/prereg_espb.json`

A·B를 하나라도 제출하기 **전에** commit한다. `ea_train.py`·`eb_train.py`는 다음을 확인하고, 아니면 exit 3으로 멈춘다.
- 두 파일이 HEAD에 commit되어 있다.
- json에 적힌 sha256(`eb_common.py`, `ea_train.py`, `eb_train.py`, `collect.py`, `compare.py`, P 산출물)이 현재 파일과 같다.

등록할 내용:
1. 0-2절 변경 목록과 0-3절 제외 목록
2. 5절 grid, 전처리, seed
3. 대표 선택 규칙(7-2절)
4. 주 비교와 보조 비교의 정의(8-2절)
5. Holm 보정 범위

---

## 5. 모델

### 5-1. E-A — 각각 예측하고 빼기 (Espley 원형, `ea_train.py`)

- **학습 단위:** outer fold k의 학습 반응(학습 짝의 a ∪ b, fold −1 짝 포함). 예측 대상은 test 반응이다.
- **타깃(반응별 절대값) 5개:** d1, d2, inter, solv, barrier.
- **짝 예측:**
  - strain = [f_d1(a) + f_d2(a)] − [f_d1(b) + f_d2(b)]
  - inter, solv, barrier = f_t(a) − f_t(b)
  - a, b를 바꾸면 부호만 바뀌는 성질(반대칭)은 자동으로 성립한다.
- **sklearn 모델:** `train_ml_single._make_pipe(est)`(X StandardScaler → est, y 표준화). `GridSearchCV(cv = P-6 inner fold 4개, scoring="neg_mean_absolute_error", refit=False)`. 최적은 평균 inner MAE가 가장 낮은 칸이고, 동률이면 grid 순서의 앞 칸이다.

| 모델 | grid (Espley ESI Table S3 ∪ `train_ml_single.GRIDS`) | 칸 수 |
|---|---|---:|
| Ridge | alpha {1e-4, 1e-3, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000} | 16 |
| KRR (RBF) | alpha {1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.2, …, 0.9, 1} × gamma {None, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 0.5, 0.9, 1} | 154 |
| SVR (RBF) | C {1, 10, 30, 50, 100, 1000} × gamma {"scale", "auto", 1e-3, 1e-2, 1e-1} × epsilon {0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1} | 240 |

- **NN-2 / NN-4:** `r6_nn`을 그대로 쓴다. 세 단계 모두 새 프로세스에서 돈다.
  1. **tune:** seed 23, `TF_ENABLE_ONEDNN_OPTS=0`. inner fold 1의 train 반응 → val 반응으로 `r6_nn.tune`.
  2. **cv:** seed 22. inner fold 4개를 고른 hps로 다시 fit(`r6_nn.fit_predict`, validation_data = 그 inner val)해 OOF 예측을 만든다.
  3. **final:** seed 22, 23, 14, 1, 2 각각 `r6_nn.fit_predict`(학습 = 학습 반응 전부, validation_data = inner fold 1의 val 반응으로 history 전용). 예측은 다섯 seed의 평균이다.
- **inner OOF 예측(대표 선택용):** 최적 칸으로 inner fold j마다 j 밖의 학습 반응으로 fit하고 j의 반응을 예측한다. sklearn 모델은 refit, NN은 위 cv 단계다.
- **unit:** (모델, 타깃, fold) = 5 × 5 × 5 = **125개**.
  - `$EB/units/EA/<model>__<target>__f<k>.{json,parquet}`
  - parquet: rxn_id, role ∈ {test, oof}, inner_fold, yhat, NN이면 yhat_s22 … yhat_s2
  - json: 최적 칸, grid 표(칸별 inner 절대 MAE), 시간, 지문, 판

### 5-2. E-B — 차이를 직접 학습 (task 완전 일치, `eb_train.py`)

- **학습 단위:** outer fold k의 학습 짝. y는 채널 ΔΔ(strain, inter, solv, barrier = `channel_truth`와 같은 값)다.
- **입력과 반대칭:** `d1_train.py` 그대로다.
  - 입력 z = [Δx / s_Δ, (x̄ − μ) / s] (Δx = x_a − x_b, x̄ = (x_a + x_b) / 2)
  - 모든 모델은 g(−Δx, x̄) = −g(Δx, x̄), g(0, x̄) = 0을 만족한다.
  - Ridge: Δx만 쓰고 절편은 없다.
  - KRR: 반대칭 곱 커널 [k(Δx_p, Δx_q) − k(Δx_p, −Δx_q)] · k(x̄_p, x̄_q).
  - SVR: 같은 커널, 양방향 학습, ½[s(Δx) − s(−Δx)].
  - NN-2/NN-4: 공유 부분망 h(`r6_nn._layers`), g = h([Δx, x̄]) − h([−Δx, x̄]).
- **grid:** `d1_train.grid_cells` 그대로다(D1PAIR arm C와 같음). Espley 원본에는 짝 모델이 없으므로 원본 grid가 없다.
- **튜닝:** GroupKFold(4, core) = P-6. 최적 = 평균 inner MAE(kcal/mol) 최저.
- **NN:**
  - tune(inner fold 1, seed 23)과 cv(inner 4개, seed 22)는 `d1_train` 그대로다.
  - **final은 seed 22, 23, 14, 1, 2로 5번 돌려 평균한다**(바뀐 곳).
  - gate M-1은 seed마다 검사한다.
- **unit:** (모델, 타깃, fold) = 5 × 4 × 5 = **100개**.
  - `$EB/units/EB/<model>__<target>__f<k>.{json,parquet}`
  - parquet: pair_id, yhat, NN이면 seed별 열
  - json: 최적 칸, 칸별 inner MAE, 최적 칸의 inner fold별 MAE, M-1 값, 지문

---

## 6. Phase A·B — 학습 (CPU, `env_espley`)

- 실행 방식은 `d1_train.py`와 같다: `run <i> <n>`은 unit 목록(정해진 순서)의 i번째 조각을 맡는다. 끝난 unit은 지문이 맞으면 건너뛴다. 출력은 원자적으로 쓴다.
- 자원 배분은 `analysis/d1_pair/jobs/submit_M.sh`를 따른다(12절).

| # | 검사 | 실패 시 |
|---|---|---|
| A-1 / B-1 | 캐시된 unit의 지문(입력 sha256, grid, 코드 버전, 판)이 현재와 같다 | STOP |
| A-2 / B-2 | 예측과 inner MAE에 NaN·inf가 없다 | STOP |
| A-3 | 학습 반응 ∩ test 반응 = ∅ (unit마다 assert) | STOP |
| B-3 | gate M-1: test 짝 200개에서 \|g(Δx) + g(−Δx)\| ≤ 1e-6, \|g(0)\| ≤ 1e-6 kcal/mol (NN은 seed마다) | STOP |
| A-4 / B-4 | NN unit에 seed 5개의 예측이 모두 있다 | STOP |

---

## 7. Phase C — 모으기와 대표 선택 (`collect.py`, `env_espley`)

### 7-1. 짝 예측 만들기
- **E-A:** 5-1절 식으로 test 짝 ΔΔ를 만든다.
  - NN은 seed별 ΔΔ를 먼저 만들고, 평균을 yhat, 표준편차를 yhat_seed_sd로 둔다. sklearn 모델의 yhat_seed_sd는 0이다.
  - strain은 같은 모델의 d1, d2 unit을 쓴다.
- **E-B:** unit 예측을 그대로 쓴다.
- **출력:**
  - `results/C_predictions.parquet`: pair_id, ref_kind, pos, fold, arm ∈ {EA, EB}, model ∈ {Ridge, KRR, SVR, NN2, NN4}, target, y, yhat, yhat_seed_sd
  - `results/C_reactions.parquet`: E-A의 반응별 test 예측. rxn_id, fold, model, target(d1, d2, inter, solv, barrier), y, yhat

### 7-2. 대표 선택 (사전 등록, test 점수 미사용)
- **후보:** 타깃 t ∈ {strain, inter, solv, barrier}마다 {EA, EB} × {Ridge, KRR, SVR, NN2, NN4} = 10개.
- **점수:** inner ΔΔ MAE.
  - outer fold k마다 inner fold 4개의 "inner val 짝 전체에 대한 MAE"를 평균한다.
  - 그 값을 다시 5개 fold에 대해 평균한다.
  - E-A는 inner OOF 반응 예측으로 inner val 짝의 ΔΔ를 만들어 계산한다.
  - E-B는 unit json의 최적 칸 inner fold별 MAE를 쓴다. NN은 cv 단계 값이다.
- **고르기:** 점수가 가장 낮은 후보다. 동률이면 EA → EB, Ridge → KRR → SVR → NN2 → NN4 순서의 앞 것이다.
- **순서 강제:**
  - `collect.py`는 먼저 json·OOF만 읽어 `results/C_selection.csv`를 쓰고, 그 sha256을 `C_report.json`에 기록한다.
  - 그 뒤에야 test 예측을 모은다.
  - 대표 모델을 "ESPB"라는 이름의 합성 모델로 만든다. 타깃마다 고른 후보의 행이다.

### 7-3. gate

| # | 검사 | 실패 시 |
|---|---|---|
| C-1 | 후보 10개 × 타깃 4개마다 짝 집합 = GNN3 CV 예측의 짝 집합(8,175)이고, fold도 같다 | STOP |
| C-2 | y = `channel_truth` 값 = GNN3 parquet의 y (≤ 1e-9) | STOP |
| C-3 | NaN 없음, unit 225개가 모두 있음 | STOP (unit이 비면 exit 2, 학습 재제출) |
| C-4 | `C_selection.csv`가 test 열을 읽지 않고 만들어졌다(코드 경로상 test parquet을 열기 전에 기록) | STOP |

---

## 8. Phase E — 평가와 비교 (`compare.py`, `env_d1fu2`, CPU; `gnn3/jobs/run.slurm`)

### 8-1. 채점 코드 동일성
- **E-0:**
  - GNN3 예측을 `gnn3/scripts/evaluate.py`의 `read_predictions`로 읽는다.
  - `M.channel_truth(F_pairs_r.csv)`, `M.pair_sets(S_folds_r.csv)`, `M.assemble`, `M.error_table`로 다시 계산한 표가 `analysis/gnn3r2/results/eval/metrics.csv`와 같아야 한다(수치 열 ≤ 1e-12).
  - 다르면 STOP한다.
- 이후의 모든 표는 같은 `M.assemble`에 모델 사전 `{"GNN3", "ESPB", "EA-Ridge", …, "EB-NN4"}`를 넣어 만든다.

### 8-2. 비교

**주 비교(확증, 사전 등록)**
- 집합: P_main, 위치 all. 타깃 4개.
- 값: MAE(GNN3), MAE(ESPB), d = MAE(ESPB) − MAE(GNN3).
- 검정: `M.bootstrap_mae_difference(err_ESPB, err_GNN3, core)`로 95 % CI와 양측 p를 낸다. core bootstrap 2,000번, seed 20261019이다.
- 다중 비교: 4 타깃에 Holm 보정 → `results/E_primary.csv`

**보조 비교(설명용, Holm은 따로 표시)** → `results/E_secondary.csv`
- P_main para, test_all, 위치별(p, m, o, direct), series별(R1, R2)
- 후보 10개 각각과 GNN3의 d와 CI
- 0을 예측하는 기준의 MAE와 모델별 개선폭(MAE(0) − MAE)
- 수축 기울기: P_main barrier에서 y에 대한 yhat의 OLS 기울기(series별)

**표 4종** (모든 모델)
- `results/E_metrics.csv` = `M.error_table`
- `results/E_rank.csv` = `M.ranking_table` (para·all family의 최저 장벽 top-1/top-3, 순서 정확도, Spearman)
- `results/E_cause.csv` = `M.cause_agreement`
- `results/E_breakdown.csv` = `M.breakdown_table` (GNN3, ESPB)

**Espley 쪽 지표 병기**
- `results/E_native.csv`: E-A 반응별 절대 test MAE. 모델 5개 × {d1, d2, inter, solv, Int = inter + solv 예측 합, barrier}.
  - fold별 MAE를 평균하고, Espley 정의의 "test 범위 대비 MAE %"(fold별 MAE / (max − min)의 평균)를 함께 적는다.
  - GNN3는 짝 모델이라 이 표에 없다고 표 아래에 적는다.
- `results/E_preml.csv`: ML 이전 xTB의 DFT 대비 MAE.
  - 반응별: xtb_dist_dipole 대 d1, xtb_dist_dipolarophile 대 d2, xtb_interaction 대 (inter + solv), xtb_e_barrier 대 barrier. 대상은 8,594 반응이다.
  - 짝: Δxtb_sum_distortion 대 strain, Δxtb_interaction 대 (inter + solv), Δxtb_e_barrier 대 barrier. 대상은 P_main test 짝이다.
  - 부호 규약은 P-8을 따른다.

---

## 9. Phase F — 그림 (`figures.py`, matplotlib, `PYENV=reactot`) → `results/figures/`

**공통 규격**
- PDF와 PNG(300 dpi)를 둘 다 저장하고, 그림마다 같은 이름의 CSV(그린 값)를 둔다.
- 폰트는 Arial(없으면 DejaVu Sans) 7 pt, 단단 89 mm 또는 양단 183 mm.
- 색은 엔티티에 고정한다(rev 5 검증 팔레트): **GNN3 `#eb6834`, Espley 쪽 `#2a78d6`**. 글자는 검정·회색이다.
- 이중 y축은 쓰지 않는다. 계열이 2개 이상이면 범례를 두고, 막대 사이에 2 px 간격을 둔다.

| 파일 | 내용 |
|---|---|
| `fig1_primary` | 타깃 4개 × (GNN3, ESPB)의 MAE 막대와 core bootstrap 95 % CI. 두 패널: P_main all, P_main para. 값 라벨을 달고, 타깃마다 d와 Holm p를 주석으로 단다. ESPB 막대 아래에 고른 후보 이름(예: EA-KRR)을 적는다 |
| `fig2_parity` | P_main barrier ΔΔ, x = DFT, y = 예측. 두 패널(GNN3, ESPB), 축 공유, y = x선, ±1 kcal/mol 회색 띠, MAE와 r 주석 |
| `fig3_candidates` | 타깃 4개 small multiples. 후보 10개의 P_main MAE 점(E-A ●, E-B ■, 모두 Espley 색)과 GNN3 가로선, MAE(0) 점선. inner 점수 순위를 점 옆에 적는다 |
| `fig4_positions` | P_main 위치(p, m, o, direct)별 MAE, 타깃 4개 × (GNN3, ESPB) |

저장한 PNG를 열어 라벨 겹침과 잘림을 직접 확인한다.

---

## 10. Phase R — 보고 · commit · push

### `results/RUN_REPORT_ESPB.md` (한국어, 표 위주)
1. 단계별 판정 표(1, 2, P, G, A, B, C, E, F)
2. **주 비교 표**(E_primary): 타깃 × MAE(GNN3), MAE(ESPB)와 고른 후보, d (95 % CI), p, Holm p, 한 줄 해석
3. 보조: para·test_all·위치·series, 0 예측 대비 개선, 수축 기울기, family top-1/top-3, 부호 정확도
4. 후보 10개 표: inner 점수, 순위, test MAE(P_main all, para)
5. Espley 쪽 지표: E_native(% range 포함), E_preml
6. 0-2절 변경 목록(X-1 … X-10)과 0-3절 제외 목록
7. 그림 4개와 각 한 줄 설명
8. 결정·편차(`DECISIONS.md` 요약), gate 값(P-7, P-8 포함)

**화합물 표기:** 개별 반응이나 치환기를 예로 들 때는 이름이 아니라 구조식 그림(RDKit, d1 SMILES, `results/figures/struct_*.png`)으로 보인다. 많으면 대표 하나만 보인다.

### manifest와 commit
- `results/espb.json`: 모든 결과 파일과 입력의 sha256, 코드 sha256, 판, SLURM job id
- commit 대상: `analysis/espb/`(코드, `results/`, 그림)와 `docs/specs/ESPB.md`. `$EB` 아래 파일은 commit하지 않는다.
- `git push -u origin espb`. push한 HEAD를 보고서에 적어 다시 commit·push한다.

**여기서 멈춘다.** 삭제할 이전 판은 없다(0-3절).

---

## 11. STOP 조건

| 상황 | 행동 |
|---|---|
| 1절 시작 조건 실패 | 멈추고 보고 |
| P-1 … P-6, P-9 실패 | 멈추고 보고 |
| 사전 등록 파일이 commit되지 않았거나 sha256이 다른데 학습을 시작하려 함 | 학습 스크립트가 exit 3으로 멈춤 |
| A-1 … A-4, B-1 … B-4 실패 | 멈추고 보고(어느 unit인지) |
| C-1, C-2, C-4, E-0 실패 | 멈추고 보고 |
| 2-3절 밖의 코드 수정이 필요 | `DECISIONS.md`에 적고 보고한 뒤 승인을 받는다 |
| 48 h 벽에 걸린 job | 같은 명령으로 다시 제출(멱등). STOP 아님 |
| ESPB가 GNN3보다 낫거나 차이가 없음 | STOP 아님. 그대로 보고한다 |

---

## 12. 계산 규모와 제출 (`jobs/submit_train.sh`, 로그인 노드에서 한 번 실행)

| 묶음 | unit | 자원 (`submit_M.sh` 기준) | array |
|---|---|---|---|
| EA 격자 (Ridge, KRR, SVR) | 75 | `--cpus-per-task=48 --mem=160G`, `ESPB_NJOBS=48` | 0-1 |
| EA NN (NN2, NN4) | 50 (tune 1 + cv 4 + final 5 프로세스씩) | `--cpus-per-task=48 --mem=180G`, `ESPB_WORKERS=24` | 0-3 |
| EB 격자 (Ridge, KRR) | 40 | `--cpus-per-task=48 --mem=160G` | 0-1 |
| EB SVR | 20 (가장 오래 걸림: 칸 768개 × inner 4) | `--cpus-per-task=48 --mem=400G` | 0-3 |
| EB NN (NN2, NN4) | 40 | `--cpus-per-task=48 --mem=180G`, `ESPB_WORKERS=24` | 0-3 |
| collect → compare → figures → report | — | `afterok` 사슬. collect는 unit이 비면 exit 2 | 각 1 |

- 합계 20칸 이하로 맞춘다(예: 2 + 4 + 2 + 4 + 4 + 3 = 19). 제출 전 `squeue` 수를 보고, 넘으면 SVR array를 뒤에 `--dependency=afterany`로 단다.
- 파티션은 `sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`로 idle CPU가 많은 쪽을 고른다.
- 로그인 노드에 남는 프로세스(poller, 반복 제출 루프)는 쓰지 않는다. 의존성 사슬과 array만 쓴다.
