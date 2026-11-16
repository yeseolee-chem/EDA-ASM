# ESPB 사전 등록 (Phase G)

- 지시서: `docs/specs/ESPB.md` · 작성 2026-11-16 16:26:21 · 학습(A·B) 제출 전에 commit한다.
- 학습 스크립트(`ea_train.py`, `eb_train.py`)는 이 파일과 `prereg_espb.json`이 HEAD에 commit되어 있고, json에 적힌 sha256이 현재 파일과 같을 때만 돈다(`prereg.gate`).

## 1. Espley 원본에서 바꾸는 점 (지시서 0-2)

| # | Espley 2024 원본 (ds3) | ESPB | 이유 |
|---|---|---|---|
| X-1 | Coley [3+2], ωB97X-D/def2-TZVP | GNN3R2 데이터셋 8,594 반응 / 8,344 짝, cr2 B3LYP-D3(BJ)/def2-TZVP SMD(water) | 데이터 동일 |
| X-2 | AM1(Gaussian) descriptor 46개 | GFN2-xTB/ALPB(water) ESPLEY46 (DIST11 + MULLIKEN15 + VALENCE15 + E5); APT 대신 Wiberg valence, q_barrier 없음 | 사용자 결정 2026-11-16 |
| X-3 | AM1 TS·GS | g2r xTB TS ($G2R/lf/X182_lf.parquet, GNN3와 같은 파일) | 입력 정보 수준 동일 |
| X-4 | 무작위 80/10/10, seed 22/23/14/1/2 | GNN3R2 core 단위 outer fold 5개 (T1__f<k>.json) | 분할 동일 |
| X-5 | seed 23 train에서 KFold(5) GridSearchCV 한 번 | outer fold마다 GroupKFold(4, group = core) = GNN3 inner split | 튜닝 조건 동일 |
| X-6 | ESI Table S3 grid, KRR polynomial(표)/RBF(본문) | ESI Table S3 ∪ rev 4 grid, KRR RBF; 효과 없는 값(Ridge tol, RBF SVR coef0·degree) 제외(P-9) | 원본보다 좁지 않은 grid |
| X-7 | y 원값 | y 표준화(rev 4 _make_pipe), NN은 r6_nn 그대로 | rev 4에서 원본을 이긴 설정 |
| X-8 | 타깃 5개(dipole·dipolarophile distortion, Interaction, ΔE‡, ΔG‡) | E-A: d1, d2, inter, solv, barrier (Interaction = inter + solv로 나눔, ΔG‡ 없음) | 우리 3채널 |
| X-9 | 반응별 절대값 | 짝 ΔΔ: E-A = f(a) − f(b), E-B = 반대칭 짝 모델(D1PAIR arm C) | task 동일 |
| X-10 | seed별 분할의 평균 MAE | NN 최종 예측 = seed 22, 23, 14, 1, 2 예측의 평균 | GNN3 5-seed 평균과 맞춤 |

## 2. 하지 않는 것 (0-3)

| 항목 | 이유 |
|---|---|
| AM1 원본 descriptor 계산 | 사용자 결정 2026-11-16 |
| feature 효과와 구조 효과를 나누는 arm | 사용자 결정 2026-11-16 |
| XGB | Espley 모델이 아님 |
| GNN3 재학습 | GNN3R2 CV 예측(commit됨)을 그대로 씀 |
| 이전 결과 삭제 | 대체하는 이전 판 없음 |

## 3. 모델 · grid · 전처리 · seed (5절)

**E-A (각각 예측하고 빼기, `ea_train.py`)** — 반응별 타깃 d1, d2, inter, solv, barrier. `train_ml_single._make_pipe(est)` = X StandardScaler → est, y 표준화. GridSearchCV(cv = P-6의 core inner fold 4개, neg MAE, refit=False), 최적 = 평균 inner MAE 최저(동률: grid 순서 앞). 최종 fit = 학습 반응 전부.

| 모델 | grid | 칸 |
|---|---|---|
| Ridge | alpha [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000] | 16 |
| KRR | alpha [1e-05, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] × gamma [None, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.9, 1] | 154 |
| SVR | C [1, 10, 30, 50, 100, 1000] × gamma ['scale', 'auto', 0.001, 0.01, 0.1] × epsilon [0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1] | 240 |

- NN-2 / NN-4: `r6_nn` 그대로. tune = seed 23, inner fold 1 train → val (Hyperband `r6_nn.HB`, `HP_STORAGE`). cv = seed 22, inner fold마다 새 프로세스로 fit_predict(validation_data = 그 inner val) → OOF. final = seed [22, 23, 14, 1, 2] 각각 새 프로세스, 학습 반응 전부, validation_data = inner fold 1 val(history 전용), 예측 = 다섯 seed 평균.
- 짝 예측: strain = [f_d1 + f_d2](a) − [f_d1 + f_d2](b), inter·solv·barrier = f(a) − f(b). NN은 seed별 ΔΔ의 평균, 표준편차 = yhat_seed_sd.

**E-B (차이를 직접 학습, `eb_train.py`)** — 짝 타깃 strain, inter, solv, barrier(채널 ΔΔ). `d1_train` 그대로: z = [Δx / s_Δ, (x̄ − μ) / s], y / s_y, 반대칭 Ridge(Δx만, 절편 없음) · KRR · SVR(반대칭 곱 커널, 양방향) · NN(공유 h).
- grid = `d1_train.grid_cells`: Ridge alpha [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] (8칸); KRR gamma_Δ × gamma_x̄ ∈ [3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]², alpha [1e-05, 0.0001, 0.001, 0.01, 0.1, 1] (384칸); SVR gamma_Δ × gamma_x̄ ∈ [3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]², C [1, 10, 100, 1000], epsilon [0.05, 0.1, 0.5] (768칸).
- 튜닝 = GroupKFold(4, core) (= P-6), 최적 = 평균 inner MAE 최저. NN: tune(inner fold 1, seed 23), cv(inner 4개, seed 22, 한 프로세스), final = seed [22, 23, 14, 1, 2] 평균. gate M-1(반대칭) ≤ 1e-06 kcal/mol, test 짝 200개, NN은 seed마다.

## 4. 대표 선택 규칙 (7-2, test 점수 미사용)

- 후보: 타깃마다 {EA, EB} × {Ridge, KRR, SVR, NN2, NN4} = 10개.
- 점수 = inner ΔΔ MAE: outer fold k마다 inner fold 4개의 'inner val 짝 전체에 대한 MAE'를 평균하고, 그 값을 5개 fold에 대해 평균. E-A는 inner OOF 반응 예측으로 inner val 짝의 ΔΔ를 만들어 계산, E-B는 최적 칸의 inner fold별 MAE(NN은 cv 단계).
- 고르기 = 점수 최저. 동률이면 EA → EB, Ridge → KRR → SVR → NN2 → NN4 순서의 앞 것.
- `collect.py`는 json과 OOF 행만 읽어 `C_selection.csv`를 먼저 쓰고(sha256 기록), 그 뒤에 test 예측을 읽는다(C-4).

## 5. 비교 (8-2)

- **주 비교(확증):** 집합 P_main(기준 짝), 위치 all, 타깃 4개. d = MAE(ESPB) − MAE(GNN3). `gnn3.metrics.bootstrap_mae_difference(err_ESPB, err_GNN3, core)`, core bootstrap 2000번, seed 20261019, 95 % CI와 양측 p. **Holm 보정 범위 = 이 4개 타깃.**
- **보조(설명용):** P_main para, test_all(all, para), P_main 위치(p, m, o, direct), series(R1, R2); 후보 10개 각각 대 GNN3(P_main all, para); 0 예측 기준 MAE와 개선폭; 수축 기울기(P_main barrier, yhat의 y에 대한 OLS 기울기, series별). Holm은 각 묶음(같은 비교·집합·부분집합의 타깃 4개) 안에서 따로 표시한다.
- 표: `M.error_table`, `M.ranking_table`, `M.cause_agreement`, `M.breakdown_table`(GNN3, ESPB); Espley 쪽 `E_native.csv`(반응별 절대 MAE와 % range), `E_preml.csv`(ML 이전 xTB).
- 채점 코드 동일성 E-0: GNN3R2 CV 예측을 같은 코드로 다시 채점한 표 = `analysis/gnn3r2/results/eval/metrics.csv` (≤ 1e-12).

## 6. 등록 파일 sha256

| 파일 | sha256 |
|---|---|
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-espb/analysis/espb/eb_common.py` | `51660eb1b322fa4d2f5788bc3e2fcce1a9a208d6023a7c69150f4e23e1f2053a` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-espb/analysis/espb/ea_train.py` | `a50db4cdb14253bd11c4c7d6b05a150de035e265addfcd406ab7646672c86526` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-espb/analysis/espb/eb_train.py` | `8d13bdf846d3cf1b0928f11a0398a34e25136aad5f7c19b0075e098dfc590eb3` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-espb/analysis/espb/collect.py` | `23f06897d90b6aa4dfbc673d45e3b2096a72151ed3c5912c7c31f995a861cdea` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-espb/analysis/espb/compare.py` | `71ba9a0048d92e170a94268ef6935b0bc2083f84cacb4c865a2981da792285e5` |
| `/gpfs/tmp_cpu2/yeseo1ee/espb/P/X46.parquet` | `e9163ce9586e17be3ad937c7b4932fc2c9ea144e7419f5f5c7feec24b26c89b4` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-espb/analysis/espb/results/P_reactions.csv` | `085306551a5621fec2e7c017c589c7c0639dc066bd5cc96d236e219feb6bff1e` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-espb/analysis/espb/results/P_inner_folds.csv` | `cf94c9c02b34615312bff7fb4928f994cd4f97e3416b45db8ccac718733255ee` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-espb/analysis/espb/results/P_report.json` | `25547e3b382d6e923b98e2283720fb17624fa460b66764be70b6caff3b9a2af9` |

## 7. 복사 기록 (2-2, 2-3)

| 원본 | 사본 | 원본 sha256 | 사본 sha256 | 바뀐 줄 |
|---|---|---|---|---|
| `analysis/d1_pair/d1_f0.py` | `analysis/espb/ea_train.py` | `6cc918984cd6…` | `a50db4cdb142…` | 427 |
| `analysis/d1_pair/d1_train.py` | `analysis/espb/eb_train.py` | `414b5cc66762…` | `8d13bdf846d3…` | 402 |
| `analysis/d1_pair/jobs/env.sh` | `analysis/espb/jobs/env.sh` | `7d326a4d0c65…` | `ccd70d036f13…` | 2 |
| `analysis/d1_pair/jobs/py.slurm` | `analysis/espb/jobs/py.slurm` | `68d692982b73…` | `9c4e9df67234…` | 2 |
| `analysis/gnn3r2/gr_common.py` | `analysis/espb/eb_common.py` | `d11d5e2a1b80…` | `51660eb1b322…` | 202 |

diff 전체: `results/COPY_DIFF.patch`. 결정·편차: `results/DECISIONS.md`.
