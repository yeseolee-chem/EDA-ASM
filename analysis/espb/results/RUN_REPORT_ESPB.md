# ESPB 실행 보고 — Espley-type baseline을 D1 짝 ΔΔ task에 맞춰 학습하고 GNN3R2와 비교

- 지시서 `docs/specs/ESPB.md` · 브랜치 `espb` · 결과 commit `08497f5b`
- scratch `/gpfs/tmp_cpu2/yeseo1ee/espb` (units, NN 작업 폴더), 결과 `analysis/espb/results/`, manifest `results/espb.json`, 사전 등록 `results/PREREG_ESPB.md`
- 대상: GNN3R2 데이터셋(반응 8,594, 짝 8,344, test 짝 8,175), 같은 core 5-fold 분할, 같은 채점 코드(`gnn3.metrics`).

## 1. 단계별 판정

| 단계 | 판정 | 비고 |
|---|---|---|
| 1 시작 조건 | 통과 | 1-1 … 1-9, 상수 확인 2-3 |
| 2 복사 | 통과 | 사본 5개, diff `COPY_DIFF.patch` |
| P 데이터 표 | 통과 | P-1 … P-9 |
| G 사전 등록 | 통과 | commit 뒤 학습, 등록 sha256 9개 = 현재 파일 |
| A 학습 (E-A) | 통과 | unit 125/125, A-1 … A-4 |
| B 학습 (E-B) | 통과 | unit 100/100, B-1 … B-4, M-1 최대 0.0e+00 kcal/mol |
| C 모으기·선택 | 통과 | C-1 … C-4 |
| E 평가·비교 | 통과 | E-0 (GNN3R2 metrics 재현, 최대 차 4.4e-16) |
| F 그림 | 통과 | 4개 × PDF·PNG·CSV |

## 2. 주 비교 (사전 등록, P_main · 위치 all)

| 타깃 | MAE GNN3 | MAE ESPB (고른 후보) | d = ESPB − GNN3 (95 % CI) | p | Holm p | 해석 |
|---|---|---|---|---|---|---|
| strain | 0.813 | 0.936 (EB-SVR) | +0.123 [+0.087, +0.162] | 0.001 | 0.004 | GNN3 lower (Holm p 0.004) |
| inter | 0.638 | 0.795 (EB-SVR) | +0.157 [+0.125, +0.190] | 0.001 | 0.004 | GNN3 lower (Holm p 0.004) |
| solv | 0.509 | 0.610 (EB-SVR) | +0.101 [+0.080, +0.122] | 0.001 | 0.004 | GNN3 lower (Holm p 0.004) |
| barrier | 0.747 | 0.865 (EB-SVR) | +0.118 [+0.082, +0.155] | 0.001 | 0.004 | GNN3 lower (Holm p 0.004) |

- n = 6,978 기준 짝, core 44개. core bootstrap 2000번, seed 20261019. d < 0이면 ESPB가 낮다.

## 3. 보조 비교 (설명용)

**ESPB 대 GNN3** (Holm은 각 행 묶음의 타깃 4개 안에서)

| 집합 | 부분 | 타깃 | n | MAE GNN3 | MAE ESPB | d (95 % CI) | Holm p |
|---|---|---|---|---|---|---|---|
| P_main | para | strain | 1,839 | 0.387 | 0.482 | +0.096 [+0.050, +0.136] | 0.004 |
| P_main | para | inter | 1,839 | 0.387 | 0.491 | +0.104 [+0.070, +0.142] | 0.004 |
| P_main | para | solv | 1,839 | 0.314 | 0.354 | +0.040 [+0.023, +0.060] | 0.004 |
| P_main | para | barrier | 1,839 | 0.360 | 0.422 | +0.063 [+0.023, +0.103] | 0.004 |
| test_all | all | strain | 8,175 | 0.854 | 0.975 | +0.121 [+0.089, +0.159] | 0.004 |
| test_all | all | inter | 8,175 | 0.641 | 0.798 | +0.157 [+0.127, +0.189] | 0.004 |
| test_all | all | solv | 8,175 | 0.501 | 0.611 | +0.110 [+0.091, +0.128] | 0.004 |
| test_all | all | barrier | 8,175 | 0.790 | 0.901 | +0.112 [+0.078, +0.149] | 0.004 |
| test_all | para | strain | 2,124 | 0.404 | 0.500 | +0.096 [+0.054, +0.134] | 0.004 |
| test_all | para | inter | 2,124 | 0.387 | 0.496 | +0.109 [+0.078, +0.145] | 0.004 |
| test_all | para | solv | 2,124 | 0.309 | 0.352 | +0.044 [+0.029, +0.060] | 0.004 |
| test_all | para | barrier | 2,124 | 0.374 | 0.436 | +0.061 [+0.022, +0.099] | 0.004 |
| P_main | p | strain | 1,839 | 0.387 | 0.482 | +0.096 [+0.050, +0.136] | 0.004 |
| P_main | p | inter | 1,839 | 0.387 | 0.491 | +0.104 [+0.070, +0.142] | 0.004 |
| P_main | p | solv | 1,839 | 0.314 | 0.354 | +0.040 [+0.023, +0.060] | 0.004 |
| P_main | p | barrier | 1,839 | 0.360 | 0.422 | +0.063 [+0.023, +0.103] | 0.004 |
| P_main | m | strain | 1,879 | 0.351 | 0.435 | +0.084 [+0.034, +0.134] | 0.004 |
| P_main | m | inter | 1,879 | 0.425 | 0.520 | +0.095 [+0.055, +0.150] | 0.004 |
| P_main | m | solv | 1,879 | 0.298 | 0.347 | +0.049 [+0.022, +0.080] | 0.004 |
| P_main | m | barrier | 1,879 | 0.320 | 0.399 | +0.079 [+0.049, +0.111] | 0.004 |
| P_main | o | strain | 1,708 | 0.921 | 1.052 | +0.131 [+0.057, +0.207] | 0.004 |
| P_main | o | inter | 1,708 | 0.810 | 1.003 | +0.194 [+0.128, +0.255] | 0.004 |
| P_main | o | solv | 1,708 | 0.624 | 0.736 | +0.112 [+0.069, +0.155] | 0.004 |
| P_main | o | barrier | 1,708 | 0.935 | 1.070 | +0.135 [+0.067, +0.203] | 0.004 |
| P_main | direct | strain | 1,552 | 1.761 | 1.953 | +0.192 [+0.057, +0.368] | 0.008 |
| P_main | direct | inter | 1,552 | 1.005 | 1.257 | +0.252 [+0.189, +0.320] | 0.004 |
| P_main | direct | solv | 1,552 | 0.868 | 1.092 | +0.224 [+0.182, +0.263] | 0.004 |
| P_main | direct | barrier | 1,552 | 1.515 | 1.727 | +0.212 [+0.122, +0.314] | 0.004 |
| P_main | R1 | strain | 4,860 | 0.772 | 0.899 | +0.127 [+0.084, +0.171] | 0.004 |
| P_main | R1 | inter | 4,860 | 0.532 | 0.679 | +0.147 [+0.114, +0.190] | 0.004 |
| P_main | R1 | solv | 4,860 | 0.400 | 0.504 | +0.104 [+0.082, +0.127] | 0.004 |
| P_main | R1 | barrier | 4,860 | 0.701 | 0.846 | +0.145 [+0.104, +0.186] | 0.004 |
| P_main | R2 | strain | 2,118 | 0.909 | 1.020 | +0.112 [-0.009, +0.240] | 0.156 |
| P_main | R2 | inter | 2,118 | 0.881 | 1.059 | +0.178 [+0.115, +0.226] | 0.004 |
| P_main | R2 | solv | 2,118 | 0.757 | 0.852 | +0.095 [+0.056, +0.132] | 0.004 |
| P_main | R2 | barrier | 2,118 | 0.852 | 0.909 | +0.057 [-0.043, +0.144] | 0.263 |

**0을 예측하는 기준 대비** (P_main; 개선 = MAE(0) − MAE)

| 부분 | 타깃 | MAE(0) | GNN3 MAE · 개선 | ESPB MAE · 개선 |
|---|---|---|---|---|
| all | strain | 1.599 | 0.813 · 0.786 | 0.936 · 0.663 |
| all | inter | 1.215 | 0.638 · 0.577 | 0.795 · 0.421 |
| all | solv | 0.723 | 0.509 · 0.215 | 0.610 · 0.114 |
| all | barrier | 1.460 | 0.747 · 0.713 | 0.865 · 0.595 |
| para | strain | 0.638 | 0.387 · 0.251 | 0.482 · 0.156 |
| para | inter | 0.519 | 0.387 · 0.132 | 0.491 · 0.028 |
| para | solv | 0.424 | 0.314 · 0.109 | 0.354 · 0.069 |
| para | barrier | 0.670 | 0.360 · 0.310 | 0.422 · 0.248 |

**수축 기울기** (P_main barrier, y에 대한 yhat의 OLS 기울기; 1이면 수축 없음)

| series | GNN3 | ESPB |
|---|---|---|
| all | 0.694 | 0.688 |
| R1 | 0.733 | 0.716 |
| R2 | 0.548 | 0.582 |

**계열 안 장벽 순위** (`M.ranking_table`, P_sens_ref)

| 범위 | 모델 | 계열 수 | top-1 (무작위) | top-3 (무작위) | Spearman 중앙값 |
|---|---|---|---|---|---|
| para | GNN3 | 100 | 0.670 (0.053) | 0.810 (0.160) | 0.904 |
| all | GNN3 | 398 | 0.530 (0.058) | 0.802 (0.174) | 0.802 |
| para | ESPB | 100 | 0.690 (0.053) | 0.840 (0.160) | 0.867 |
| all | ESPB | 398 | 0.447 (0.058) | 0.714 (0.174) | 0.736 |

**부호 정확도** (|y| ≥ 1 kcal/mol 짝, P_main)

| 위치 | 타깃 | GNN3 | ESPB |
|---|---|---|---|
| all | strain | 0.903 | 0.861 |
| p | strain | 0.873 | 0.877 |
| all | inter | 0.936 | 0.884 |
| p | inter | 0.877 | 0.761 |
| all | solv | 0.835 | 0.777 |
| p | solv | 0.842 | 0.770 |
| all | barrier | 0.897 | 0.854 |
| p | barrier | 0.934 | 0.906 |

## 4. 후보 10개

| 타깃 | 후보 | inner 점수 | 순위 | 선택 | test MAE P_main all | para |
|---|---|---|---|---|---|---|
| strain | EB-SVR | 1.002 | 1 | ✔ | 0.936 | 0.482 |
| strain | EB-NN4 | 1.006 | 2 |  | 0.931 | 0.469 |
| strain | EB-KRR | 1.020 | 3 |  | 0.970 | 0.513 |
| strain | EB-NN2 | 1.071 | 4 |  | 0.972 | 0.506 |
| strain | EB-Ridge | 1.266 | 5 |  | 1.179 | 0.636 |
| strain | EA-KRR | 1.300 | 6 |  | 1.257 | 0.753 |
| strain | EA-NN4 | 1.336 | 7 |  | 1.277 | 0.763 |
| strain | EA-NN2 | 1.337 | 8 |  | 1.282 | 0.765 |
| strain | EA-SVR | 1.372 | 9 |  | 1.340 | 0.854 |
| strain | EA-Ridge | 1.745 | 10 |  | 1.651 | 1.117 |
| inter | EB-SVR | 0.798 | 1 | ✔ | 0.795 | 0.491 |
| inter | EB-NN4 | 0.807 | 2 |  | 0.792 | 0.475 |
| inter | EB-KRR | 0.809 | 3 |  | 0.803 | 0.506 |
| inter | EB-NN2 | 0.854 | 4 |  | 0.802 | 0.493 |
| inter | EB-Ridge | 0.888 | 5 |  | 0.870 | 0.489 |
| inter | EA-SVR | 0.892 | 6 |  | 0.880 | 0.523 |
| inter | EA-KRR | 0.896 | 7 |  | 0.883 | 0.531 |
| inter | EA-NN2 | 0.929 | 8 |  | 0.923 | 0.531 |
| inter | EA-NN4 | 0.938 | 9 |  | 0.909 | 0.540 |
| inter | EA-Ridge | 0.995 | 10 |  | 0.964 | 0.571 |
| solv | EB-SVR | 0.619 | 1 | ✔ | 0.610 | 0.354 |
| solv | EB-KRR | 0.628 | 2 |  | 0.619 | 0.366 |
| solv | EB-NN4 | 0.636 | 3 |  | 0.619 | 0.343 |
| solv | EB-NN2 | 0.661 | 4 |  | 0.631 | 0.369 |
| solv | EB-Ridge | 0.684 | 5 |  | 0.682 | 0.374 |
| solv | EA-SVR | 0.712 | 6 |  | 0.710 | 0.447 |
| solv | EA-KRR | 0.715 | 7 |  | 0.712 | 0.451 |
| solv | EA-NN2 | 0.721 | 8 |  | 0.712 | 0.436 |
| solv | EA-NN4 | 0.725 | 9 |  | 0.733 | 0.453 |
| solv | EA-Ridge | 0.801 | 10 |  | 0.787 | 0.479 |
| barrier | EB-SVR | 0.927 | 1 | ✔ | 0.865 | 0.422 |
| barrier | EB-NN4 | 0.936 | 2 |  | 0.860 | 0.435 |
| barrier | EB-KRR | 0.942 | 3 |  | 0.872 | 0.450 |
| barrier | EB-NN2 | 0.990 | 4 |  | 0.863 | 0.409 |
| barrier | EB-Ridge | 1.137 | 5 |  | 1.058 | 0.538 |
| barrier | EA-KRR | 1.162 | 6 |  | 1.130 | 0.691 |
| barrier | EA-SVR | 1.175 | 7 |  | 1.119 | 0.662 |
| barrier | EA-NN2 | 1.241 | 8 |  | 1.185 | 0.701 |
| barrier | EA-NN4 | 1.264 | 9 |  | 1.214 | 0.702 |
| barrier | EA-Ridge | 1.530 | 10 |  | 1.443 | 0.960 |

## 5. Espley 쪽 지표

**E-A 반응별 절대 test MAE** (fold별 MAE의 평균; % = MAE / test 범위)

| 모델 | d1 | d2 | inter | solv | Int | barrier |
|---|---|---|---|---|---|---|
| EA-Ridge | 1.321 (4.8 %) | 1.534 (7.3 %) | 1.204 (5.6 %) | 0.976 (5.4 %) | 0.958 (6.0 %) | 1.895 (5.0 %) |
| EA-KRR | 1.226 (4.4 %) | 0.888 (4.3 %) | 1.019 (4.7 %) | 0.837 (4.7 %) | 0.877 (5.5 %) | 1.468 (3.9 %) |
| EA-SVR | 1.166 (4.2 %) | 0.894 (4.3 %) | 1.043 (4.8 %) | 0.842 (4.7 %) | 0.905 (5.7 %) | 1.456 (3.9 %) |
| EA-NN2 | 1.327 (4.8 %) | 0.970 (4.7 %) | 1.218 (5.6 %) | 0.988 (5.5 %) | 1.003 (6.3 %) | 1.692 (4.5 %) |
| EA-NN4 | 1.178 (4.3 %) | 1.013 (4.9 %) | 1.197 (5.5 %) | 0.905 (5.0 %) | 1.027 (6.5 %) | 1.694 (4.5 %) |

- GNN3는 짝 모델이라 반응별 절대값을 내지 않아 이 표에 없다.

**ML 이전 xTB 대 DFT** (부호 규약 P-8: xtb_e_barrier = xtb_sum_distortion + xtb_interaction)

| 수준 | 타깃 | xTB | n | MAE | 평균 부호 오차 | r |
|---|---|---|---|---|---|---|
| reaction | d1 | `xtb_dist_dipole_kcal` | 8,594 | 7.599 | -7.551 | 0.866 |
| reaction | d2 | `xtb_dist_dipolarophile_kcal` | 8,594 | 3.557 | -3.408 | 0.603 |
| reaction | Int | `xtb_interaction_kcal` | 8,594 | 4.827 | 4.794 | 0.612 |
| reaction | barrier | `xtb_e_barrier_kcal` | 8,594 | 6.950 | -6.164 | 0.656 |
| pair_P_main_test | strain | `delta xtb_sum_distortion_kcal` | 6,978 | 2.042 | -0.174 | 0.210 |
| pair_P_main_test | Int | `delta xtb_interaction_kcal` | 6,978 | 0.786 | -0.027 | 0.583 |
| pair_P_main_test | barrier | `delta xtb_e_barrier_kcal` | 6,978 | 1.881 | -0.202 | 0.229 |

## 6. Espley 원본에서 바꾼 점과 하지 않은 것

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

| 하지 않은 것 | 이유 |
|---|---|
| AM1 원본 descriptor 계산 | 사용자 결정 2026-11-16 |
| feature 효과와 구조 효과를 나누는 arm | 사용자 결정 2026-11-16 |
| XGB | Espley 모델이 아님 |
| GNN3 재학습 | GNN3R2 CV 예측(commit됨)을 그대로 씀 |
| 이전 결과 삭제 | 대체하는 이전 판 없음 |

## 7. 그림 (`results/figures/`)

- `fig1_primary` — 주 비교: P_main에서 타깃별 GNN3와 ESPB의 MAE(core bootstrap 95 % 구간), all과 para
- `fig2_parity` — P_main barrier ΔΔ의 DFT 대 예측(GNN3, ESPB), y = x와 ±1 kcal/mol 띠
- `fig3_candidates` — 타깃별 후보 10개의 P_main MAE와 inner 점수 순위, GNN3 선과 0 예측 선
- `fig4_positions` — P_main 위치(p, m, o, direct)별 MAE, GNN3와 ESPB

## 8. 결정 · 편차 · gate 값

- 결정·편차 전체: `results/DECISIONS.md`.
- P-7 반응별 라벨 닫힘 최대 |barrier − 9개 합| = 0.098 kcal/mol.
- P-8 xTB 규약: |xtb_e_barrier − (xtb_sum_distortion + xtb_interaction)| 최대 7.80e-12, |xtb_sum_distortion − (dipole + dipolarophile)| 최대 0.00e+00.
- P-9 효과 없는 hyperparameter: Ridge tol 0.0e+00, RBF SVR coef0·degree 0.0e+00 (≤ 1e-10).
- 학습 시간: E-A 42.9 CPU-job-h(unit 합), E-B 51.4 h.

