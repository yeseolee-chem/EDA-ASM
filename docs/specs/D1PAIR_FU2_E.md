# D1PAIR FU2 Phase E·R — 공식 평가와 보고 — Claude Code 실행 지시서

**대상:** 브랜치 `d1-pair` `2d745d23` (`analysis/d1_pair/`). 이어서 commit한다.
**이 파일:** `docs/specs/D1PAIR_FU2_E.md`로 commit한다.
**새 코드:** `fu2/evaluate.py`, `fu2/tests/test_eval.py`. **결과:** `results/fu2/E_*`, `results/fu2/figures/`, `results/fu2/SUMMARY_FU2.md`.
**규칙의 근거:** `results/PREREG_D1PAIR_ADD2.md`(이하 ADD2)가 정한다. 이 문서는 구현 방법만 더한다. 둘이 다르면 **ADD2를 따르고 보고**한다.
**순서:** E-T(코드 + 검사 → commit → **멈춤: 사용자 코드 검토**) → E-0 → E-1 → E-2 → E-7 → E-3 → E-4 → E-5 → 그림 → E-6 → R
**보고:** 단계마다. STOP이면 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다.

---

## 0. 무엇을 하나

Phase M은 시험을 치른 단계다(30 unit, 예측 `results/fu2/M_predictions.parquet`). Phase E는 **ADD2에 미리 적어 둔 채점 기준으로 공식 채점**하는 단계다.

- 대표 구성을 고르고(E-1), C-XGB와 주 비교로 판정하고(E-2), 부 비교·Hammett 지표를 내고(E-3·E-4), 기술 통계를 붙인다(E-5).
- 검토자가 같은 규칙으로 따로 계산한 값이 있다(가채점, 8절). 공식 값이 이와 같은지 확인한다(E-7).
- LODO·LOSO(M-4)는 이 문서에 넣지 않는다. 그것은 ADD3 사전 등록 뒤 별도 명령서로 한다.

## 1. 입력

| 무엇 | 파일 | 확인 |
|---|---|---|
| 원자 모델 예측 | `results/fu2/M_predictions.parquet` | 행 472,440, NaN 0 |
| unit 기록 | `$D1PAIR_SCRATCH/fu2/units/T1__<model>__<variant>__f<k>.json` (30개) | E-0 지문 |
| 원자 모델 가중치 (E-6만) | 같은 폴더 `…__s<0-4>.pt` | 존재 |
| C-XGB 예측 | `$D1PAIR_SCRATCH/M/units/T1/T1__C__XGB__<타깃>__f<k>.parquet` (50개) | ADD2 json `comparator_sha256` |
| C 대표 (rev 9, SVR 포함 6개) | `results/E_representative.json`의 `T1|C|<타깃>` + 해당 모델의 rev 9 unit parquet | sha256 기록 |
| XGBW (FU1) | `$D1PAIR_SCRATCH/M/units_fu/XW/` | sha256 기록 |
| 짝·분할 | `results/S_folds.csv`, `results/F_pairs.csv`, `results/fu2/S_splits.json`, `$D1PAIR_SCRATCH/fu2/splits/pairs.parquet` | ADD2 1절 sha256 |
| 사전 등록 | `results/PREREG_D1PAIR_ADD2.md`, `prereg_d1pair_add2.json` | md sha256 = json `md_sha256` |

## 2. 코드 (`fu2/evaluate.py`)

명령: `check`(E-0) · `rep`(E-1) · `primary`(E-2, E-7) · `secondary`(E-3) · `hammett`(E-4) · `describe`(E-5) · `figures` · `explain`(E-6) · `all`(= check → rep → primary → secondary → hammett → describe → figures).

**새로 만들지 말고 가져다 쓸 것 (같은 함수 = 같은 채점):**

| 용도 | 함수 |
|---|---|
| bootstrap d, CI, p | `d1_eval.boot_diff` (B 2,000, seed 20261019) |
| Holm | `d1pair_common.holm` |
| Clopper–Pearson | `d1_eval.clopper` |
| R²_fam, ρ 오차 | `d1_eval.family_eval`, `d1_eval.hammett_frame` |
| 꺾임·V 판정 | `hammett_eval.family_series`, `hammett_eval.calls` (MIN_X_FAM 12) |
| 학습·검증 짝, s_y | `fu2.train.Data`, `Data.split` 그리고 `fit()`이 s_y를 구할 때 쓰는 **같은 코드 경로** |

- `d1_eval.bends`는 arm B 대표로 고정되어 있다. 그 반복문 본문을 그대로 옮긴 `bends_tab(d)`를 만들고, T-2로 같음을 확인한다.
- 비교 함수 하나로 통일: `compare(x, b, pairs)` → `boot_diff(e_x, e_b, cores)`. **x = 비교 상대, b = 원자 쪽**이다. 그래서 d > 0이면 원자 쪽이 낫다(ADD2 4절과 같은 부호).
- 모든 쓰기는 원자적으로, 다시 돌려도 같은 결과(idempotent).

## 3. E-T — 코드 검사 (실행 전; FU2 시험 점수는 계산하지 않는다)

`fu2/tests/test_eval.py`를 sbatch로 돌린다. rev 9 결과와 합성 데이터만 쓴다.

| # | 검사 | 통과 기준 |
|---|---|---|
| T-1 | 환경: `env_d1fu2`에서 numpy, pandas, scipy, sklearn, pyarrow, matplotlib, torch, rdkit import | 모두 됨. 없으면 설치하지 말고 보고 |
| T-2 | `bends_tab`에 rev 9 arm B 대표(T1) 예측을 넣음 | `results/E_bends.csv` T1 행과 모든 열이 같음 |
| T-3 | `family_eval` 경로에 rev 9 arm B 대표를 넣음 | `results/E_metrics.csv`의 r2_fam_median, rho_err_median과 같음 (1e-12) |
| T-4 | s_y 경로: unit 30개 각각, 바깥 학습 짝으로 s_y를 다시 계산 | unit json `final[0].sy`와 같음 (1e-12). 같은 함수로 튜닝 학습 짝의 s_y를 구한다 |
| T-5 | 안쪽 분할 크기 | 30 unit 모두 json `n_inner_train`·`n_inner_val` = `S_splits.json` 값 |
| T-6 | `compare` 부호·p | 합성: x = b이면 d = 0, p = 1; b가 모든 짝에서 오차 0.1 작으면 d = 0.1, CI 하한 > 0 |
| T-7 | 짝 목록 gate (음성 검사) | 한 모델에서 짝 하나를 지우면 STOP(exit 3) |
| T-8 | 대표 동점 규칙 | 합성 동점에서 AW-A → AW-A+G → GNN-A → GNN-A+G → TF-A → TF-A+G 순서의 앞 것 |

통과하면 commit·push하고 보고한다(commit hash, 검사 결과, `evaluate.py` sha256). **여기서 멈추고 사용자 검토를 기다린다**(ADD2 9절: 평가 코드와 대표 고르기 코드는 실행 전에 검토).

## 4. E-0 — 입력 gate (`check`)

- ADD2: md sha256 = json `md_sha256`, 두 파일이 commit 뒤 바뀌지 않음.
- unit 30개: 지문 `code_sha256` = json `code_sha256_fingerprint_keys`, 지문 `add2` = ADD2 md sha256.
- C-XGB parquet 50개 sha256 = json `comparator_sha256`.
- 시험 짝 목록: 원자 데이터 없는 짝을 모든 모델에서 똑같이 뺀 뒤, 원자 구성 6개 · C-XGB · C 대표 · XGBW · null의 (fold, pair_id) 집합이 같음.
- unit별 GPU 이름을 `E_gpu.csv`로 낸다(보고만).
- 결과 `E_check.json`. 하나라도 다르면 **STOP**.

## 5. E-1 — 대표 구성 (`rep`, ADD2 3절)

1. 구성 6 × fold 5마다, unit json `chosen.index` 칸의 `tune[].val_per_target`(표준화)을 읽는다.
2. 그 fold 튜닝 학습 짝(첫 안쪽 fold의 학습 쪽, 원자 있는 짝)의 타깃별 표준편차(ddof 0, T-4의 같은 함수)를 곱해 kcal/mol로 바꾼다.
3. 타깃마다 fold 5개 평균이 가장 낮은 구성 = 대표. 동점은 T-8 순서.

출력: `E_representative.json`(타깃 → 구성), `E_val_kcal.csv`(구성 × fold × 타깃의 검증 MAE kcal/mol). **시험 점수는 읽지 않는다.**

## 6. E-2 — 주 비교 (`primary`, ADD2 4절)

- 타깃 9개(d1, d2, elst, pauli, oi, disp, cpcm, cds, barrier). barrier는 모델이 직접 낸 값.
- 시험 짝: T1 fold 0–4 시험 짝 중 `in_P_main` & `ref_kind == "reference"`, 다섯 fold 합침.
- d = MAE(C-XGB) − MAE(원자 대표), `boot_diff`, Holm 9개.
- 판정: Holm p < 0.05 이고 CI 하한 > 0 인 타깃 수 → 5개 이상 "원자 입력이 182개 feature보다 낫다", 2–4개 "일부 채널에서만", 1개 이하 "이득 없음". 나빠진 타깃(Holm p < 0.05, CI 상한 < 0) 수도 적는다.
- 출력 `E_primary_tests.csv`: target, rep, n_pairs, n_cores, mae_CXGB, mae_atom, d, ci_lo, ci_hi, p, p_holm, better, worse, folds_atom_lower(fold별 MAE에서 원자가 낮은 fold 수).

## 7. E-7 — 가채점 대조 (E-2 바로 뒤)

공식 E-1·E-2 값을 8절 값과 비교한다.

- 대표 구성: 10개 타깃 모두 같아야 한다.
- n_pairs, MAE, d: 차이 ≤ 1e-6. ci_lo, ci_hi, p: 같은 bootstrap이라 차이 ≤ 1e-9.
- **다르면 STOP**하고 두 값을 나란히 보고한다. 값을 맞추려고 코드를 고치지 않는다.
- 출력 `E_crosscheck.csv`.

## 8. 가채점 값 (검토자, 2026-10-21, `FU2_RESULTS_JUDGMENT.md`)

대표: barrier, d1, d2, disp, cpcm, cds = **GNN-A+G** · elst, pauli, oi, c_ghost = **AW-A+G**.

| 타깃 | 대표 | n | MAE C-XGB | MAE 원자 | d | ci_lo | ci_hi | p | p_holm |
|---|---|---|---|---|---|---|---|---|---|
| d1 | GNN-A+G | 6884 | 0.456793 | 0.414311 | 0.042482 | 0.017794 | 0.064004 | 0.001999 | 0.009995 |
| d2 | GNN-A+G | 6884 | 0.339738 | 0.328760 | 0.010977 | −0.005557 | 0.029202 | 0.190905 | 0.381809 |
| elst | AW-A+G | 6884 | 1.127615 | 1.018368 | 0.109247 | 0.060284 | 0.175941 | 0.001000 | 0.008996 |
| pauli | AW-A+G | 6884 | 2.156280 | 1.937573 | 0.218707 | 0.114549 | 0.351784 | 0.001000 | 0.008996 |
| oi | AW-A+G | 6884 | 1.238167 | 1.138605 | 0.099562 | 0.050517 | 0.158294 | 0.001000 | 0.008996 |
| disp | GNN-A+G | 6884 | 0.176034 | 0.175996 | 0.000038 | −0.009202 | 0.009100 | 0.965517 | 0.965517 |
| cpcm | GNN-A+G | 6884 | 0.401902 | 0.381013 | 0.020889 | 0.007183 | 0.034424 | 0.006997 | 0.027986 |
| cds | GNN-A+G | 6884 | 0.079739 | 0.076301 | 0.003438 | 0.000099 | 0.006998 | 0.044978 | 0.134933 |
| barrier | GNN-A+G | 6884 | 0.554189 | 0.465983 | 0.088206 | 0.064392 | 0.111542 | 0.001000 | 0.008996 |

가채점 판정: 개선 6개(d1, elst, pauli, oi, cpcm, barrier), 악화 0개 → "원자 입력이 182개 feature보다 낫다".

## 9. E-3 — 부 비교 (`secondary`, ADD2 5절, 보정 없음)

모두 T1, P_main 시험 짝(위와 같은 짝), `compare(x = 상대, b = 원자 쪽)`, 타깃 9개.

| 비교 | 쌍 |
|---|---|
| 모델 사이 (같은 변형) | AW 대 GNN, AW 대 TF, GNN 대 TF × 변형 A, A+G |
| 변형 사이 | 모델마다 A+G 대 A (182개가 더하는 몫) |
| 원자가 더하는 몫 | 모델마다 A+G 대 C-XGB |
| para 짝만 | 원자 대표 대 C-XGB (`pos == "p"`) |
| C 대표 | 원자 대표 대 rev 9 C 대표(SVR 포함 6개 중; SUMMARY에 적음) |
| XGBW | 원자 대표 대 XGBW |
| null | 원자 대표 대 ΔΔ = 0 |

출력 `E_secondary_tests.csv`: comparison, x, b, target, set, n_pairs, mae_x, mae_b, d, ci_lo, ci_hi, p.

## 10. E-4 — 지표와 Hammett 수준 (`hammett`, ADD2 7절)

대상 모델: 원자 구성 6개, 원자 대표, C-XGB, C 대표, XGBW, null.

- `E_metrics.csv`: 짝 집합 P_main · A_main(anchor) · P_sens · P_main_no_d0_overlap × 위치 all · p · m · o · direct → n, MAE, RMSE. P_main all 행에 R²_fam 중앙값·사분위, ρ 오차 중앙값(`family_eval`).
- `E_bends.csv`(계열 × 타깃 × 모델, `bends_tab`)와 `E_bend_rates.json`: 민감도·위양성률·V(Clopper–Pearson 95 %). 전체 계열, 숨은 V 6계열(`hidden_V_fam`), 숨은 꺾임 30계열(`hidden_bend_fam`) 각각.
- `E_stages_descriptive.csv`: rev 9 단계 기준을 **설명용으로만** 적용(판정 아님) — 단계 1 R²_fam 중앙값 ≥ 0.5; 단계 2(oi, cpcm) para MAE ≤ 0.13, 민감도 ≥ 0.7, 위양성률 ≤ 0.1. 원자 대표와 C-XGB.
- `E_target_multiples.csv`: para MAE ÷ ADD2 7절 목표 MAE(barrier 0.16, d1 0.19, d2 0.04, elst 0.21, pauli 0.41, oi 0.29, cpcm 0.22, cds 0.01, disp 0.05). 파일과 SUMMARY에 "사후 정의"라고 적는다.

## 11. E-5 — 기술 통계 (`describe`; ADD2 밖, 사후 추가, 판정에 쓰지 않음)

| 출력 | 내용 |
|---|---|
| `E_epochs.csv` | 튜닝 칸 120개: model, variant, fold, cell, best_epoch, epochs_run, val, chosen. 요약: best_epoch > 270인 칸 수, epochs_run = 300인 칸 수 |
| `E_seed_sd.csv` | 구성 × 타깃별 `yhat_seed_sd` 중앙값·90 %·99 % |
| `E_axis.json` | P_main 시험 짝에서 (elst, pauli, oi)의 정답 ΔΔ 주성분(적재, 분산 비중)과 원자 대표 오차의 같은 축 분산 비중; corr(pauli, elst + oi) 정답·오차; DFT 꺾임이 세 채널에서 같이 나오는 계열 수(pauli 꺾임 중 elst 또는 oi도 꺾인 수) |

가채점 참고값: best_epoch 최대 266, > 270 칸 0개; 첫 주성분 비중 정답 0.979, 오차 0.965; pauli 꺾임 22계열 중 21계열 동반.

## 12. 그림 (`figures`, `results/fu2/figures/`, png + pdf + csv, 설명은 `CAPTIONS_FU2.md`)

| 그림 | 내용 |
|---|---|
| `figF2_1_mae` | 타깃 × [null, C-XGB, XGBW, C 대표, AW, GNN, TF (각 A, A+G)] 상대 MAE(÷ null), 전체·para 두 판 |
| `figF2_2_hammett_level` | R²_fam 중앙값, 꺾임 민감도·위양성률(CI 막대), 원자 대표 대 C-XGB |
| `figF2_3_examples` | rev 9 fig3 계열 중 3개 × barrier·oi·elst: DFT, C-XGB, 원자 대표 |
| `figF2_4_atoms` | E-6 해석 그림 |
| `figF2_5_epochs` | 튜닝 칸 120개의 best_epoch, 상한 300·270 선 |

rev 9 `d1_figures.py`와 같은 글꼴·색 규칙을 쓴다. 수치는 csv에서 읽는다.

## 13. E-6 — 해석 (`explain`, 탐색)

- 계열: rev 9 `results/figures/fig3_hammett.csv`의 6계열.
- AW 기여 지도: 그 계열이 시험인 fold의 AW-A+G 최종 모델 5개(seed)로 `AW.atom_contrib`를 구해 seed 평균. a와 b의 원자별 ε를 분자 그림(RDKit 2D)에 색으로, 채널 oi·elst·pauli.
- TF cross-attention 지도: TF-A+G 최종 모델의 cross-attention 블록(1개, a→b와 b→a가 가중치 공유). attention 가중치는 forward hook으로 읽는다(`BiasedMHA`의 `return_attn`; `models.py`는 바꾸지 않는다). head·seed 평균.
- 0으로 바꾸기 검사: X 작용기 원자의 dq_dist, dq_int, dv_dist, dv_int를 원래 단위에서 0으로 바꾸고 같은 표준화를 거쳐 예측 변화 |Δŷ|를 채널별로 낸다. 대조 = 같은 수의 무작위 비반응 원자. X 원자 찾는 방법은 E-T 보고에 적어 검토받는다.
- ADD2 "알려진 점"의 heavy RMSD > 1 Å 반응을 그림에 표시한다.
- 그림 설명에 "attention·기여는 설명이 아니라 진단"이라고 적는다.
- 출력 `E_explain.csv`, `figF2_4_atoms`.

## 14. Phase R — `results/fu2/SUMMARY_FU2.md` (한국어)

- 맨 앞 세 문장: Q·T 결과(반응 수), 주 비교 판정, 가장 나은 구조.
- 절: Q → T → A(ADD2 commit) → M(구성별 표, gate M-1′, GPU) → E(대표, 주 비교, E-7 대조, 부 비교, Hammett, 기술 통계, 해석) → 이탈 → 한계.
- 반드시 적을 한계:
  - T1은 이미 여러 번 본 분할이다(새 모델은 시험 짝을 튜닝에 쓰지 않았다).
  - 구조 출발점이 DFT TS(G1)다.
  - 원자별 기여와 attention은 해석이 아니라 진단이다.
  - elst·pauli·oi 대표(AW-A+G)는 전 위치 검증 MAE로 골랐다. para 위치 성능은 따로 적는다(`E_metrics.csv`).
  - 목표 MAE 배수와 E-5는 사후 정의다.
- 수치는 모두 파일에서 옮긴다.

## 15. 실행 (HPC)

- python은 모두 `sbatch fu2/jobs/fu2.slurm`(`--export=ALL,PYENV=fu2` = env_d1fu2), `--time=48:00:00`, `-p cpu1,cpu2`(GPU 필요 없음; E-6도 CPU). 로그인 노드에서는 편집·git·`squeue`·`scancel`·`sbatch`만. MaxJobs 10, MaxSubmit 20.
- job 순서: ① E-T(`fu2/tests/test_eval.py`) → **검토 대기** → ② `evaluate.py all`(E-0–E-5, E-7, 그림 1·2·3·5) → ③ `evaluate.py explain`(E-6, 그림 4) → ④ SUMMARY 작성·commit.
- ②와 ③은 의존성 사슬(`--dependency=afterok`)로 낼 수 있다.

## 16. 규칙

- 바꾸지 않는 것: ADD2·PREREG·ADD1, `fu2/` 학습 코드(`models.py`, `train.py`, `fu2_common.py`, `tokens.py`), `$D1PAIR_SCRATCH/fu2/units/`, rev 9 `M/units/`, rev 9·FU1 결과 파일.
- 새 파일은 `fu2/evaluate.py`, `fu2/tests/test_eval.py`, `results/fu2/E_*`, `results/fu2/figures/`, `results/fu2/SUMMARY_FU2.md`, 이 명령서에만.
- 결과를 본 뒤 대표 규칙·비교·판정 규칙을 바꾸지 않는다. 바꾸면 이탈로 기록한다.

## 17. STOP

| 단계 | 조건 |
|---|---|
| E-T | T-1–T-8 중 하나라도 실패; 사용자 검토 전에 E-0 이후 실행 |
| E-0 | ADD2 sha256, unit 지문(`code_sha256`, `add2`), C-XGB sha256 불일치; 모델 사이 시험 짝 목록 불일치 |
| E-1 | 대표를 고를 때 시험 점수를 읽음; 검증 MAE NaN |
| E-7 | 가채점과 다름 (8절 허용 오차 밖) |
| 공통 | 입력 파일 없음; 결과 파일에 NaN(정의상 NaN인 칸 제외, 이유를 적음) |
