# PREREG DEPLOY — 라벨 없는 g2 입력(lf) · S · TF 배포 모델 사전 등록 (2026-11-03)

이 문서는 DEPLOY-MODEL의 사전 등록이다(`docs/specs/DEPLOY_MODEL.md` 10절, sha256 `20cae1dea92526e56859852de6797f503b6b7492222a52866c0a56aa8d29ae60`). `analysis/deploy/prereg_deploy.py`(deploy-prereg-1)가 결과 파일만 읽어 만들었다(git HEAD `bddcc9e39dd5be1544f7873b6634bca289387831`, 작성 2026-11-03 16:29:29). 규칙은 명령서를 그대로 인용한다(인용 블록 = 원문). **K 학습(`train_tf.py run`)과 W(`train_tf.py final`)는 이 md와 `prereg_deploy.json`이 commit되고 바뀌지 않았을 때만 돈다**(`prereg_deploy.py check`).

## 1. 설계 (0-2절)와 사용 가능 규칙 (8절)
> ### 0-2. 확정된 설계 (사용자 결정 2026-11-03)
>
> | 항목 | 값 |
> |---|---|
> | 구조 | g2 = SMILES → autodE 1.4.5 + `XTB2 ALPB(water)`, **반응당 1회** (G2c 3회 아님) |
> | 입력 arm | **S** — 원자 토큰 + 전역 182열. HF 없음 |
> | 모델 | **TF 단독** (FU2 M-2 TF, 변형 A+G) |
> | 학습 데이터 | **D1만** (D0 없음 — LADDER g2와 비교 가능하게) |
> | 라벨 | `$LAD/F_pairs_cr2.csv` (LADDER와 같은 파일, 같은 y) |
> | 분할 | FU2 T1 5 fold (LADDER와 같은 파일) |
> | 배포 모델 | **전체 D1으로 한 번 학습** (설정 = 교차검증 선택의 최빈값, epoch = 최적 epoch 중앙값; GEO1b Y 모델 선례), seed 5개 |

> ## 8. 사용 가능 반응 · 짝 (`analysis/deploy/data_lf.py`)
>
> ### 8-1. 사용 가능 규칙
>
> 반응 사용 가능 ⇔ lf 토큰 있음(T-a 통과) **그리고** `X182_lf`에 행이 있고 182열 모두 유한. **HF는 요구하지 않는다.**
> 짝 사용 가능 ⇔ a와 b가 모두 사용 가능. 이 규칙으로 걸러낸 짝 집합 = `P_lf`.
>
> ### 8-2. `DeployData(FT.Data)`
>
> `fu2/train.py`의 `Data` 하위 클래스. **`train_atoms.LadData`가 S arm에서 만드는 속성과 같은 속성을 같은 방식으로**
> 만든다(`onehot`, `bin`, `num`, `pos`, `mask`, `etype`, `frag`, `n`, `glob`, `y`, `ra`, `rb`, `P`, `d_in`, `d_glob`,
> `static()` 상속, `split(k)`). 차이는 사용 가능 규칙(H 없음)과 토큰 · 182 경로뿐이다.
>
> - 라벨 = `$LAD/F_pairs_cr2.csv`의 `dd_<t>` (10 타깃).
> - 분할 = FU2 T1 파일. `split(k)` → `train`(바깥 학습 T_main), `test`(P_sens ∪ A_main), `inner_train`, `inner_val`(첫 내부
>   `GroupKFold(4)` fold). 사용 불가 짝은 네 목록에서 모두 뺀다(LADDER 규칙).
> - 누출 검사(`LadData.split`과 같음): 시험 짝이 학습에 없음, 내부 fold가 학습을 분할함, 시험 반응이 학습 반응에 없음.
> - `split("all")` (W 단계 전용): 사용 가능한 `in_T_main` 짝 전부(fold −1 core 포함), 시험 · 검증 목록 없음.
>
> ---

## 2. 학습 규칙 (11-2절), 최종 모델 (13절), 평가 (12절)
> ### 11-2. 학습 규칙 (FU2 M-1 · LADDER S arm 그대로; `fu2/train.py`의 `fit`, `Std`, `gate_m1`, `predict`, `batch`, `epoch_batches`를 import해서 수정 없이 호출)
>
> | 항목 | 값 |
> |---|---|
> | 모델 | `Mo.build("TF", cell, d_in=29, n_out=10, d_glob=182, "A+G")` |
> | grid | 층 ∈ {2, 3} × 폭 ∈ {64, 128} (4칸) |
> | 튜닝 | `inner_train`으로 학습, `inner_val`로 선택. 지표 = 주요 9개 타깃 MAE / SD(y_train) 평균. 동률은 grid 순서 |
> | 최종 (fold) | 바깥 학습 짝 전부, 튜닝 최적 epoch에서 멈춤(같은 cosine 일정), seed 20261019 + 0..4, 예측 평균 |
> | 표준화 | 토큰 수치열 · 전역 토큰 = fit 학습 반응의 평균 · SD. y / SD(y_train) (타깃별, ddof 0, 평균 제거 없음) |
> | 손실 | 타깃별 평균 \|ŷ − y\| (표준화) + 0.1 × 평균 \|ŷ_barrier − Σ₉ ŷ\| / s_barrier |
> | 최적화 | AdamW lr 1e-3, wd 1e-4, cosine 300 epoch, dropout 0.1, batch 64 짝, patience 30, 튜닝 seed 20261019 |
> | 증강 | epoch마다 짝마다 (a, b)/(b, a) 순서 무작위, y 부호 함께 |
> | 관문 | 모든 fit에 C-4 |

> ## 13. Phase W — 최종 배포 모델 (`train_tf.py final`)
>
> G-C 통과 후에만 실행한다.
>
> | 항목 | 규칙 |
> |---|---|
> | 학습 짝 | `DeployData.split("all")` — 사용 가능한 `in_T_main` 짝 전부(5 fold의 시험 core 포함, fold −1 core 포함) |
> | 칸 | 5 fold 선택 칸의 **최빈값**. 동률이면 fold별 튜닝 기록의 그 칸 검증 지표 평균이 가장 낮은 칸, 그것도 같으면 grid 순서 |
> | epoch | 5 fold 최적 epoch의 **중앙값** (5개라 정수). `fu2/train.py` `fit`의 `best_epoch`는 1-기반 epoch 번호이고(`for ep in range(1, …)`), fold 최종 fit은 `n_epochs=best_epoch`로 그만큼 학습한다(`run_unit` 399–405행). 그래서 `n_epochs = median(best_epoch)`. GEO1b Y 모델의 `floor(median(best_ep + 1))`는 0-기반 번호에 대한 같은 규칙이다 |
> | seed | 20261019 + 0..4 |
> | 참고 (예상) | LADDER g2 · S · TF에 이 규칙을 대면: 칸 2층 × 64 (f1, f2), epoch 47 (70, 39, 47, 57, 25의 중앙값). lf 결과가 다르면 그대로 따른다 |
> | 표준화 | 학습 짝 전체의 반응으로 통계 (각 `.pt`에 `mu, sd, gmu, gsd, sy`로 저장 — `run_unit`과 같은 형식) |
> | 관문 | C-4 (무작위 학습 짝 200개로) |
> | 산출 | `$DEP/model/D__lf__S__TF__all__s{0..4}.pt`, `$DEP/model/MODEL_CARD.json`, `results/W_report.json` |
>
> `MODEL_CARD.json`: 칸, epoch, seed, 학습 짝 수 · sha256, 토큰 버전 · 원소 · 열 목록, 182열 이름, `s_y`(타깃별), 표준화
> 통계 sha256, 코드 sha256(`fu2/train.py`, `models.py`, 이 디렉터리 파일 전부), torch · CUDA · GPU, 교차검증 성능(E 요약),
> 적용 범위 문턱(`E_ad.json`), 적용 범위 조건(원소 9종, 중성, TS 원자 8–63, D1 dipole 5종, 위치 p · m).
>
> 최종 모델에는 시험 집합이 없다. **성능 추정치는 K의 교차검증 값을 인용한다.** 참고로 학습 짝 MAE만 보고한다.
>
> ---

> ## 12. Phase E — 평가 (`analysis/deploy/eval_tf.py`)
>
> | 산출 | 내용 |
> |---|---|
> | `E_metrics.csv` | lf · S · TF의 타깃 × 위치(all, p, m, o, direct) × 집합(P_lf 자기 집합, 공통 집합) → n, MAE, RMSE, para MAE core 부트스트랩 95 % CI(2,000회, seed 20261019) |
> | `E_compare.csv` | 공통 para P_main 짝에서 lf 대 LADDER g2 · S · TF: MAE 차, 상대 차와 CI (`analysis/geo2/eval_geo2.py`의 `boot_diff`와 같은 core 부트스트랩) → **G-C 판정** |
> | `E_bends.csv` | para 계열(점 ≥ 12, `hammett_eval.family_series`)마다 타깃별: DFT 꺾임 판정, 예측 판정, AICc 이득 g(예측), ψ (`hammett_eval.calls`, `bent_gain`) |
> | `E_bend_rates.json` | 타깃별 민감도 · 위양성률(+CI), **연속 g AUC**(DFT 판정이 정답, 예측 g가 점수; 계열 부트스트랩 2,000회 CI), 숨은 꺾임 계열(장벽 직선 & 채널 ≥ 1 꺾임) 민감도 · 위양성률 · 정밀도. LADDER g2 · S · TF를 같은 계열에서 나란히 |
> | `E_ad.json` | **적용 범위 문턱**: 타깃별 시험 짝 `yhat_seed_sd`의 95 백분위 (배포에서 범위 밖 표시에 사용) |
> | `E_report.json` | G-C 판정, 위 요약, 입력 sha256 |
>
> meta 위치(σ_m)는 설명용으로 같은 표를 낸다(판정에 쓰지 않음).
>
> ---

구현 메모(코드, 결과를 보기 전): 꺾임 계열 = 시험 예측의 reference 짝(σ 있음, P_sens 전체) 위 `hammett_eval.family_series`(점 ≥ 12, H 포함) — 0-4절 89계열과 같은 규칙; para MAE = P_main reference para 짝(0-4절 1,658짝). 상대 차이 CI = `eval_geo2.boot_diff`와 같은 core 재표본에서 (MAE_lf − MAE_g2)/MAE_g2. E_ad 문턱 = lf 시험 짝 `yhat_seed_sd`의 타깃별 95 백분위(numpy linear).

## 3. 관문
### G-A (7절)
> **G-A (사전 등록 관문)**: 재검출 실패 반응 ≤ **2 %** of L의 `.done` 반응. 넘으면 STOP — 배포에서 그만큼의 반응을 잃는다는
> 뜻이므로 사용자 판단이 필요하다.
>
> ---

- 판정: 재검출 실패 0 / L `.done` 8616 = 0.00 % (기준 ≤ 2 %) → **통과** (`Q_parity.json`)

### G-C (10절)
> **G-C (비열등, 사전 등록 관문)** — 기준값은 사용자가 commit 전에 바꿀 수 있다
> - 대상: lf · S · TF 교차검증 예측 대 LADDER `L__g2__S__TF` 예측, **두 쪽 모두 예측이 있는 para P_main 짝**
> - 9개 주요 타깃(d1 d2 elst pauli oi disp cpcm cds barrier)마다 **상대 차이 (MAE_lf − MAE_g2) / MAE_g2의 core 부트스트랩
>   95 % 상한 ≤ +5 %**
> - 하나라도 넘으면 STOP — W(최종 모델)를 만들지 않고 보고
>
> ---

- 기준값: 9개 주요 타깃마다 상대 차이 95 % 상한 ≤ **+5 %** (사용자가 commit 전에 바꿀 수 있다: md와 json `gates.G_C.rel_ci_hi_max`를 함께 바꾼다)

## 4. 지금까지 본 것
- lf로 학습한 결과: **0개** (`$DEP/units/D__lf__*`, `results/K_*`, `$DEP/model` 없음; 2026-11-03 16:29:29)
- LADDER g2 · S · TF (D1 T1 시험; `eval_tf.py selftest`가 `L_predictions.parquet`에서 다시 계산, `E_selftest.json` sha256 `c7b82804865a81b115d46ff9672fdbe8d522e4867f9e5fedacb83df16e1d8fe7`):

| 타깃 | oi | cpcm | pauli | elst | d1 | d2 | disp | cds | barrier |
|---|---|---|---|---|---|---|---|---|---|
| para MAE (kcal/mol, 1658짝) | 1.014 | 0.299 | 1.735 | 0.879 | 0.374 | 0.298 | 0.131 | 0.027 | 0.327 |
| 연속 g AUC (para 89계열) | 0.845 | 0.787 | 0.775 | 0.757 | 0.916 | 0.686 | 0.635 | 0.466 | 0.834 |

- 숨은 꺾임 계열(같은 89계열): 민감도 0.654, 위양성률 0.206, 정밀도 0.567, 기저율 0.292

- GEO2b `analysis/geo2/results/SUMMARY_GEO2B.md` (sha256 `da28772812546a086fff249e9c3fa534d85f40637805f20a3ae4a023cc64dc39`), 2절 그대로:

> ## 2. D1 결과 (P_main para 짝 1,699, kcal/mol MAE; null = 0 예측)
> | 타깃 | null | S: AW | GNN | TF | XGB | ENS4 | SH: AW | GNN | TF | XGB | ENS4 | **REP8** (95 % CI) |
> |---|---|---|---|---|---|---|---|---|---|---|---|---|
> | d1 | 0.564 | 0.468 | **0.373** | 0.393 | 0.461 | 0.385 | 0.472 | 0.418 | 0.429 | 0.470 | 0.409 | 0.373 [0.281, 0.493] |
> | d2 | 0.330 | 0.361 | **0.287** | 0.307 | 0.360 | 0.304 | 0.361 | 0.343 | 0.358 | 0.378 | 0.324 | 0.287 [0.194, 0.400] |
> | elst | 1.247 | 1.008 | **0.869** | 0.894 | 1.005 | 0.874 | 1.004 | 0.986 | 1.007 | 1.028 | 0.954 | 0.894 [0.622, 1.225] |
> | pauli | 2.426 | 1.947 | **1.665** | 1.702 | 1.882 | 1.665 | 1.918 | 1.894 | 1.945 | 2.001 | 1.829 | 1.702 [1.190, 2.317] |
> | oi | 1.395 | 1.152 | 1.008 | 0.997 | 1.073 | **0.990** | 1.118 | 1.104 | 1.095 | 1.137 | 1.056 | 0.997 [0.728, 1.324] |
> | disp | 0.132 | 0.150 | **0.110** | 0.128 | 0.160 | 0.117 | 0.150 | 0.147 | 0.151 | 0.171 | 0.139 | 0.160 [0.134, 0.190] |
> | cpcm | 0.420 | 0.366 | **0.318** | 0.335 | 0.362 | 0.321 | 0.365 | 0.363 | 0.377 | 0.370 | 0.351 | 0.370 [0.298, 0.453] |
> | cds | 0.030 | 0.045 | **0.028** | 0.030 | 0.039 | 0.031 | 0.042 | 0.041 | 0.039 | 0.042 | 0.038 | 0.041 [0.033, 0.049] |
> | c_ghost | 0.068 | 0.051 | 0.046 | 0.048 | 0.053 | **0.045** | 0.048 | 0.050 | 0.049 | 0.050 | 0.046 | 0.050 [0.042, 0.059] |
> | barrier | 0.642 | 0.484 | **0.356** | 0.379 | 0.448 | 0.367 | 0.478 | 0.401 | 0.418 | 0.462 | 0.391 | 0.356 [0.277, 0.439] |
> - REP8 = 타깃마다 D1 내부 검증 MAE로 고른 대표(사전 등록). 고른 것: d1·d2·barrier S-GNN, elst·pauli·oi S-TF, disp S-XGB, cpcm SH-XGB, cds SH-GNN, c_ghost SH-XGB.
> - 굵은 글씨는 설명용 최저값이다(사후 선택이므로 결과로 쓰지 않는다).
> - 시험 짝에서는 대부분 S-GNN이 가장 낮다. 내부 검증으로 고른 REP8은 disp·cpcm·cds에서 S-GNN보다 높다.
> - disp·cds는 모델과 null 차이가 작거나 null보다 나쁘다(disp REP8 0.160 대 null 0.132, cds 0.041 대 0.030).
> - SH arm(HF 기준선 + 보정)은 이번 rung에서 S arm보다 나아지지 않았다.
> | 선택 | oi para MAE (CI) | oi 민감도 / FPR | cpcm para MAE (CI) | cpcm 민감도 / FPR | 통과 |
> |---|---|---|---|---|---|
> | REP8 (주) | 0.997 [0.728, 1.324] | 0.52 / 0.19 | 0.370 [0.298, 0.453] | 0.65 / 0.13 | 아니오 |
> | REP·S | 0.997 | 0.52 / 0.19 | 0.362 | 0.65 / 0.14 | 아니오 |
> | REP·SH | 1.137 | 0.48 / 0.08 | 0.370 | 0.65 / 0.13 | 아니오 |
> | ENS4·S (보조) | 0.990 | 0.60 / 0.25 | 0.321 | 0.75 / 0.32 | 아니오 |
> | ENS4·SH (보조) | 1.056 | 0.44 / 0.19 | 0.351 | 0.65 / 0.19 | 아니오 |
> - **목표에 미치지 못했다.** oi는 목표의 약 7.7배, cpcm은 약 2.5–2.8배다. 개별 모델(설명용)도 통과한 것이 없다.

- GEO2b `analysis/geo2/results/SUMMARY_DEPLOY.md` (sha256 `c28c9b1e09c1fe21c760325cf11d51bfcd551475ebbe347e84dc1db6e4b54c7b`), 2절 그대로:

> ## 2. 결과 (D1 P_main para 1,697짝, kcal/mol MAE, REP8, core bootstrap 95 % CI)
> - (a) dft 모델 + **DFT 라벨 구조** (LADDER 결과, 같은 짝)
> - (b) dft 모델 + **GEO1b 구조** = 배포 경로
> - (c) GEO2b 모델(geo 구조로 학습) + GEO1b 구조 (참고)
> | 타깃 | null | (a) DFT 구조 | **(b) 배포** | (c) GEO2b | (b) − (a) |
> |---|---|---|---|---|---|
> | d1 | 0.563 | 0.067 [0.060, 0.075] | **5.585** [3.308, 8.249] | 0.372 | +5.52 [3.24, 8.17] |
> | d2 | 0.330 | 0.037 [0.028, 0.046] | **1.961** [1.291, 3.027] | 0.286 | +1.92 [1.26, 3.00] |
> | elst | 1.244 | 0.097 [0.082, 0.115] | **1.260** [0.961, 1.582] | 0.890 | +1.16 [0.87, 1.48] |
> | pauli | 2.422 | 0.119 [0.095, 0.145] | **2.513** [1.830, 3.212] | 1.696 | +2.39 [1.72, 3.09] |
> | oi | 1.393 | 0.103 [0.078, 0.132] | **1.615** [1.197, 2.065] | 0.993 | +1.51 [1.11, 1.94] |
> | disp | 0.132 | 0.001 [0.001, 0.002] | **0.181** [0.138, 0.225] | 0.160 | +0.18 [0.14, 0.22] |
> | cpcm | 0.419 | 0.060 [0.049, 0.072] | **0.483** [0.390, 0.590] | 0.370 | +0.42 [0.34, 0.52] |
> | cds | 0.030 | 0.000 [0.000, 0.000] | **0.050** [0.039, 0.061] | 0.041 | +0.05 [0.04, 0.06] |
> | c_ghost | 0.068 | 0.039 [0.032, 0.046] | **0.088** [0.073, 0.104] | 0.050 | +0.05 [0.04, 0.07] |
> | barrier | 0.642 | 0.108 [0.090, 0.129] | **6.912** [4.200, 9.979] | 0.356 | +6.80 [4.10, 9.88] |
> - **LADDER 목표(oi·cpcm para MAE ≤ 0.13, 꺾임 민감도 ≥ 0.7, FPR ≤ 0.1): 배포 경로 (b)는 통과하지 못했다.** REP8, REP·S, REP·SH, ENS4 모두 불통과다. (a)는 같은 짝에서 oi 0.103, cpcm 0.060이다.
> - (b)는 10개 타깃 모두 null(0 예측)보다 나쁘다. d1·d2·barrier는 null의 6–11배다. (c) GEO2b 모델보다도 모든 타깃에서 나쁘다(disp만 CI가 0을 포함).
> | 타깃 | AW·S | GNN·S | TF·S | XGB·S | AW·SH | GNN·SH | TF·SH | XGB·SH |
> |---|---|---|---|---|---|---|---|---|
> | d1 | 4.74 | 3.87 | 1.09 | 2.12 | 5.58 | 5.85 | 5.08 | 5.34 |
> | oi | 1.50 | 2.70 | 1.29 | 1.38 | 1.67 | 1.62 | 1.51 | 1.49 |
> | cpcm | 0.70 | 1.13 | 0.42 | 0.41 | 0.46 | 0.48 | 0.42 | 0.42 |
> | barrier | 5.33 | 4.27 | 1.22 | 2.55 | 6.91 | 7.04 | 6.25 | 6.59 |

- Q 재현 보고 (`results/Q_parity.json`, sha256 `eadf3e7bf0b5ec416d7334096fec7f27b409bd5c2129790361a9ebf6f7d57cbe`):

```json
{
 "spec": "docs/specs/DEPLOY_MODEL.md 7 (Phase Q; STOP 16 Q)",
 "written": "2026-11-03 15:46:48",
 "tol": 1e-06,
 "inputs": {
  "x182_g2": "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/X182_d1_g2.parquet",
  "x182_g2_sha256": "2cecf407e5f7cfa3ff75ce4e7a6482ad9ee2802c8b3ae449843dfcbbb636e384",
  "x182_lf": "/gpfs/tmp_cpu2/yeseo1ee/deploy/X182_lf.parquet",
  "x182_lf_sha256": "8917a728c231a48baff5bafe697dc88f745e9c0fc8d807690805c47dbf42bd18",
  "tokens_g2": "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/fu2/tokens",
  "tokens_lf": "/gpfs/tmp_cpu2/yeseo1ee/deploy/fu2_lf/tokens",
  "g2_T_manifest_sha256": "6590f59d8fbc9fc9d2625bb94388aa1b5257f63154fe94e73dd24e29368e00e5",
  "lf_T_manifest_sha256": "72e0523c2ac14f2312431f86109191858de314eed875f74486d6eefeb5db2b92",
  "F_report_sha256": "aebb713e70aed492c7b13cf4a2555e036bd6ccd526b30932712b7a1f67f6ee5f",
  "L_status": "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/L_status.csv",
  "T_lf_check_stop": false
 },
 "counts": {
  "lf_folders": 9266,
  "lf_done": 8616,
  "g2_done": 8616,
  "g2_done_without_lf_folder": 0,
  "g2_ok": 8499,
  "lf_ok": 8502,
  "common_both_ok": 8499,
  "lf_only_ok": 3,
  "g2_only_ok": 0
 },
 "lf_only_ok": {
  "rxn": [
   1004463,
   1009844,
   1009881
  ],
  "g2_reasons": {
   "g1feat:spe_fail:RuntimeError:xtb parse fail (Wiberg table)": 3
  }
 },
 "g2_only_ok": {
  "rxn": [],
  "lf_reasons": {}
 },
 "agreement": {
  "x182_equal": 8492,
  "x182_bit_equal": 4145,
  "tokens_equal": 8492,
  "all_equal": 8492,
  "x182_maxdiff": 0.004999999999999893,
  "x182_maxdiff_nonB6": 0.004999999999999893,
  "tok_num_maxdiff": 0.004999999999999893
 },
 "categories": {
  "equal": 8492,
  "perm_symmetric_map": 7
 },
 "differing_columns": {
  "x182_n_reactions": {
   "mulliken_R_dph_0": 6,
   "mulliken_R_dph_1": 6,
   "wbo_valence_R_dph_0": 3,
   "wbo_valence_R_dph_1": 3,
   "dist_R_dip_ab": 1,
   "dist_R_dip_bc": 1,
   "mulliken_R_dip_0": 1,
   "mulliken_R_dip_2": 1,
   "wbo_valence_R_dip_0": 1,
   "wbo_valence_R_dip_2": 1
  },
  "x182_max_absdiff": {
   "mulliken_R_dph_0": 0.0037821400000000033,
   "mulliken_R_dph_1": 0.0037821400000000033,
   "wbo_valence_R_dph_0": 0.0019999999999997797,
   "wbo_valence_R_dph_1": 0.0019999999999997797,
   "dist_R_dip_ab": 0.0007892968892180274,
   "dist_R_dip_bc": 0.0007892968892180274,
   "mulliken_R_dip_0": 0.0031609200000000115,
   "mulliken_R_dip_2": 0.0031609200000000115,
   "wbo_valence_R_dip_0": 0.004999999999999893,
   "wbo_valence_R_dip_2": 0.004999999999999893
  },
  "token_num_n_reactions": {
   "q_rel": 7,
   "dq_dist": 7,
   "v_rel": 4,
   "dv_dist": 4
  }
 },
 "differing_reactions": {
  "1000397": {
   "category": "perm_symmetric_map",
   "cols": "mulliken_R_dph_0;mulliken_R_dph_1;wbo_valence_R_dph_0;wbo_valence_R_dph_1",
   "tok": "q_rel;dq_dist;v_rel;dv_dist",
   "perm_identity": false,
   "map_source": "g1_redetect"
  },
  "1000401": {
   "category": "perm_symmetric_map",
   "cols": "mulliken_R_dph_0;mulliken_R_dph_1;wbo_valence_R_dph_0;wbo_valence_R_dph_1",
   "tok": "q_rel;dq_dist;v_rel;dv_dist",
   "perm_identity": false,
   "map_source": "g1_redetect"
  },
  "1000669": {
   "category": "perm_symmetric_map",
   "cols": "mulliken_R_dph_0;mulliken_R_dph_1",
   "tok": "q_rel;dq_dist",
   "perm_identity": false,
   "map_source": "g1_redetect"
  },
  "1000675": {
   "category": "perm_symmetric_map",
   "cols": "mulliken_R_dph_0;mulliken_R_dph_1",
   "tok": "q_rel;dq_dist",
   "perm_identity": false,
   "map_source": "g1_redetect"
  },
  "1000698": {
   "category": "perm_symmetric_map",
   "cols": "mulliken_R_dph_0;mulliken_R_dph_1",
   "tok": "q_rel;dq_dist",
   "perm_identity": false,
   "map_source": "g1_redetect"
  },
  "1000941": {
   "category": "perm_symmetric_map",
   "cols": "mulliken_R_dph_0;mulliken_R_dph_1;wbo_valence_R_dph_0;wbo_valence_R_dph_1",
   "tok": "q_rel;dq_dist;v_rel;dv_dist",
   "perm_identity": false,
   "map_source": "g1_redetect"
  },
  "1001535": {
   "category": "perm_symmetric_map",
   "cols": "dist_R_dip_ab;dist_R_dip_bc;mulliken_R_dip_0;mulliken_R_dip_2;wbo_valence_R_dip_0;wbo_valence_R_dip_2",
   "tok": "q_rel;dq_dist;v_rel;dv_dist",
   "perm_identity": false,
   "map_source": "g1_redetect"
  }
 },
 "permutation": {
  "identity": 8609,
  "non_identity": 7,
  "status": {
   "ok": 8616
  },
  "perm_L": {
   "absent": 8616
  },
  "L_status_identity_agrees": {
   "True": 8616,
   "nan": 650
  },
  "coord_maxdiff": 0.0,
  "rel": {
   "rel1": {
    "identity": 8616
   },
   "rel2": {
    "identity": 8616
   }
  }
 },
 "L3": {
  "A_idx_differs": [
   1004463,
   1009844,
   1009881
  ],
  "formed_differs": []
 },
 "L4_dnu_cm": {
  "max_abs": 4.4277148703031344e-07,
  "n_gt_1": 0
 },
 "map_source": {
  "g1_redetect": 8501,
  "none": 765
 },
 "redetection": {
  "rule": "xtb_status 'partition:*' or 'partition_mismatch_g1' (spec 5-4 F-2)",
  "n": 0,
  "n_L_done": 8616,
  "n_L_done_L_report": 8616,
  "rate": 0.0,
  "statuses": {},
  "rxn": [],
  "by_dipole": {},
  "by_skeleton": {},
  "ladder_map_source": {}
 },
 "G_A": {
  "threshold": 0.02,
  "rate": 0.0,
  "pass_": true
 },
 "interpretation": {
  "rule": "spec 7: (a) ladder_dft_map, (b) b6_only, (c) a bug if non-B6 columns differ at random; decision: perm_symmetric_map (TS atom order differs; reactant-side columns are a permutation within their groups) is explained, any other difference > 1e-6 = 'unexplained' = STOP",
  "n_unexplained": 0,
  "unexplained": {}
 },
 "n_compare_errors": 0,
 "stop": false,
 "stop_reasons": []
}
```

## 5. 입력 sha256

| 파일 | sha256 |
|---|---|
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/labels_lf.json` | `e7fdf9b57371f7c8e2851cd7fef4ef131604b6257047bd5e75f55ee326b5f720` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/input_meta_lf.csv` | `104ab323de72fb7e4a999744e5957914e3d4dfedefbeddafa8c1bf4941451a09` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/X182_lf.parquet` | `8917a728c231a48baff5bafe697dc88f745e9c0fc8d807690805c47dbf42bd18` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/fu2_lf/T_manifest.csv` | `72e0523c2ac14f2312431f86109191858de314eed875f74486d6eefeb5db2b92` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/fu2_lf/T_report.json` | `7d36292c0078f5fb32d94e89894ef79ff4e7258c52de58b835066d4520d183d5` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/fu2_lf/Q_report.json` | `d9d657fbbc781d78cf3fcf504a0fc14092cd61faeb999217189909eb9eb680c0` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/fu2_lf/Q_manifest.csv` | `b886ca126df69b8c8e905ad7d61dd11d2600a2288922c9c7c5b175b85606b4ad` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/F_pairs_cr2.csv` | `f2fe5cab1c61f082b7cf929ca1146d96f8e451d37ed7d4ee2b5b127b4a7fa6ce` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/d1_pair/results/fu2/S_splits.json` | `eefe495a775babf2604be6f21156e4ed2869a1988bd08d1542c0d7d4b7779413` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/d1_pair/results/S_folds.csv` | `944a1d1439eb0d54dff90f58f40476c21109cf3a5b71b4bb0f755651b216dcb7` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/d1_ladder/results/L_predictions.parquet` | `1de11e77ea82571b139d6070ad556386e2a74b482af5c357498614bee37b3e4d` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/d1_ladder/results/V_report.json` | `d4b2c0dfd340cb81596352dd6ca74452c1cce4a1d8601e05170abe1097a46f1f` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f0.json` | `8613360436df1cf55f4929c0e57d91e2b74a7a3579489315b0548c7f22f54521` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f1.json` | `28c40b0c4c2b8fa3f12bc753cf3976b5851e16327730527a26b3392dd3c8dc4c` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f2.json` | `0f1939c9e3b8194785205dacbc10eec725cdcd93ae89b2e784f726dac334735e` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f3.json` | `992a7ba21d4ac57948d78b83f5fc90efdcc82ea03a9667b548a8ea5c1665ff27` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f4.json` | `168b876bd111a3e7a078b493db50a75b4db7ae09b6cd0e6eb51eb32b64bafec6` |

- C 검사: `results/C_tests.json` sha256 `5eb1301028b56d1aa9f240bcecc294ca19d2cf2a0316a3e37259f19516f339f8` (all_pass)

## 6. 코드 sha256 (작성 시)

| 파일 | sha256 |
|---|---|
| `analysis/deploy/c1_dft_free.py` | `d4aacd21135c867e7213897a7a05e494a42f684ca45b0f76dbeac955b1c7c152` |
| `analysis/deploy/check_start.py` | `647a1d829bef62c35308a054d17f783397cb9a9a464c51b9580e265ec6ee55ca` |
| `analysis/deploy/data_lf.py` | `2ffe3313898015a6ae749a5348ab774354722fccbef7c5454cf1e3c04e417eb8` |
| `analysis/deploy/dep_common.py` | `27d77566871d0e838345e92067d689c47d8459a9087c42cb79aafb5bc87f5f8a` |
| `analysis/deploy/eval_tf.py` | `8b1fe4e06e94e7ad094cf066ec45627823e3b6a772110ce20ba0d5fd3b90f9c3` |
| `analysis/deploy/feat_lf.py` | `28be59794a476a362c80babe92cd60c6457138b70ca1a58a6d3f4051e7c00a2a` |
| `analysis/deploy/finalize_lf.py` | `7bf79b8d5670e255146a2834e76c5405b74cc9ee430a1859e4b29f38d2240ead` |
| `analysis/deploy/g2_deploy.py` | `cc0cc09f2f427ec464c944b6a0a252fb6546d40236da6d96cda091929bdc069a` |
| `analysis/deploy/meta_lf.py` | `cb48851329f0855e5bcc45574ffe19eccb73dc3e0c01faeb3a2622148cd80bc9` |
| `analysis/deploy/parity.py` | `af23a7836dcd2877331a37b56009f94f5ef4bb0b0920fdbc1790dd7f41ff0948` |
| `analysis/deploy/predict_tf.py` | `d5332129bc0bb19e2551ecc5de4945cc366531bb021ef5f9220ba4974e37e591` |
| `analysis/deploy/prereg_deploy.py` | `63fbc1e0bd4b5941bb7739def642611d36b3bde0b3d147759297b4543184c3dc` |
| `analysis/deploy/tests.py` | `85a0e448bbd5d54f809d953ce0587d7c7e1e52aef841f2cf28b707d2c9c9739e` |
| `analysis/deploy/tokens_lf.py` | `a24995927746c4c9fe7dd26ce78aa3aadf0269e9a4cc4f33105bec1354111815` |
| `analysis/deploy/train_tf.py` | `2755fafbd94332c16008f9ca93bb021f7f9b1af520081633ba0158f6e65624b7` |
