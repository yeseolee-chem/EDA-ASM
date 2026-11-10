# PREREG CH3M — 3채널 직접 대 10타깃 합 (TF · GNN · XGB) 사전 등록 (2026-11-10)

`docs/specs/CH3M_MODELS.md` 6절의 사전 등록이다(명령서 sha256 `7d6af9a731234a22fc6de6c986e0f3f9b58042265c9565ae7263577ca31a5d1e`). `analysis/ch3m/prereg_m.py`(ch3m-prereg-1)가 결과 파일만 읽어 만들었다(git HEAD `93669fb4218b9f22f8803e9aa24806ae27ac0f95`, 작성 2026-11-10 17:31:02). 인용 블록 = 명령서 원문. **K(`train_m.py run`, `xgb_m.py run`)와 W(`train_m.py final`)는 이 md와 `prereg_ch3m.json`이 commit되고 바뀌지 않았을 때만 돈다**(`prereg_m.py check`).

## 1. 실험 칸 (0-2)
> ### 0-2. 실험 칸 (2 × 3)
>
> | 모델 | 10타깃 학습 → 3채널로 더함 | 3채널 직접 (strain · inter · solv · barrier) |
> |---|---|---|
> | TF | **새로** `D__lf10__S__TF__f{0..4}` | 기존 ch3 재사용: `analysis/ch3/results/K3_predictions.parquet` (`D__lf3__S__TF`) |
> | GNN | **새로** `D__lf10__S__GNN__f{0..4}` | **새로** `D__lf3__S__GNN__f{0..4}` |
> | XGB | **새로** `D__lf10__S__XGB__f{0..4}` | **새로** `D__lf3__S__XGB__f{0..4}` |
>
> - 모든 새 단위는 ch3과 같은 **재생성 lf 입력** `$CH3/lf`를 쓴다. 그래서 6칸이 모두 같은 입력 위에 있다.
> - **기준(현재 배포 모델) `dep`:** deploy commit의 `analysis/deploy/results/K_predictions.parquet`(lf TF 10타깃, 원래 입력)을 더한 값.
> - **TF 10타깃을 다시 학습하는 이유:**
>   - 6칸을 같은 입력으로 맞춘다.
>   - 같은 모델을 다시 학습했을 때 생기는 차이(TF10 대 dep)를 **흔들림 기준**으로 쓴다. ch3 C-3′ 진단에서 이 흔들림이 수 %였다.
> - **AW는 하지 않는다.** LADDER g2에서 이미 확실히 나빴다(0-3 표).

## 2. 이미 아는 것과 가설 H1 (0-3)
> ### 0-3. 이미 아는 것 (참고값, 판정 아님)
>
> LADDER g2 · S 10타깃 교차검증 예측(`analysis/d1_ladder/results/L_predictions.parquet`)을 3채널로 더한 값이다.
> P_main para 공통 1,658짝에서 계산했고, 입력은 g2(lf와 1e-6 수준 차이)다.
>
> | 모델 | strain | inter | solv | barrier |
> |---|---|---|---|---|
> | g2 TF 10합 | 0.360 | 0.373 | 0.294 | 0.327 |
> | g2 GNN 10합 | 0.362 | 0.368 | 0.288 | 0.325 |
> | g2 AW 10합 | 0.427 | 0.404 | 0.317 | 0.385 |
> | g2 XGB 10합 | 0.436 | **0.772** | 0.309 | 0.392 |
> | dep (lf TF 10합) | 0.361 | 0.383 | 0.310 | 0.330 |
> | ch3 (lf TF 3직접) | 0.390 | 0.382 | 0.306 | 0.349 |
>
> para에서 본 채널 오차의 상쇄:
>
> | | TF | GNN | XGB |
> |---|---|---|---|
> | corr(오차 pauli, 오차 elst + oi) | −0.99 | −0.99 | −0.96 |
> | elst · pauli · oi 오차를 따로 더한 크기 → 합의 오차 | 3.64 → 0.34 | 3.74 → 0.34 | 4.02 → 0.72 |
> | corr(오차 d1, 오차 d2) | −0.54 | −0.64 | −0.51 |
>
> **사전 가설 H1:**
> - 10타깃 합에서 채널 오차가 거의 완전히 상쇄되는 모델(TF · GNN)은 3채널 직접 학습의 이득이 없다.
> - 상쇄가 덜 되는 모델(XGB: 타깃마다 따로 학습)은 inter에서 이득이 있다.
>
> 8-4절에서 설명용으로 확인한다.

## 3. 공통 설계 (0-4)
> ### 0-4. 공통 설계
>
> | 항목 | 값 |
> |---|---|
> | 라벨 | `$LAD/F_pairs_cr2.csv`의 `dd_<t>` 10개. 3채널 = `y10 @ M` (`analysis/ch3/ch3_common.M`: strain = d1 + d2, inter = elst + pauli + oi + disp + c_ghost, solv = cpcm + cds, barrier) |
> | 입력 | lf · S arm. TF · GNN = 원자 토큰(d_in 29) + 전역 182열. XGB = 전역 182열만(LADDER · rev 9 arm S와 같음) |
> | 분할 | FU2 T1 5 fold (`$D1PAIR_SCRATCH/fu2/splits/T1__f{k}.json`). 사용 가능 짝 P_lf 7,388, 시험 짝 7,279 (ch3 K3와 같음) |
> | TF · GNN 학습 | DEPLOY 11-2 = FU2 M-1 / LADDER arm S. `Mo.build(model, cell, d_in=29, n_out, d_glob=182, "A+G")`. grid: TF 2/3층 × 64/128, **GNN 4/6층 × 64/128** (`fu2/models.py GRID`). AdamW 1e-3 / wd 1e-4, cosine 300 epoch, dropout 0.1, batch 64짝, patience 30, (a,b)/(b,a) 순서 무작위 증강. 튜닝은 inner_train에서 하고 칸 선택은 inner_val로 한다. final = 바깥 학습 짝, 튜닝 최적 epoch, seed 20261019 + 0..4 평균 |
> | TF · GNN 타깃 모드 | 10모드 = deploy 값(항등 벌점 Σ₉, 검증 지표 = 주 9타깃 평균). 3모드 = ch3 값(벌점 Σ₃, 검증 지표 = 4타깃 평균). `analysis/ch3/data_ch3.use_targets`로 바꾼다 |
> | XGB 학습 | LADDER `train_xgb.py` arm S 규칙 그대로 (= rev 9 arm C). `d1_train.Scale`, `_xgb_fit`(양방향), `_xgb_g`(반대칭), grid 12칸. 타깃마다 따로: inner GroupKFold(4, core) 평균 MAE로 칸을 고르고, 바깥 학습 짝 전부로 final fit. seed 없음(한 번 fit) |
> | XGB 3모드 | 타깃 strain · inter · solv · barrier를 각각 따로 튜닝한다. barrier는 10모드와 같은 데이터 · 규칙이라 **같은 값이 나와야 한다**(C-XB) |
> | 관문 | 모든 fit에서 C-4 반대칭(\|g(a,b)+g(b,a)\|, \|g(a,a)\| ≤ 1e-4 kcal/mol, 무작위 시험 짝 200개), C-5 누출 검사 |
> | GPU | RTX 3090 (`-p gpu1 --gres=gpu:rtx3090:1`). deploy · LADDER · ch3과 같은 종류. 단위 지문에 GPU 이름을 적는다 |

## 4. 평가 정의 전부, 해석 규칙(8-2), 교체 규칙(8-3) (8절)
> ## 8. Phase E — 평가 (`eval_m.py run`)
>
> ### 8-1. 공통
>
> **모델 7개 (3채널 예측):**
> - 10타깃 모델(`dep`, `TF10`, `GNN10`, `XGB10`)은 yhat을 `M`으로 더한다. barrier는 직접 예측을 쓴다.
> - 3채널 모델(`TF3` = ch3 K3, `GNN3`, `XGB3`)은 그대로 쓴다.
>
> **정답 확인:** 정답 y는 모두 `F_pairs_cr2.csv @ M`이다. 각 예측 파일의 y가 이 값과 같은지 먼저 확인한다(다르면 STOP).
>
> **나머지 정의는 `eval_ch3.py`와 같다:**
> - 짝 속성(`S_folds.csv` core · fam · in_P_main · in_P_sens, 라벨 파일 dipole · cfg · series · pos · Xb)
> - 집합 P_main = reference & in_P_main
> - **7개 모델이 모두 예측한 공통 짝**만 쓴다.
> - 부트스트랩은 core 단위 2,000회, seed 20261019 (`analysis/geo2/eval_geo2.py boot_diff`)
>
> **주 집합 = P_main · para 공통 짝.** 0-3과 같으면 1,658짝이다.
>
> ### 8-2. Q1 — 3채널 직접 대 10타깃 합, 모델마다 → `EM_3v10.csv`
>
> TF · GNN · XGB 각각, 타깃 4개(strain · inter · solv · barrier)마다 다음을 계산한다.
> - 상대 차 = (MAE(x3) − MAE(x10합)) / MAE(x10합)과 95 % CI
> - 대상은 P_main para(주)와 P_main all(보조)이다.
>
> **흔들림 기준 noise_t:**
> - noise_t = \|상대 차(TF10 대 dep)\|. 같은 모델을 다시 학습했을 때의 차이다.
> - `EM_noise.csv`에 TF10 대 dep의 상대 차 · CI도 적는다.
>
> **해석 규칙 (사전 등록, 판정용 아님):**
>
> | 표시 | 조건 |
> |---|---|
> | 3 나음 | 95 % 상한 < 0 이고 점추정 < −max(noise_t, 2 %) |
> | 10합 나음 | 95 % 하한 > 0 이고 점추정 > +max(noise_t, 2 %) |
> | 같음 | 그 밖 |
>
> ### 8-3. Q2 — 배포 모델 교체 규칙 → `EM_choice.csv`
>
> **후보:** `TF3`, `GNN10`, `GNN3`, `XGB10`, `XGB3`. `TF10`은 dep과 같은 모델이라 후보가 아니다.
> **기준:** `dep`.
> **지표:** P_main para 공통 짝에서 타깃 4개마다 상대 차 r_t = (MAE_후보 − MAE_dep) / MAE_dep, 95 % CI.
>
> **교체 조건 (둘 다 만족):**
> - (a) 4타깃 모두 95 % 상한 ≤ +5 %
> - (b) 4타깃 중 2개 이상에서 점추정 ≤ −5 % **그리고** 95 % 상한 < 0
>
> **판정:**
> - 여러 후보가 통과하면 r_t 4개의 평균이 가장 낮은 후보를 고른다. 동률이면 TF3 → GNN10 → GNN3 → XGB10 → XGB3 순서로 고른다.
> - 아무도 통과하지 못하면 **배포 모델은 그대로**(deploy lf TF 10합)다. STOP하지 않는다.
> - −5 %의 근거: 같은 deploy 모델을 seed만 바꿔 다시 학습해도 barrier MAE가 +3.6 % 달라졌다(deploy `NULL_gc.json`). 그보다 큰 개선만 교체 사유로 본다.
>
> ### 8-4. 오차 상쇄 진단 (H1, 설명용) → `EM_cancel.csv`
>
> 10타깃 모델 4개(`dep`, `TF10`, `GNN10`, `XGB10`)에 대해 P_main para · all에서 계산한다.
> - corr(오차 pauli, 오차 elst + oi), corr(오차 d1, 오차 d2), corr(오차 cpcm, 오차 cds)
> - 묶음마다 상쇄 비율 = 평균 \|Σ 오차\| / 평균 Σ \|오차\| (낮을수록 많이 상쇄됨)
> - 8-2의 상대 차를 함께 적는다.
>
> H1에 맞는지는 표로만 보고한다(모델 3개라 검정하지 않는다).
>
> ### 8-5. 보조 지표 (7개 모델 모두, 판정 아님)
>
> `eval_ch3.py` 정의 그대로 쓴다(P_sens 기준 공통 짝, 예측 자신의 barrier 부호).
>
> | 산출 | 내용 |
> |---|---|
> | `EM_metrics.csv` | 타깃 × 위치: MAE, RMSE, r, 부호 정답률(\|y\| ≥ 1) |
> | `EM_rank.csv` | 계열 안 장벽 순위: 두 작용기 순서 정답률(0.41–1 / 1–2 / > 2), 실제 최저가 예측 1등 · 상위 3, Spearman 중앙값 · 25 % |
> | `EM_cause.csv` | 주도 채널 · 유형(S / SI / I) 일치율. 반응 하나와 칸 평균, 기준값 함께. 칸 평균 r · MAE |
> | `EM_ad.json` | 적용 범위 문턱(시험 짝 `yhat_seed_sd` 95 백분위). XGB는 seed가 하나라 기록하지 않는다 |
> | `EM_identity.csv` | \|ŷ_barrier − (ŷ_strain + ŷ_inter + ŷ_solv)\| 평균 · p95 |
>
> ### 8-6. selftest (`eval_m.py selftest`, K 전에 실행)
>
> - (i) `TF3` 대 `dep`이 ch3 `E3_compare.csv`의 P_main para 4줄을 소수 넷째 자리까지 재현한다.
> - (ii) LADDER `L_predictions.parquet` g2 · S의 TF · GNN · XGB · AW를 같은 코드로 더하면 0-3 표를 소수 셋째 자리까지 재현한다.
> - 다르면 정의 차이를 보고한다(STOP 아님).
>
> 산출: `results/EM_selftest.json`, `EM_report.json`(8-2 표시, 8-3 판정, 요약, 입력 sha256).
>
> ---

## 5. 9절 (최종 모델 · 추론기)
> ## 9. Phase W · I — 최종 모델과 추론기 (8-3에서 교체될 때만)
>
> | 고른 후보 | 할 일 |
> |---|---|
> | 없음 | 배포 모델 = lf TF 10합 그대로. 1-8에서 deploy 최종 모델 파일이 있으면 그것을 쓰고 W · I는 하지 않는다. **없으면** 같은 모델을 재생성 입력에서 다시 만든다: `train_m.py final TF10`(아래 GNN 행과 같은 DEPLOY 13 규칙, `D__lf10__S__TF__all__s{0..4}.pt`) → I-1 · I-2. 약 0.15 GPU-h |
> | `TF3` | ch3 최종 모델(`$CH3/model/D__lf3__S__TF__all__s{0..4}.pt`, `analysis/ch3/results/MODEL_CARD_ch3.json`, `I3_selftest.json`)이 이미 있다. 기록만 한다 |
> | `GNN10` · `GNN3` | `train_m.py final <후보>`: DEPLOY 13 규칙(학습 짝 `split("all")`, 칸 = fold 선택 최빈값(동률: 튜닝 검증 평균 낮은 칸 → grid 순서), epoch = 최적 epoch 중앙값, seed 20261019 + 0..4, C-4 · C-T) → `$CH3M/model/D__<rung>__S__GNN__all__s{0..4}.pt`, `MODEL_CARD.json`(사본 `results/MODEL_CARD_ch3m.json`), `results/WM_report.json`. 그다음 `predict_m.py` I-1(`--model cv`로 KM 시험 짝 재예측 ≤ 1e-4) · I-2(`--model final`로 학습 짝 1,000개, 같은 GPU에서 ≤ 1e-6) → `results/IM_selftest.json` |
> | `XGB10` · `XGB3` | **STOP하고 보고**한다. XGB 최종 모델 저장 · 추론기는 아직 없어서 별도 명령서가 필요하다. 0-3으로 보아 일어날 가능성은 낮다 |
>
> ---

### 5-1. 구현 정의 (`eval_m.py` DEFINITIONS, K 결과를 보기 전에 고정)

- **models**: dep = deploy K_predictions (lf TF 10타깃, 원래 입력) @ M; TF10 / GNN10 / XGB10 = KM rung lf10 @ M; TF3 = ch3 K3_predictions; GNN3 / XGB3 = KM rung lf3. barrier는 각 모델의 직접 예측
- **pairs**: 7개 모델이 모두 예측한 공통 시험 짝; 주 집합 = P_main(reference & in_P_main) · para
- **q1**: EM_3v10: 모델마다 x3 대 x10합, 상대 차 r = (MAE(x3) − MAE(x10합)) / MAE(x10합), core 부트스트랩 2,000회 seed 20261019 95 % CI (eval_geo2.boot_diff 재표본, eval_ch3.boot_rel); P_main para(주) · all(보조) · m. 표시: 3 나음 = 상한 < 0 & r < −max(noise_t, 0.02); 10합 나음 = 하한 > 0 & r > +max(noise_t, 0.02); 그 밖 같음. noise_t = |r(TF10 대 dep)|, 같은 위치(para 행은 P_main para, all 행은 P_main all)
- **q2**: EM_choice: 후보 ['TF3', 'GNN10', 'GNN3', 'XGB10', 'XGB3'] 대 dep, P_main para r_t. 교체 = (a) 4타깃 모두 95 % 상한 ≤ +5 % 그리고 (b) 2타깃 이상 r ≤ -5 % & 상한 < 0. 여럿이면 r_t 평균 최소, 동률 ['TF3', 'GNN10', 'GNN3', 'XGB10', 'XGB3'] 순서. 없으면 배포 모델 그대로
- **cancel**: EM_cancel: 10타깃 모델(dep, TF10, GNN10, XGB10)의 오차 e_t = yhat_t − y_t; corr(e_pauli, e_elst + e_oi), corr(e_d1, e_d2), corr(e_cpcm, e_cds); 묶음 상쇄 비율 = 평균 |Σ e| / 평균 Σ |e| (strain, inter, solv, epo = elst + pauli + oi); Σ MAE(묶음 타깃) → MAE(묶음 합)
- **aux**: EM_metrics / EM_rank / EM_cause / EM_identity: eval_ch3 정의 그대로(9-3 주 집합 P_sens_ref, 예측 자신의 barrier 부호); EM_ad: TF · GNN · dep 모델의 자기 타깃별 시험 짝 yhat_seed_sd 95 백분위(XGB 없음)

### 5-2. selftest (`results/EM_selftest.json`)

- (i) TF3 vs dep = ch3 E3_compare P_main para (4 dp): 통과
- (ii) LADDER g2 sums + dep + ch3 = spec 0-3 table (3 dp): 통과

| 행 | 명령서 0-3 (strain, inter, solv, barrier) | 재현 | 일치 |
|---|---|---|---|
| dep (lf TF 10합) | [0.361, 0.383, 0.31, 0.33] | [0.361, 0.383, 0.31, 0.33] | 예 |
| ch3 (lf TF 3직접) | [0.39, 0.382, 0.306, 0.349] | [0.39, 0.382, 0.306, 0.349] | 예 |
| g2 TF 10합 | [0.36, 0.373, 0.294, 0.327] | [0.36, 0.373, 0.294, 0.327] | 예 |
| g2 GNN 10합 | [0.362, 0.368, 0.288, 0.325] | [0.362, 0.368, 0.288, 0.325] | 예 |
| g2 XGB 10합 | [0.436, 0.772, 0.309, 0.392] | [0.436, 0.772, 0.309, 0.392] | 예 |
| g2 AW 10합 | [0.427, 0.404, 0.317, 0.385] | [0.427, 0.404, 0.317, 0.385] | 예 |

## 6. 지금까지 본 것
- K 학습 결과: **0개** (`$CH3M/units/D__*`, `results/KM_*` 없음; 2026-11-10 17:31:02)
- 본 것: 기준 dep · ch3(TF3) 예측과 LADDER g2 예측(0-3 표, selftest), C 검사(g2 fold 0 회귀: TF · GNN · XGB, smoke 2 epoch / 2칸)
- C 검사: all_pass True

## 7. 입력 · 기준 파일 sha256

| 파일 | sha256 |
|---|---|
| `/gpfs/tmp_cpu2/yeseo1ee/ch3/lf/X182_lf.parquet` | `cd5ec799f088a72aa31a2682240f0657469d07db5f5fc56050aa92765fb7f199` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3m/analysis/ch3/results/lf_inputs/fu2_lf/T_manifest.csv` | `d9c914380a235aae919d503bd5eed2034cfba5c20073c519476dfd0be5bacf72` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3m/analysis/ch3/results/lf_inputs/fu2_lf/T_report.json` | `5eecddf5319d7be77c809dd85a09ae79d4643961bf0dc135582e1b9bb62b732e` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3m/analysis/ch3/results/lf_inputs/fu2_lf/Q_report.json` | `b431c5c9e82aed5974e5d5944e9b45f603e5877273229ee7792eb3ff0b17c420` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/F_pairs_cr2.csv` | `f2fe5cab1c61f082b7cf929ca1146d96f8e451d37ed7d4ee2b5b127b4a7fa6ce` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3m/analysis/d1_pair/results/fu2/S_splits.json` | `eefe495a775babf2604be6f21156e4ed2869a1988bd08d1542c0d7d4b7779413` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3m/analysis/d1_pair/results/S_folds.csv` | `944a1d1439eb0d54dff90f58f40476c21109cf3a5b71b4bb0f755651b216dcb7` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3m/analysis/deploy/results/K_predictions.parquet` | `a7c30339e46b5455972ba4bb93493d3f50f33db4b77d8bc6f3a83a5b53f58616` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3m/analysis/ch3/results/K3_predictions.parquet` | `4faefc9c5e38c1a8e6b1592c1c3a476cc31c9edfa1d1fea9e8ce388cde0fa24e` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3m/analysis/d1_ladder/results/L_predictions.parquet` | `1de11e77ea82571b139d6070ad556386e2a74b482af5c357498614bee37b3e4d` |
| `/gpfs/tmp_cpu2/yeseo1ee/ch3m/xgb_pack/lf.npz` | `b4bc27cf6088df5c22ca3e1fcf519f156d140e42eb1b8cbba9866ce5e98eb4de` |
| `/gpfs/tmp_cpu2/yeseo1ee/ch3m/xgb_pack/lf.json` | `db97a723616646057ef82c6762dad573f22f8124f346559425dd6bde8dd0a0ef` |
| `/gpfs/tmp_cpu2/yeseo1ee/ch3m/xgb_pack/g2.npz` | `d07a9ec1f2e183bb3b7cbec9452a9ae62a248c00c9cb2bb74e3836acb0f3d8de` |
| `/gpfs/tmp_cpu2/yeseo1ee/ch3m/xgb_pack/g2.json` | `e6754b13bd4854a0a154718bf95402995299ad8579635907cbfb24b257fccb68` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3m/analysis/ch3m/results/XP_report.json` | `e79df84ff9177bc88733f028a789a736e1c3a11aa78889607b6cae320b207493` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3m/analysis/ch3m/results/C_tests.json` | `f2551cd395c8c6b50b33564e942dff4a4f20bdd358fa7b475842b07d3335e986` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3m/analysis/ch3m/results/S0_start.json` | `1a6419ab2cc832d3cd59ab62c56bc6b1d6ce02b20f6a0ec774897caaf0845c3b` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f0.json` | `8613360436df1cf55f4929c0e57d91e2b74a7a3579489315b0548c7f22f54521` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f1.json` | `28c40b0c4c2b8fa3f12bc753cf3976b5851e16327730527a26b3392dd3c8dc4c` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f2.json` | `0f1939c9e3b8194785205dacbc10eec725cdcd93ae89b2e784f726dac334735e` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f3.json` | `992a7ba21d4ac57948d78b83f5fc90efdcc82ea03a9667b548a8ea5c1665ff27` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f4.json` | `168b876bd111a3e7a078b493db50a75b4db7ae09b6cd0e6eb51eb32b64bafec6` |

## 8. 코드 sha256 (작성 시)

| 파일 | sha256 |
|---|---|
| `analysis/ch3m/ch3m_common.py` | `c233e0ce73479957b98994d1c1da1eeb221185dd0350602da6b413730e8b89b4` |
| `analysis/ch3m/commit_m.py` | `c6a23e9203acac698c60cb17e484cb9ac024e1aa3d140fcb88b09bab22fe9517` |
| `analysis/ch3m/devtest_m.py` | `06b898cc1bd51d69abfe84da05121e84074000c52534f035f829813087623201` |
| `analysis/ch3m/eval_m.py` | `eda887257f8fcb731ccf98e84afade6a9edb5330d28da7ecbddbcf015c4f6372` |
| `analysis/ch3m/final_m.py` | `afdbdbf8312c36538307fb98f39ff924ebb719d7c44e5a1875ab539777708ed9` |
| `analysis/ch3m/launch_m.py` | `7dcd713c11ad00011341c742057e04a820e77b492bc39664605131b7fc432a55` |
| `analysis/ch3m/predict_m.py` | `3ffbcba2e519c85babd9326bfbe5aed74b7579c2660b57cc99584913726ddf55` |
| `analysis/ch3m/prereg_m.py` | `ba8772eb64e40421c6e49bf3da47f3fc6ed5fd7a5bde45bcaa80aa2a95ac134b` |
| `analysis/ch3m/s0_m.py` | `78ed96fcec3b212c8c477a4a351048739f07ffc34e4db9de9b9118ab6753d56d` |
| `analysis/ch3m/tests_m.py` | `1126f5a28820034c61d1bf8714796aec8f383a55726fde02badbf2da8b5c3f82` |
| `analysis/ch3m/train_m.py` | `fad1f1116954c27409a0b6d2510857e955d83db0185db0e1f1d196463b684446` |
| `analysis/ch3m/xgb_m.py` | `a92c111c31180d8a31c2b9c65d0a3009ba5763e164f68e31cc02648a77daf6a3` |
| `analysis/ch3m/xgb_pack.py` | `fb063f9c23d868ae26f56cf5e915e66f7eba408a3db8a6754456f0f4f2acbeaa` |
