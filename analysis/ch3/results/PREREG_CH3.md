# PREREG CH3 — 3채널(변형 · 상호작용 · 용매) + 장벽 모델 사전 등록 (2026-11-10)

`docs/specs/CH3_MODEL.md` 7절의 사전 등록이다(명령서 sha256 `20fb95dba2b948a9905c313fcb97b04a0db570ebd4592fc84e30df5de2c64932`). `analysis/ch3/prereg_ch3.py`(ch3-prereg-1)가 결과 파일만 읽어 만들었다(git HEAD `e3731a315dcaa7547ee6aa13f3c81c0b8e315aa9`, 작성 2026-11-10 13:48:00). 규칙은 명령서를 그대로 인용한다(인용 블록 = 원문). **K 학습(`train_ch3.py run`)과 W(`train_ch3.py final`)는 이 md와 `prereg_ch3.json`이 commit되고 바뀌지 않았을 때만 돈다**(`prereg_ch3.py check`).

## 1. 설계 (0-2절)
> ### 0-2. 확정 설계
>
> | 항목 | 값 |
> |---|---|
> | 타깃 (4개) | `strain` = d1 + d2 · `inter` = elst + pauli + oi + disp + c_ghost · `solv` = cpcm + cds · `barrier` |
> | 쉬운 말 | 비트는 비용 · 끌고 미는 힘 · 물의 효과 · 장벽(검산과 순위용) |
> | 라벨 | `$LAD/F_pairs_cr2.csv`의 `dd_<t>`를 위 식으로 더한 값. strain + inter + solv − barrier 잔차 최대 ≈ 0.1 kcal/mol (라벨 자체의 수치 잡음) |
> | 입력 | lf · S arm: 원자 토큰(d_in 29) + 전역 182열. DFT 값 없음. deploy 코드로 다시 만든다(3절) |
> | 모델 | `Mo.build("TF", cell, d_in=29, n_out=4, d_glob=182, "A+G")` |
> | 학습 규칙 | DEPLOY 11-2와 같음. 바뀌는 곳 3개뿐: 출력 10 → 4, 항등 벌점 Σ₉ → Σ₃ (0.1 × 평균 \|ŷ_barrier − (ŷ_strain + ŷ_inter + ŷ_solv)\| / s_barrier), 튜닝 검증 지표 = 4 타깃 표준화 MAE 평균 |
> | 분할 | FU2 T1 5 fold (DEPLOY와 같은 파일) |
> | 비교 기준 | **lf 합**: deploy commit의 `analysis/deploy/results/K_predictions.parquet`(lf 8채널 교차검증 예측)을 위 식으로 더한 값. barrier는 lf 직접 예측 |
> | 최종 모델 | 사용 가능한 D1 짝 전부, 칸 = fold 선택 최빈값, epoch = 최적 epoch 중앙값, seed 20261019 + 0..4 (DEPLOY 13과 같은 규칙) |
> | 단위 이름 | 교차검증 `D__lf3__S__TF__f{0..4}`, 최종 `D__lf3__S__TF__all__s{0..4}` |

## 2. 학습 (8절)과 최종 모델 (10절)
> ## 8. Phase K — 교차검증 학습 (`train_ch3.py run` / `collect`)
>
> | 항목 | 값 |
> |---|---|
> | 단위 | `D__lf3__S__TF__f{0..4}` → `$CH3/units/<unit>.{json,parquet}`, `__s{0..4}.pt` |
> | 학습 규칙 | DEPLOY 11-2 그대로 (grid 2/3층 × 64/128폭, AdamW lr 1e-3 wd 1e-4, cosine 300 epoch, dropout 0.1, batch 64짝, patience 30, (a,b)/(b,a) 순서 무작위 증강, seed 20261019 + 0..4 평균). 타깃 · 벌점 · 검증 지표만 0-2대로 |
> | 멱등 | 단위 json이 있으면 지문(라벨 · 입력 · 분할 · 코드 sha256, GPU 이름) 비교 후 건너뜀. 다르면 STOP |
> | GPU | RTX 3090 (`-p gpu1 --gres=gpu:rtx3090:1`), 배열 `--array=0-4%5`. 예상 약 1 GPU-h |
> | collect | `results/K3_predictions.parquet`. 열은 deploy `K_predictions`와 같고 `rung = "lf3"`, `target ∈ TARGETS3`. `K3_val.csv`, `K3_report.json` |
>
> ---

> ## 10. Phase W — 최종 모델 (`train_ch3.py final`)
>
> 9절 결과와 상관없이 실행한다(약 0.2 GPU-h).
>
> | 항목 | 규칙 (DEPLOY 13과 같음) |
> |---|---|
> | 학습 짝 | `Ch3Data.split("all")` |
> | 칸 · epoch | fold 선택 칸의 최빈값(동률: 튜닝 검증 평균이 낮은 칸 → grid 순서), epoch = 5 fold 최적 epoch 중앙값 |
> | seed | 20261019 + 0..4 |
> | 관문 | C-4 (무작위 학습 짝 200개), C-T |
> | 산출 | `$CH3/model/D__lf3__S__TF__all__s{0..4}.pt`, `$CH3/model/MODEL_CARD.json`(복사본을 `results/MODEL_CARD_ch3.json`에), `results/W3_report.json` |
>
> `MODEL_CARD`는 DEPLOY 13의 항목에 더해 `targets = TARGETS3`, 변환 행렬 `M`, 9-2 채택 여부를 적는다.
>
> ---

## 3. 평가 지표 정의 전부와 채택 규칙 (9절, 9-2 포함)
> ## 9. Phase E — 평가 (`eval_ch3.py run`)
>
> ### 9-1. 공통
>
> - **두 모델:** `ch3` = K3 예측. `lf_sum` = deploy `K_predictions`의 yhat을 0-2 식으로 더한 값(barrier는 직접 예측).
>   정답 `y`는 둘 다 `$LAD/F_pairs_cr2.csv`. deploy `K_predictions`의 `y`가 이 파일과 같은지 먼저 확인한다(다르면 STOP).
> - **짝 속성:** D1PAIR `S_folds.csv`(core, fam, sigma, in_P_main, in_P_sens)와 `F_pairs_cr2.csv`의 `dipole`, `cfg`(빈 값 → `-`),
>   `series`, `pos`, `Xb`.
> - **집합:** P_main = reference & in_P_main (주 비교). 두 모델이 모두 예측한 짝만 쓴다(공통 집합).
> - **부트스트랩:** core 단위 2,000회, seed 20261019, `analysis/geo2/eval_geo2.py`의 `boot_diff`.
>
> ### 9-2. 주 비교와 채택 규칙 → `E3_compare.csv`
>
> P_main · para 공통 짝에서 타깃마다 d = MAE(ch3) − MAE(lf_sum), 상대 차 d / MAE(lf_sum), 95 % CI.
>
> **채택 규칙 (사전 등록):**
> - strain · inter · solv 각각 상대 차 점추정 ≤ 0 **그리고** 95 % 상한 ≤ +5 %
> - barrier는 95 % 상한 ≤ +5 %
>
> 모두 만족하면 **ch3을 배포 모델로 채택**한다. 아니면 STOP하지 않고 두 모델을 나란히 보고하며, 결정은 사용자에게 넘긴다.
>
> ### 9-3. 보조 지표 (두 모델 모두, 판정 아님)
>
> | 산출 | 내용 |
> |---|---|
> | `E3_metrics.csv` | 타깃 × 위치(all, p, m, o, direct): n, MAE, RMSE, 상관 r, 부호 정답률(\|y\| ≥ 1인 짝), para MAE의 95 % CI |
> | `E3_rank.csv` | 계열 안 장벽 순위. 계열 = fam × pos, P_sens reference 짝에 H(0, 0)를 더해 점 ≥ 8. 두 작용기 순서 정답률(실제 \|Δ\| 구간 0.41–1, 1–2, > 2), 실제 최저 작용기가 예측 1등인 비율 · 상위 3 안인 비율(무작위 기대값 3/n 함께), Spearman 중앙값 · 25 % 분위. para와 all |
> | `E3_cause.csv` | **반응 하나:** \|y_barrier\| ≥ 1인 짝에서 (i) 주도 채널 = argmax(strain, inter, solv) × sign(barrier)의 일치율, (ii) 유형 S(strain이 barrier와 같은 방향, 나머지 inter + solv는 반대) · SI(둘 다 같은 방향) · I(나머지만 같은 방향)의 일치율, 각각 가장 흔한 범주만 찍었을 때의 기준값. **칸 평균:** 칸 = dipole × cfg × series × pos × Xb, 짝 ≥ 4. \|평균 y_barrier\| ≥ 1인 칸에서 같은 두 일치율과 기준값. 모든 칸에서 타깃별 칸 평균의 r · MAE |
> | `E3_ad.json` | 적용 범위 문턱: 타깃별 시험 짝 `yhat_seed_sd`의 95 백분위 (ch3) |
> | `E3_report.json` | 9-2 판정, 위 요약, 입력 sha256 |
>
> 참고값 (이번 대화에서 lf 합으로 계산, 같은 정의):
>
> | 지표 | 값 |
> |---|---|
> | 짝 단위 MAE (all) | strain 0.81, inter 0.65, solv 0.52, barrier 0.75 |
> | 부호 정답률 | strain 0.90, inter 0.93, solv 0.84 |
> | 주도 채널 일치율 | 반응 하나 0.72 (기준 0.63), 칸 평균 0.84 (기준 0.71) |
> | 유형 일치율 | 반응 하나 0.69 (기준 0.45), 칸 평균 0.83 (기준 0.53) |
>
> `eval_ch3.py selftest`는 lf_sum으로 이 값을 재현해야 한다(소수 둘째 자리). 다르면 정의 차이를 보고한다(STOP 아님).
>
> ---

- 채택 기준값: strain · inter · solv 상대 차 점추정 ≤ 0 그리고 95 % 상한 ≤ +5 %, barrier 95 % 상한 ≤ +5 % (P_main · para 공통 짝, core 부트스트랩 2,000회, seed 20261019, `eval_geo2.boot_diff`와 같은 재표본으로 상대 차 CI). 못 넘으면 STOP 없이 나란히 보고.

### 3-1. 구현 정의 (`eval_ch3.py` DEFINITIONS, ch3 결과를 보기 전에 고정)

- **pairs**: 공통 집합 = ch3(K3_predictions)와 lf_sum(deploy K_predictions)이 모두 예측한 시험 짝. 짝 속성 = S_folds.csv(core, fam, sigma, in_P_main, in_P_sens, ref_kind, pos) + F_pairs_cr2.csv(dipole, cfg(빈 값 '-'), series, pos, Xb); pos · ref_kind가 서로 다르면 STOP
- **targets**: y(두 모델 공통) = F_pairs_cr2 dd_<10 targets> @ M; lf_sum yhat = deploy K_predictions yhat 10열 @ M (strain = d1+d2, inter = elst+pauli+oi+disp+c_ghost, solv = cpcm+cds, barrier = lf 직접 예측)
- **sets**: P_main: ref_kind == reference & in_P_main; P_sens_ref: ref_kind == reference & in_P_sens; ref_all: ref_kind == reference; test_all: every common test pair (reference + anchor)
- **compare**: E3_compare: P_main · 위치 p(판정) 공통 짝, 타깃마다 d = MAE(ch3) - MAE(lf_sum) (eval_geo2.boot_diff, core 2,000회, seed 20261019), 상대 차 r = d / MAE(lf_sum)와 같은 재표본의 95 % CI (eval_tf.boot_rel); 위치 m · all은 설명용
- **adoption**: strain · inter · solv: r ≤ 0 그리고 r 95 % 상한 ≤ +5 %; barrier: r 95 % 상한 ≤ +5 %. 모두 만족 → ch3 채택
- **metrics**: E3_metrics: 모델 × eval_set (P_main, P_sens_ref, test_all; 9-3 짝 MAE · 부호의 주 집합 = P_sens_ref, para MAE CI = P_main) × 타깃 × 위치(all, p, m, o, direct): n, MAE, RMSE, Pearson r, 부호 정답률(|y| ≥ 1 짝에서 sign(yhat) = sign(y); yhat = 0은 오답), P_main · p의 MAE core 부트스트랩 95 % CI (boot_diff(e, 0, core))
- **aux_set_choice**: 9-3 보조 지표(짝 MAE all, 부호, 원인)의 주 집합은 selftest(lf_sum)가 참고값을 가장 많이 재현한 정의(P_sens_ref = reference & in_P_sens 공통 시험 짝, 예측 자신의 barrier 부호: 15개 중 13개)로 고정했다. P_main 정의는 11/15. 유형 일치율 두 값(반응 하나 0.68 대 0.69, 칸 평균 0.82 대 0.83)은 어느 후보로도 함께 재현되지 않아 정의 차이로 보고한다. 9-2 채택 판정은 명령서대로 P_main para.
- **rank**: E3_rank: 계열 = fam × pos, P_sens_ref 짝(ref_kind == reference & in_P_sens) 중 공통 짝, Xb 중복은 첫 행, H(y = yhat = 0) 추가, 점 ≥ 8. 대상 = barrier. 두 작용기 순서: 계열 안 모든 점 쌍, 실제 |Δy| 구간 [0.41, 1), [1, 2), [2, ∞)에서 sign(Δyhat) = sign(Δy) 비율(쌍 합산). 최저: 실제 최저 점(argmin y, 첫째)의 예측 순위 (= 1 + 그보다 yhat이 작은 점 수)가 1 / ≤ 3인 계열 비율, 무작위 기대값 = 계열 평균 1/n, min(3, n)/n. Spearman(y, yhat) 계열별 중앙값 · 25 % 분위. 범위 para(pos p 계열)와 all(모든 위치 계열)
- **cause_single**: E3_cause 반응 하나: P_sens_ref 공통 짝, |y_barrier| ≥ 1. 주도 채널 = argmax_c (v_c × sign(v_barrier)), c ∈ (strain, inter, solv); 유형: s = sign(v_barrier), S = strain·s > 0 & (inter+solv)·s ≤ 0, SI = 둘 다 > 0, I = strain·s ≤ 0 & (inter+solv)·s > 0, 그 밖 none. 실제는 y, 예측은 yhat(예측 자신의 barrier 부호). 일치율과 기준값(실제 범주 중 가장 흔한 범주의 비율)
- **cause_cell**: E3_cause 칸 평균: P_sens_ref 공통 짝, 칸 = dipole × cfg × series × pos × Xb, 짝 ≥ 4; 칸 평균 y · yhat로 |평균 y_barrier| ≥ 1인 칸에서 같은 두 일치율과 기준값; 모든 칸에서 타깃별 칸 평균의 Pearson r · MAE
- **ad**: E3_ad: 타깃별 ch3 시험 짝(K3 전부) yhat_seed_sd의 95 백분위(numpy linear)

### 3-2. selftest (lf 합으로 9-3 참고값 재현, `results/E3_selftest.json`)

| 지표 | 참고값 | 재현 | 소수 둘째 자리 일치 |
|---|---|---|---|
| 짝 MAE all strain | 0.81 | 0.8113 | 예 |
| 짝 MAE all inter | 0.65 | 0.6485 | 예 |
| 짝 MAE all solv | 0.52 | 0.5186 | 예 |
| 짝 MAE all barrier | 0.75 | 0.7459 | 예 |
| 부호 정답률 strain | 0.9 | 0.8956 | 예 |
| 부호 정답률 inter | 0.93 | 0.9304 | 예 |
| 부호 정답률 solv | 0.84 | 0.8365 | 예 |
| 주도 채널 반응 하나 | 0.72 | 0.7181 | 예 |
| 주도 채널 반응 하나 기준값 | 0.63 | 0.6308 | 예 |
| 주도 채널 칸 평균 | 0.84 | 0.8402 | 예 |
| 주도 채널 칸 평균 기준값 | 0.71 | 0.7078 | 예 |
| 유형 반응 하나 | 0.69 | 0.6836 | 아니오 |
| 유형 반응 하나 기준값 | 0.45 | 0.4478 | 예 |
| 유형 칸 평균 | 0.83 | 0.8174 | 아니오 |
| 유형 칸 평균 기준값 | 0.53 | 0.5297 | 예 |

정의 차이 메모: 유형 반응 하나: 참고 0.69, 주 정의 0.6836; 일치하는 다른 정의 ['P_main/own', 'P_main/sum']. 유형 칸 평균: 참고 0.83, 주 정의 0.8174; 일치하는 다른 정의 ['P_main/true', 'P_sens_ref/true', 'ref_all/true', 'test_all/true'].

## 4. 지금까지 본 것
- ch3로 학습한 교차검증 결과: **0개** (`$CH3/units/D__lf3__*`, `results/K3_*`, `$CH3/model/D__lf3__*` 없음; 2026-11-10 13:48:00)
- 본 것: lf 합(비교 기준, deploy `K_predictions.parquet`)의 9-3 지표(selftest), C-3′(10타깃 deploy 재현, fold 0; 결과 `C3r_regress.json` ok False), C-8 smoke(ch3 모드 fold 0, 2 epoch; 성능 판단에 쓰지 않음, ok True)
- 입력 재생성 확인 V: `results/V_inputs.json` all_pass True

## 4-1. C-3′ 편차 (사용자 결정, `results/C_deviation.json`)

- 사용자: 11-10 13:1x: 추천 방식대로 진행 (C-3′ 실패를 입력 미세 차이 편차로 기록하고 PREREG → K → E → W → I 진행; 조건 D1 재현 · D2 도 ±2 % 이탈)
- C 검사 실패 항목: ['C-3′ 10-target regression vs deploy K fold 0'] — 나머지 C 검사는 모두 통과
- C-3′: 칸 {'n_layers': 3, 'width': 64} (deploy {'n_layers': 3, 'width': 64}), 최적 epoch 차 25, para MAE 상대 차 최대 0.11651884271627252
- D1 deploy C-3 재실행: LADDER 단위와 튜닝 검증값 차 최대 0.0, epoch 차 [0, 0, 0, 0] → 같은 입력이면 학습이 결정적
- D2 seed만 바꾼 fold 0: 칸 {'cell': {'n_layers': 3, 'width': 128}, 'best_epoch': 95, 'index': 3, 'grid_index': 3}, epoch 차 {'null_vs_deploy': 18, 'null_vs_c3r': -7}, ±2 % 안 타깃 수 {'null_vs_deploy': 2, 'c3r_vs_deploy': 6, 'null_vs_c3r': 1}, 상대 차 최대 0.1539623276691455
- 해석: C-3′ 차이는 재생성 입력의 ≤ 8.6e-7 차(진동수 유래 B6 열)가 학습에서 증폭된 것이며, 같은 입력의 seed 차이로도 C-3′ 기준을 넘는다. 조건 1–3이 모두 성립해 K · W를 진행한다(`run_dev.py`).


## 5. 비교 기준 파일과 입력 sha256

| 파일 | sha256 |
|---|---|
| `/gpfs/tmp_cpu2/yeseo1ee/ch3/lf/labels_lf.json` | `e7fdf9b57371f7c8e2851cd7fef4ef131604b6257047bd5e75f55ee326b5f720` |
| `/gpfs/tmp_cpu2/yeseo1ee/ch3/lf/input_meta_lf.csv` | `104ab323de72fb7e4a999744e5957914e3d4dfedefbeddafa8c1bf4941451a09` |
| `/gpfs/tmp_cpu2/yeseo1ee/ch3/lf/X182_lf.parquet` | `cd5ec799f088a72aa31a2682240f0657469d07db5f5fc56050aa92765fb7f199` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3/analysis/ch3/results/lf_inputs/fu2_lf/T_manifest.csv` | `d9c914380a235aae919d503bd5eed2034cfba5c20073c519476dfd0be5bacf72` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3/analysis/ch3/results/lf_inputs/fu2_lf/T_report.json` | `5eecddf5319d7be77c809dd85a09ae79d4643961bf0dc135582e1b9bb62b732e` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3/analysis/ch3/results/lf_inputs/fu2_lf/Q_report.json` | `b431c5c9e82aed5974e5d5944e9b45f603e5877273229ee7792eb3ff0b17c420` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3/analysis/ch3/results/lf_inputs/fu2_lf/Q_manifest.csv` | `b886ca126df69b8c8e905ad7d61dd11d2600a2288922c9c7c5b175b85606b4ad` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/F_pairs_cr2.csv` | `f2fe5cab1c61f082b7cf929ca1146d96f8e451d37ed7d4ee2b5b127b4a7fa6ce` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3/analysis/d1_pair/results/fu2/S_splits.json` | `eefe495a775babf2604be6f21156e4ed2869a1988bd08d1542c0d7d4b7779413` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3/analysis/d1_pair/results/S_folds.csv` | `944a1d1439eb0d54dff90f58f40476c21109cf3a5b71b4bb0f755651b216dcb7` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3/analysis/deploy/results/K_predictions.parquet` | `a7c30339e46b5455972ba4bb93493d3f50f33db4b77d8bc6f3a83a5b53f58616` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3/analysis/deploy/results/K_report.json` | `17bc5ea2faf15f73a34f0cc2a2b0ae8292273ecc2bb274bea297f23a6b4ce9d1` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3/analysis/ch3/results/V_inputs.json` | `b4bb38202a004f93487078196f61c2b51313817bfaa0da63e07de6d4f9b75af0` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f0.json` | `8613360436df1cf55f4929c0e57d91e2b74a7a3579489315b0548c7f22f54521` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f1.json` | `28c40b0c4c2b8fa3f12bc753cf3976b5851e16327730527a26b3392dd3c8dc4c` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f2.json` | `0f1939c9e3b8194785205dacbc10eec725cdcd93ae89b2e784f726dac334735e` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f3.json` | `992a7ba21d4ac57948d78b83f5fc90efdcc82ea03a9667b548a8ea5c1665ff27` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f4.json` | `168b876bd111a3e7a078b493db50a75b4db7ae09b6cd0e6eb51eb32b64bafec6` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3/analysis/ch3/results/C_tests.json` | `ad84d21dd2048fd5987c565ac81484ffad98dd2ae41237d7ef2f900f1eccfd75` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-ch3/analysis/ch3/results/C_deviation.json` | `35e06c50944c385cab3dc2ecb598d035566babaeaed23177190281992217120d` |

- C 검사: `results/C_tests.json` sha256 `ad84d21dd2048fd5987c565ac81484ffad98dd2ae41237d7ef2f900f1eccfd75` (all_pass False; C-3′ 편차 = results/C_deviation.json)
- eval selftest: `results/E3_selftest.json` sha256 `610c6ae39d2dd63356ef74d0a24b4c635aa46fa7d0cc78ee179e1b72596fc87f`

## 6. 코드 sha256 (작성 시)

| 파일 | sha256 |
|---|---|
| `analysis/ch3/c_deviation.py` | `003d8e6af097f2d7961c563c373069f1df8a543b50f03861335384d7efa92c01` |
| `analysis/ch3/ch3_common.py` | `d6b77a4ceb8cfd472de7477daa5799dd9d87eddc6c34c9520f635f3419c50956` |
| `analysis/ch3/commit_phase.py` | `95c94cc00d6e7868557ac61595d3541438486cdf22cdf8c4932882e9174ccc57` |
| `analysis/ch3/data_ch3.py` | `d2469f2c9ac39cd4c24d919ebf5546366c07e5133e63ca2d889aa2fef1c3373c` |
| `analysis/ch3/devtest_ch3.py` | `af64fff21760055fe62917e1d6bd9aed70045abac3586b40528fe5b07d51922c` |
| `analysis/ch3/diag_c3r.py` | `90afb152fee5d7d0fb2efc1c08f73e3f2c504f736309136e89db491cc8fdd3d0` |
| `analysis/ch3/eval_ch3.py` | `b1fccfcf590b5922f99d536f44f594103bb812f6aa46fca8fbb2b227bb383d13` |
| `analysis/ch3/launch_c.py` | `863273d5d51a70bf97230d40e3ffbcbc42928eeb229d678a5426d04e310ba942` |
| `analysis/ch3/launch_k.py` | `a8d162a462a6b964131076b3e6d835c65d7fdf540999112dba354bfde72342a6` |
| `analysis/ch3/predict_ch3.py` | `b549ada0a3505b3902d6cd4c2af1f87df3175ebf7505e76305a24c7cfd9c09d6` |
| `analysis/ch3/prereg_ch3.py` | `00cb1f15ec943d99e2f1ab7b5cf00d523bbb8141021271ad696d9cdb79b41bb9` |
| `analysis/ch3/run_dev.py` | `07c6f4becd0d385bb6fedb1442e6e31014fcb2c6b3dc0deb73dfc4d61a24e2bf` |
| `analysis/ch3/s0_start.py` | `e416790a6afe05e48ccdbb7df891225f2f1c12f7e26c80f587413e22d801812d` |
| `analysis/ch3/tests_ch3.py` | `ad2fb02a4bf1e7a83c926ea2544c006c2fc5b04c358773c03ae42d2e80a41383` |
| `analysis/ch3/train_ch3.py` | `8648071ffde1e835419c501fde34ac7ade67bcf98ada637ab80f52085f0eed90` |
| `analysis/ch3/verify_inputs.py` | `19374804b2252b18dc0beb67f5fcdc12d4934d94c66d21f7ea26c605473ce44e` |
