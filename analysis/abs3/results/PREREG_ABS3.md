# PREREG ABS3 — 반응 하나의 3채널 절대값 모델(TF · GNN · XGB) 대 짝(ΔΔ) 모델 사전 등록 (2026-11-11)

`docs/specs/ABS3_MODELS.md` 6절의 사전 등록이다(명령서 sha256 `9218bbae849bffcf61bc571f48b80df2622585acc3f0cdf380da909fd3002e6a`). `analysis/abs3/prereg_abs.py`(abs3-prereg-1)가 결과 파일만 읽어 만들었다(git HEAD `d65450d4ca4397fc8c14b5685cecf14ac68f736d`, 작성 2026-11-11 09:30:06). 인용 블록 = 명령서 원문. **K(`train_abs.py run`, `xgb_abs.py run`)는 이 md와 `prereg_abs3.json`이 commit되고 바뀌지 않았을 때만 돈다**(`prereg_abs.py check`).

## 1. 질문, 가설 H2, 실험 칸, 짝 모델과 다른 점, 공통 설계 (0절 전체)
> ## 0. 무엇을, 왜
>
> ### 0-1. 목표 (사용자 결정 2026-11-11)
>
> 짝 모델은 두 반응(치환 반응 a와 같은 계열의 H 기준 반응 b)을 함께 넣고 **차이 ΔΔ(a − b)**를 낸다.
> 절대값 모델은 **반응 하나**를 넣고 그 반응의 3채널 값을 낸다. 같은 입력 · 라벨 · 분할에서 세 모델(TF · GNN · XGB)로 두 방식을 비교한다.
>
> | 질문 | 내용 |
> |---|---|
> | **Q-A** | 절대값 모델은 반응 하나의 3채널을 얼마나 맞히는가? 짝 모델은 이것을 할 수 없다 |
> | **Q-D** | 절대값 모델의 예측을 빼서 만든 ΔΔ = ŷ(a) − ŷ(b)가 짝 모델의 ΔΔ보다 나은가, 같은가, 나쁜가? 모델 종류마다 답한다 |
> | **Q-C** | Q-D의 결과를 오차 상쇄로 설명할 수 있는가? 같은 계열 두 반응의 절대값 오차가 얼마나 같이 움직이는가를 본다 |
>
> - **사전 가설 H2:** 짝 모델이 ΔΔ에서 더 낫다. 짝으로 묶으면 계통 오차가 상쇄되기 때문이다.
>   - 근거: DeepDelta(Fralish et al., *J. Cheminform.* 2023, doi:10.1186/s13321-023-00769-x)는 두 분자를 함께 넣는 짝 모델이 분자 하나 모델(ChemProp, Random Forest)보다 성질 차이를 더 잘 맞혔다고 보고했다. 10개 ADMET 벤치마크 중 Pearson r 기준 70 %, MAE 기준 60 %에서 그랬다. 이유로 계통 오차 상쇄를 들었고, 작은 데이터(< 1,500)에 가장 맞는다고 했다.
>   - 이번 데이터는 약 8,000반응으로 더 크므로, 그 이점이 줄어들 수 있다.
> - 이 실험은 **비교**만 한다. 최종 모델(W)과 배포 모델 교체는 범위 밖이다(13절).
>
> ### 0-2. 실험 칸
>
> | 모델 | 절대값 (새로) | 짝 ΔΔ (재사용) |
> |---|---|---|
> | TF | `A__lf3__S__TF__f{0..4}` | TF3 = `analysis/ch3/results/K3_predictions.parquet` |
> | GNN | `A__lf3__S__GNN__f{0..4}` | GNN3 = `analysis/ch3m/results/KM_predictions.parquet`(rung lf3, GNN) |
> | XGB | `A__lf3__S__XGB__f{0..4}` | XGB3 = 같은 파일(rung lf3, XGB) |
>
> 입력은 짝 모델과 같은 재생성 lf 입력 `$CH3/lf`(g2 구조, 라벨 없는 마무리, xTB 특성, DFT · HF 없음)이다.
>
> ### 0-3. 짝 모델과 다른 점 (이것만 다르다)
>
> | 항목 | 짝 모델 | 절대값 모델 |
> |---|---|---|
> | 학습 단위 | 짝 (a, b) | 반응 하나 |
> | 출력 | ΔΔ 4개 | 절대값 4개 |
> | 정답 | `F_pairs_cr2.csv`의 `dd_<t>` @ M | 반응별 cr2 라벨 @ M (4절) |
> | 머리 | `½[f(r_a, r_b) − f(r_b, r_a)]`, 입력 2·d_r | `f(r)`, 입력 d_r. 은닉층은 같게(2층, 폭 2 × width) |
> | 표준화 | ΔΔ의 표준편차(평균 0) | 학습 반응의 평균과 표준편차 |
> | 증강 | (a, b) ↔ (b, a) 순서 바꾸기 | 없음 |
> | XGB 특성 | [Δx / s, (x̄ − μ) / s], 양방향, 반대칭 | (x − μ) / s, 한 방향 |
> | TF 교차 주의 | 있음: `cross` 블록에서 a의 토큰이 b를, b의 토큰이 a를 본다 | **없음**: 반응이 하나라 볼 상대가 없다 |
>
> 인코더는 `fu2/models.py`의 것을 그대로 쓴다(토큰 29 + 전역 182, "A+G").
> - GNN: `readout` = 원자 합 + 주의 풀링 + 전역 투영.
> - TF: `tokens` → `pooled`(평균 + 주의 풀링). 짝 TF의 `cross` 블록은 빠진다. 그래서 TF 비교에는 "교차 주의가 있느냐"라는 구조 차이가 함께 들어간다. 요약에 이 점을 적는다.
>
> ### 0-4. 공통 설계
>
> | 항목 | 값 |
> |---|---|
> | 반응 집합 | 짝 모델 분할의 각 목록(train · test · inner_train · inner_val)에 들어 있는 사용 가능 짝(`DeployData('lf')`, P_lf)의 반응 a와 b를 모두 모은 것. T1과 inner 분할은 core 단위로 묶여 있으므로 train과 test 반응 집합이 겹치지 않아야 한다(4절 확인). 그래서 절대값 모델은 짝 모델과 **같은 반응**을 보고 같은 반응에서 시험받는다 |
> | TF · GNN 학습 | 짝 모델(DEPLOY 11-2 = FU2 M-1, ch3 3모드)과 같은 값: grid(TF 2/3층 × 64/128, GNN 4/6층 × 64/128), AdamW 1e-3 / wd 1e-4, cosine 300 epoch, dropout 0.1, patience 30, seed 20261019 + 0..4 평균. 튜닝은 inner_train / inner_val, 검증 지표 = 4타깃 표준화 MAE 평균(ch3 3모드와 같은 정의). final = 바깥 학습 반응, 튜닝 최적 epoch |
> | 배치 | 반응 64개(짝 모델의 "64짝"과 같은 숫자; 기록) |
> | 손실 | FU2 `loss_fn`과 같은 형태: 표준화 L1 + λ · \|ŷ_barrier − (ŷ_strain + ŷ_inter + ŷ_solv)\| / s_barrier. 여기서 ŷ는 kcal(= μ + s · 출력)이고 λ = `fu2_common.LAMBDA` |
> | 토큰 수치 표준화 | FU2와 같은 함수를 쓰고, 통계는 그 fit의 **학습 반응만**으로 낸다 |
> | XGB | `train_ml_single.GRIDS["XGB"][0]`를 clone하고 `d1_train.grid_cells("XGB")` 12칸으로 바꾼다(짝 XGB와 같은 추정기와 grid). 타깃마다 따로: inner 4 fold(짝 XGB와 같은 규칙인 거르기 전 바깥 학습 짝 목록 → core GroupKFold(4) → 사용 가능 짝 → 그 짝들의 반응)에서 평균 MAE로 칸을 고르고, 바깥 학습 반응 전부로 final fit. X = 182열(`$CH3/lf/X182_lf.parquet`)을 학습 반응 평균 · 표준편차로 z-score한 것. y = 표준화 |
> | 관문 | C-5 누출(시험 반응이 학습 반응 · 표준화 통계 · inner 집합에 없음). 반대칭 관문은 해당 없음 |
> | GPU | RTX 3090(`-p gpu1 --gres=gpu:rtx3090:1`), 짝 모델과 같은 종류 |
>
> ---

## 2. 평가: 지표 정의와 해석 규칙 (8절)
> ## 8. Phase E — 평가 (`eval_abs.py run`)
>
> ### 8-1. 공통
>
> 정답 확인과 집합 · 부트스트랩 규칙은 CH3M 8-1과 같다.
> - 정답 확인: 짝 정답은 `F_pairs_cr2 @ M`, 반응 정답은 4절 y4. 다르면 STOP한다.
> - 집합: P_main · para 공통 짝이 주 집합이고, all은 보조다.
> - 부트스트랩: core 단위 2,000회, seed 20261019.
>
> ### 8-2. Q-A — 절대값 정확도 → `EA_abs.csv`
>
> - **대상:** 시험 반응 전부(fold마다 그 fold의 시험 반응). 모델 3개 × 타깃 4개.
> - **지표:** MAE, RMSE, r, R², MAE의 core 부트스트랩 95 % CI.
> - **기준값:**
>   - 학습 반응 평균으로 예측했을 때의 MAE(fold마다 계산)
>   - 정답 표준편차
> - **나눠 보기:** dipole별, 부착(direct · o · m · p)별, 치환 반응 대 H 기준 반응.
> - **주의:** T1은 core(dipole × 골격) 단위로 떼어 놓은 분할이라, 시험 반응은 **처음 보는 core**의 반응이다.
>
> ### 8-3. Q-D — ΔΔ에서 절대값 모델 대 짝 모델 → `ED_compare.csv`
>
> 모델 종류마다 A-x와 x3을 비교한다(TF: A-TF 대 TF3, GNN: A-GNN 대 GNN3, XGB: A-XGB 대 XGB3).
>
> - **지표:** P_main para(주)와 all(보조)에서 타깃 4개마다 상대 차 = (MAE(A-x) − MAE(x3)) / MAE(x3), 95 % CI.
> - **해석 규칙 (CH3M 8-2와 같음):**
>   - 흐름 기준 noise_t = CH3M `EM_noise.csv`의 같은 위치 · 타깃 \|TF10 대 dep\| 값이다.
>   - 문턱은 max(noise_t, 2 %)이다.
>
> | 표시 | 조건 |
> |---|---|
> | 절대값 나음 | 95 % 상한 < 0 이고 점추정 < −문턱 |
> | 짝 나음 | 95 % 하한 > 0 이고 점추정 > +문턱 |
> | 같음 | 그 밖 |
>
> - **추가 한 줄:** 가장 좋은 절대값 모델(4타깃 para MAE 평균 최소) 대 GNN3(현재 가장 좋은 짝 모델)을 같은 형식으로 적는다.
>
> ### 8-4. Q-C — 오차 상쇄 → `EC_cancel.csv`
>
> 절대값 모델마다 P_main para · all 시험 짝에서 타깃마다 다음을 계산한다.
> - corr(e(a), e(b)): 같은 짝 두 반응의 절대값 오차 상관
> - 상쇄 비율 = 평균 \|e(a) − e(b)\| / 평균(\|e(a)\| + \|e(b)\|). 낮을수록 차이를 낼 때 오차가 많이 지워진다는 뜻이다.
> - 8-3의 상대 차를 함께 적는다.
>
> H2에 맞는지는 표로만 보고하고, 검정하지 않는다.
>
> ### 8-5. 보조 지표 (판정 아님, 6개 모델: A-TF · A-GNN · A-XGB · TF3 · GNN3 · XGB3)
>
> 정의는 `eval_ch3.py` 그대로 쓴다(P_sens 기준 공통 짝, 예측 자신의 barrier 부호).
>
> | 산출 | 내용 |
> |---|---|
> | `EA_metrics.csv` | ΔΔ: 타깃 × 위치별 MAE, RMSE, r, 부호 정답률(\|y\| ≥ 1) |
> | `EA_rank.csv` | 계열 안 장벽 순위: 두 작용기 순서 정답률(0.41–1 / 1–2 / > 2), 실제 최저가 예측 1등 · 상위 3, Spearman |
> | `EA_cause.csv` | 주도 채널 · 유형(S / SI / I) 일치율. 반응 하나와 칸 평균, 기준값 함께 |
> | `EA_identity.csv` | 반응 단위 \|ŷ_barrier − (ŷ_strain + ŷ_inter + ŷ_solv)\|(절대값 모델)와 짝 단위 같은 값(6개 모델) |
>
> ### 8-6. selftest (`eval_abs.py selftest`, K 전에 실행)
>
> 같은 코드로 CH3M `EM_metrics.csv`의 GNN3 · TF3 · XGB3 P_main para MAE 행과 `EM_rank.csv`의 세 행을 소수 넷째 자리까지 재현한다. 다르면 정의 차이를 보고한다(STOP 아님).
>
> 산출: `results/EA_selftest.json`, `EA_report.json`(8-2 · 8-3 · 8-4 요약, 표시, 입력 sha256)
>
> ---

### 2-1. 구현 정의 (`eval_abs.py` DEFINITIONS, K 결과를 보기 전에 고정)

- **models**: A-TF / A-GNN / A-XGB = 절대값 모델(KA_predictions, 반응 하나의 strain · inter · solv · barrier); 짝 ΔΔ = yhat(a) − yhat(b) (같은 fold 모델, KA_pairs); TF3 = ch3 K3_predictions; GNN3 / XGB3 = CH3M KM_predictions rung lf3
- **truth**: 반응 정답 = labels_d1_cr2.json의 10 키 @ M; 짝 정답 = F_pairs_cr2 dd_<10> @ M (eval_ch3.labels). 다르면 STOP
- **pairs**: 6개 모델이 모두 예측한 공통 시험 짝; 주 집합 = P_main(reference & in_P_main) · para, 보조 = P_main all
- **q_a**: EA_abs: 시험 반응 전부(fold마다 그 fold의 시험 반응, KA_predictions), 모델 × 타깃 × 묶음(all, dipole, attach, role H / sub): n, MAE, MAE의 core 부트스트랩 95 % CI(eval_geo2.boot_diff(e, 0, core), 2,000회, seed 20261019; core = 반응이 속한 짝들의 S_folds core), RMSE, Pearson r, R² = 1 − SSE / SST(묶음 안 y 평균 기준), 기준값: 학습 반응 평균 예측의 MAE(fold마다 그 fold 학습 반응의 타깃 평균), 정답 표준편차(ddof 0)
- **q_d**: ED_compare: 모델 종류마다 A-x 대 x3 (TF: A-TF 대 TF3, GNN: A-GNN 대 GNN3, XGB: A-XGB 대 XGB3), eval_ch3.compare (P_main 위치 p · m · all, 상대 차 r = (MAE(A-x) − MAE(x3)) / MAE(x3), core 부트스트랩 2,000회 seed 20261019 95 % CI). 표시(p · all 행): 문턱 = max(noise_t, 0.02), noise_t = CH3M EM_noise.csv 같은 위치 · 타깃의 |r(TF10 대 dep)|; 절대값 나음 = 상한 < 0 & r < −문턱; 짝 나음 = 하한 > 0 & r > +문턱; 그 밖 같음. 추가 줄: 4타깃 P_main para MAE 평균이 가장 낮은 절대값 모델 대 GNN3, 같은 형식
- **q_c**: EC_cancel: 절대값 모델마다 P_main para · all 공통 시험 짝, 타깃마다 e(r) = yhat(r) − y(r)(반응 정답); corr(e(a), e(b)) (Pearson); 상쇄 비율 = 평균 |e(a) − e(b)| / 평균(|e(a)| + |e(b)|); MAE(a), MAE(b), MAE(ΔΔ) = 평균 |e(a) − e(b)|; ED_compare의 상대 차를 함께 적음
- **aux**: EA_metrics / EA_rank / EA_cause: eval_ch3 정의 그대로(공통 짝, 9-3 주 집합 P_sens_ref, 예측 자신의 barrier 부호); EA_identity: 반응 단위 |yhat_barrier − (yhat_strain + yhat_inter + yhat_solv)|(절대값 모델, 시험 반응)와 짝 단위 같은 값(6개 모델, 공통 짝)

### 2-2. selftest (`results/EA_selftest.json`, 8-6)

- EM_metrics P_main para MAE rows (GNN3 · TF3 · XGB3, 4 dp): 통과
- EM_rank para rows (GNN3 · TF3 · XGB3, 4 dp): 통과

## 3. 지금까지 본 것
- K 학습 결과: **0개** (`$ABS3/units/A__*`, `results/KA_*` 없음; 2026-11-11 09:30:06)
- 본 것: 반응별 라벨 · 분할 검사(LS), 짝 모델 예측(TF3 · GNN3 · XGB3, CH3M에서 이미 공개), C 검사(과적합 시험 64반응, smoke 2 epoch / 2칸)
- LS: all_pass True, 반응 7610개; C 검사: all_pass True

## 4. 입력 · 기준 파일 sha256

| 파일 | sha256 |
|---|---|
| `/gpfs/tmp_cpu2/yeseo1ee/ch3/lf/X182_lf.parquet` | `cd5ec799f088a72aa31a2682240f0657469d07db5f5fc56050aa92765fb7f199` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/ch3/results/lf_inputs/fu2_lf/T_manifest.csv` | `d9c914380a235aae919d503bd5eed2034cfba5c20073c519476dfd0be5bacf72` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/ch3/results/lf_inputs/fu2_lf/T_report.json` | `5eecddf5319d7be77c809dd85a09ae79d4643961bf0dc135582e1b9bb62b732e` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/ch3/results/lf_inputs/fu2_lf/Q_report.json` | `b431c5c9e82aed5974e5d5944e9b45f603e5877273229ee7792eb3ff0b17c420` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/F_pairs_cr2.csv` | `f2fe5cab1c61f082b7cf929ca1146d96f8e451d37ed7d4ee2b5b127b4a7fa6ce` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/labels_d1_cr2.json` | `7e5882bce4d47dcd8164b61320a390530463bfa14731029c122d7f09920b6755` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/d1_ladder/results/labels_d1_cr2.parquet` | `696d2ec6ee1b75b5f33bad91fcf9368c528644eecdde4c3d226be3200cd447df` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/d1_ladder/results/V_report.json` | `d4b2c0dfd340cb81596352dd6ca74452c1cce4a1d8601e05170abe1097a46f1f` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/d1_pair/results/fu2/S_splits.json` | `eefe495a775babf2604be6f21156e4ed2869a1988bd08d1542c0d7d4b7779413` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/d1_pair/results/S_folds.csv` | `944a1d1439eb0d54dff90f58f40476c21109cf3a5b71b4bb0f755651b216dcb7` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f0.json` | `8613360436df1cf55f4929c0e57d91e2b74a7a3579489315b0548c7f22f54521` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f1.json` | `28c40b0c4c2b8fa3f12bc753cf3976b5851e16327730527a26b3392dd3c8dc4c` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f2.json` | `0f1939c9e3b8194785205dacbc10eec725cdcd93ae89b2e784f726dac334735e` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f3.json` | `992a7ba21d4ac57948d78b83f5fc90efdcc82ea03a9667b548a8ea5c1665ff27` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f4.json` | `168b876bd111a3e7a078b493db50a75b4db7ae09b6cd0e6eb51eb32b64bafec6` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/ch3/results/K3_predictions.parquet` | `4faefc9c5e38c1a8e6b1592c1c3a476cc31c9edfa1d1fea9e8ce388cde0fa24e` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/ch3m/results/KM_predictions.parquet` | `da01baf40b125f75b03cf5f8e79b37468ddd87cca36c0196334952d63ea6ea6e` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/ch3m/results/EM_noise.csv` | `44645f98876d50981f8cb5ebd8dd2e7efdcd2bd58d491cf4a1861c014a7497c5` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/ch3m/results/EM_metrics.csv` | `266f21e8f40a421353107363d9c382d7ee79c4b00d3ff9f65b4f5fbde00e5c56` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/ch3m/results/EM_rank.csv` | `ec665a55be58a44f13bd959c7b563ec1da357520017d9300bcac2c6201e70c62` |
| `/gpfs/tmp_cpu2/yeseo1ee/abs3/pack/lf_abs.npz` | `a9b150a2ced4aae0bb1268aa19c045126959045bde9ff87624bb5c77110fc6d7` |
| `/gpfs/tmp_cpu2/yeseo1ee/abs3/pack/lf_abs.json` | `0629793b7c0dce5d16e324b88895cf65f4e3e40e12bfbc6b27d295f8536d76ff` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/abs3/results/LS_report.json` | `8a1e86214fce57bfa20dd692b3c6fb03a7cedcdbbc898a5cb96591597b5564df` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/abs3/results/C_tests.json` | `bf3ba550c1d6fea1e162a12e7eefde488f2c674a9936d3c94ee3ae9ef3ae96d0` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-abs3/analysis/abs3/results/S0_start.json` | `771d9214ec09dc2ac02207fc343466387c093c1719c9efa5fdca2b00aebceb30` |

## 5. 코드 sha256 (작성 시)

| 파일 | sha256 |
|---|---|
| `analysis/abs3/abs_common.py` | `747cce705643f314378a87dce1ec2edae3a754218f4cd17a87b8987f7563a376` |
| `analysis/abs3/abs_data.py` | `927919544f3c43fe8c5ed437dacbdf71bca8ee06b1cd09d5ba963085ee96199d` |
| `analysis/abs3/abs_model.py` | `ed72abe8b07ef1ddaef4056fba15c2ea74da57180875441402fc68e41c51886a` |
| `analysis/abs3/abs_pack.py` | `b528bfc9f67489fadae661ea5f3e9bec60e0a4c0b659272f37855f48c036175c` |
| `analysis/abs3/abs_pairs.py` | `95930208bd1f0f8de1d370b71bfe442843708b519b210dfabf09bc63edef31d5` |
| `analysis/abs3/abs_xgbcore.py` | `00720764c11acc692ba07582adf3258dc3c2f0c73cce7b6712642dd1c1cbc51e` |
| `analysis/abs3/commit_abs.py` | `5c3b5aed52236e26933b57be6b5efbb7fee9332b9921b9ac7d62077a5d34b68b` |
| `analysis/abs3/devtest_abs.py` | `b62b5721af710366efddc7b2247d22363930b61f20aa2b9310277c957dbd6fb7` |
| `analysis/abs3/eval_abs.py` | `592174e1cdc019b9f5c44ae41449c1506251c1c97912ac4e0b63990e89a11be8` |
| `analysis/abs3/launch_abs.py` | `37b6f20d010f03b84690518fd326c397764cfa62bfa971ae0684d5b5163136ff` |
| `analysis/abs3/prereg_abs.py` | `b038ec2ea6f886039525c581c5701a05851e9a10b0f3b5b71fa4228508727e58` |
| `analysis/abs3/s0_abs.py` | `5618d30d8ae213901e0a2f15b3ceceb7ef4aafb7d4f042fa4315bde0bfa95a12` |
| `analysis/abs3/tests_abs.py` | `7fb600978533db319a6474caef72e39f33e9f6cf265119fc9dd13ddce4cd0164` |
| `analysis/abs3/train_abs.py` | `3fd50aa4acfb92326c9100531cd950fa1943b6f9837e59fb99336675a93177f5` |
| `analysis/abs3/xgb_abs.py` | `ec398d258707bd86e6c65a491955cca5e698238afd1a2858b4390d9cf3562215` |
| `analysis/abs3/jobs/cpu_abs.slurm` | `1dbc8bf60a8a3cf5d0fbec88ff9b1d05289900db01106d41e7f36d78582b02a8` |
| `analysis/abs3/jobs/gpu_abs.slurm` | `f95a46630cf7883b86e846b0f9505ce4e096abfe546e315fa660f3e8bfb2a128` |
| `analysis/abs3/jobs/s0_abs.slurm` | `94599a1d4dac8c29fbedb3a9240b47190d7b3854f98d742ae3a5240a7e48d58b` |
