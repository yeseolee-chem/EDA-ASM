# PREREG ADD1 — D1PAIR 사전 등록 추가본 (FU1, 2026-10-20)

이 문서는 `PREREG_D1PAIR.md`(commit 10f99d8d, 학습 전)에 더하는 추가본이다. 원래 사전 등록은 바꾸지 않는다.
근거: `docs/specs/D1PAIR_FU1.md`, 잠정 T1 결과(`results/interim_T1_noSVR/`, SVR 제외, git 09ed8a47).

## A-1. 결정과 기록

- **F1** T2를 즉시 중단한다(사용자 결정 2026-10-20). 이미 끝난 T2 unit 117개는 지우지 않고 보관하되 분석하지 않는다
  (`$D1PAIR_SCRATCH/M/SKIP_GROUPS` = T2; 취소: 1025347_0 (d1m_nn, NN-2/NN-4, in run T2); 1025512_0 (d1m_svrH, SVR helper, in run T2); 1025512_1 (d1m_svrH, in run T2); 1025512_2 (d1m_svrH, in run T2); 1025351 d1m_collect + 1025352 d1e_all (stale chain: depended on the cancelled tasks); 다시 제출: 1025589 d1m_svrH2 array 0-2 (SVR helpers, same slices, run LOSO ::: LODO ::: T1, reverse order); 1025589_2 was cancelled at once to free a slot for the K check (1025592) and resubmitted as 1025594 (array 2)).
  gate K-a(캐시 지문 30개) 통과, K-b(unit 목록) 통과.
- **F2** learning curve를 돌린다(arm C, 탐색). **F3** XGB 탐색 범위를 넓힌다(arm C, 모델 XGBW, 탐색).
- **D-1** T2 중단(이탈, `results/DEVIATIONS.md`).
- **LODO·LOSO 결과는 이 commit 전에 계산·열람하지 않았다.** `results/`에 Phase E 산출물(`E_*`, `M_*`)이 없음을 확인했다
  (확인 시각 2026-10-20 09:18). LODO·LOSO unit은 학습 중이며 캐시(scratch)에만 있다. 예외로 기록할 것:
  상태 점검 중 학습 로그의 마지막 줄 출력에서 LOSO unit 하나(arm B, NN-4, elst, fold 4)의 시험 MAE 한 값이 화면에 한 번
  보였다. 집계·비교(B 대 C 포함)·판정은 계산하지도 보지도 않았다. 이후 점검은 개수만 보도록 바꿨다.

## A-2. 확증 비교 추가

- T1의 "C < B"는 결과를 본 뒤 나온 관찰이라 확증이 아니다. **아직 보지 않은 LODO와 LOSO에서** 확인한다.
- 비교: split(LODO, LOSO) × 9 타깃(d1, d2, elst, pauli, oi, disp, cpcm, cds, barrier)마다 arm C 대표 모델 대 arm B 대표 모델.
  대표 모델 = PREREG 규칙(6개 모델 중 바깥 fold 평균 안쪽 CV MAE 최저, SVR 포함).
- 지표: P_main 시험 짝 MAE 차이 d = MAE(B) − MAE(C). 95 % CI와 양측 p: 그 split의 시험 짝에서 core 단위 cluster bootstrap 2,000회
  (seed 20261019; p = min(1, 2 min((1 + #{d* ≤ 0})/(B + 1), (1 + #{d* ≥ 0})/(B + 1)))). 18개에 Holm 보정. 가설 B − C > 0;
  "C가 낫다" 확증 = Holm p < 0.05 이고 CI 하한 > 0.
- SVR까지 모두 끝난 뒤 Phase E와 함께 계산한다(`d1_eval.add1_tests` → `results/E_add1_tests.csv`).

## A-3. 탐색 분석의 설계와 판독 규칙 (L, X)

결과는 **탐색**이며 PREREG의 대표 모델·arm·단계 판정을 바꾸지 않는다.

### Phase L — learning curve
- arm C, 모델 KRR·XGB(PREREG grid), 타깃 10개, T1 바깥 fold 0–4; 학습 비율 f ∈ {0.25, 0.5, 0.75} × 반복 r ∈ {0, 1, 2}
  (900 unit); f = 1.0 = 끝난 T1 unit(fingerprint 재확인).
- core 단위 축소: fold k의 학습 쪽 core(기준 짝 없는 core 포함)를 이름순 정렬 → `numpy.random.default_rng(20261019 + 100 k + r)
  .permutation` → 앞에서 ceil(f × core 수)개(같은 (k, r)에서 0.25 ⊂ 0.5 ⊂ 0.75). 고른 core의 학습 짝 전부. 시험 짝 = T1 unit과 같음.
  학습 core < 4면 STOP. 튜닝·스케일·반대칭·gate M-1 = d1_train.
- 판독: 모델 × 타깃 × 비율마다 학습 짝 수, P_main 시험 MAE(전체, para; 반복 평균·범위), 안쪽 CV MAE.
  **판정:** f = 0.5 → 1.0의 para MAE 감소율이 ≥ 10 %면 "데이터를 더 늘리면 아직 좋아진다", < 3 %면 "포화 — feature가 한계",
  그 사이 "불확실". 감소율 ≥ 3 %일 때만 참고 외삽: f = 0.5, 0.75, 1.0의 평균 para MAE에 MAE = a·n^(−b)를 log-log 최소제곱으로
  맞추고, 채널별 목표 MAE(사후 정의, 꺾임 크기 중앙값 × 0.28: barrier 0.16, d1 0.19, d2 0.04, elst 0.21, pauli 0.41, oi 0.29, cpcm 0.22, cds 0.01, disp 0.05)에 닿는
  학습 짝 수. 외삽은 참고용.

### Phase X — XGB 넓은 grid
- arm C, 모델 XGBW, 타깃 10개 × T1 fold 5 = 50 unit. grid 36칸: n_estimators {500, 1000, 2000, 4000} × learning_rate
  {0.05, 0.1, 0.2} × max_depth {5, 7, 9}; 나머지는 rev 6 XGB 설정(random_state 42, n_jobs 1, tree_method hist). 양방향 학습,
  ½[h(Δx) − h(−Δx)]. 안쪽 GroupKFold(4), 최저 평균 안쪽 MAE, 같으면 grid 순서. gate M-1.
- 판독: 같은 시험 짝에서 C/XGB 대 C/XGBW의 P_main MAE(전체, para), 안쪽 CV MAE, 차이(XGB − XGBW)의 core bootstrap 95 % CI(2,000);
  Hammett 수준(R²_fam 중앙값, 채널별 꺾임 민감도·위양성률, 목표 MAE 대비 배수); 고른 값 분포와 grid 끝 걸림.
  **어떤 한 값이든 50 %를 넘는 unit이 그 값의 grid 최댓값(4000, 0.2, 9)을 고르면 STOP**(보고 후 사용자 결정; 2026-10-20 사용자
  지시에 따라 추천 옵션으로 진행).
