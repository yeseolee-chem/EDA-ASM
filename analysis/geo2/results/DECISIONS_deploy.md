# DECISIONS — GEO2b DEPLOY: LADDER dft-rung 에너지 모델 × geo-rung 입력 (2026-11-03)

- 작성 시점: 코드만 쓰고 아무것도 실행하지 않은 상태(geo 입력으로 돌리기 전). 사용자 지시(2026-11-03): 갈림길마다
  추천 / 가장 단순한 선택을 하고 여기에 적는다. 코드: `analysis/geo2/deploy/`(deploy_common, predict_atoms,
  predict_xgb, compat_deploy, eval_deploy). LADDER·GEO2b 결과와 scratch는 읽기만 한다. 쓰는 곳은
  `$GEO2B/deploy/`와 `analysis/geo2/results/DEPLOY_*`뿐이다.

## DP-1 세 경우
- (a) `dft`: LADDER dft rung 40 단위(AW·GNN·TF·XGB × S·SH × T1 fold 0–4)를 DFT 라벨-TS 입력에 적용한 값
  = `analysis/d1_ladder/results/L_predictions.parquet`의 dft 행.
- (b) `dft_on_geo`(배포): 같은 단위를 geo-rung 입력(GEO1b 보정 TS의 token, X182, HF/6-31G* H)에 적용한다.
  fold k 짝은 fold k 단위로만 예측한다(다른 fold의 DFT 구조로 학습). geo 구조 누수 검사: 시험 짝 양 끝 반응의
  `Z_status.csv` source가 f{k}가 아니면 STOP.
- (c) `geo`: GEO2b geo-trained 단위(`L_predictions_geo.parquet`의 D1 행).
- 재학습 없음. 예외: XGB는 모델 파일이 없어서 LADDER `train_xgb.py` 경로(prep_fit + _task)로 unit json의
  선택 cell을 써서 final fit만 다시 한다(같은 데이터·순서·random_state; GEO2b R-3에서 전체 재학습이 차이 0).

## DP-2 회귀 gate (geo 입력 결과보다 먼저, 같은 job 안에서)
- 원자 모델: 저장된 5개 `.pt`를 불러오고, 같은 학습 짝으로 Std(SH: StdSH)를 다시 계산해서 저장값(mu, sd, gmu, gsd,
  s_y, s_r, β)과 완전히 같은지 본다(다르면 STOP). DFT 입력 시험 짝을 `train_atoms.final_kcal`로 예측하고 5 seed
  평균을 yhat으로 둔다. `L_predictions.parquet` dft 행과 max |Δyhat| ≤ 1e-6 kcal/mol이면 통과(exact).
  1e-6을 넘으면 GPU float32 비결정성 한계로 1e-4 kcal/mol(= FU2/LADDER gate M-1′ 허용치 ANTISYM_TOL) 이하일 때만
  `within_fp32_bound`로 통과시킨다. 이때 실제 크기를 보이려고 seed 0의 batch 구성 민감도(bs 32 대 256)도 적는다.
  그 밖은 STOP이고 배포 예측 파일을 쓰지 않는다. LADDER 단위는 모두 NVIDIA A10에서 돌았으므로 GPU job은 A10
  파티션(gpu2, gpu6)으로 낸다.
- XGB: CPU라 결정적이므로 exact(≤ 1e-6)만 통과로 본다.
- 단위 지문: unit json fingerprint와 지금 값이 `code_sha256`(train_atoms.py 또는 train_xgb.py 한 파일)과 `gpu` 밖에서
  다르면 STOP. 단위 parquet sha256 = json 기록, test / validation 짝 sha256 = 지금 split.

## DP-3 입력 호환 (보고, 배치가 다르면 STOP)
- token: elements, bin_cols, num_cols, glob_cols, ext_used가 같아야 한다. 버전 문자열은 rung 태그(-dft / -geo)만
  다르다. 반응별 원자 수·원소 수·역할 플래그·결합 수 일치 개수와 수치 열 이동량을 적는다.
- X182 열 이름과 순서가 같아야 한다(XGB S·SH feature 열 포함).
- SH: dft 단위의 Std와 β를 그대로 쓰고 geo H를 넣는다(β 재적합 없음). d1·d2·barrier H의 기준 에너지 규칙이 다르다
  (dft: refs_cr2 own/Q/V, geo: 키별 최저 HF 'min', 제약 키는 자기 기준). 보고만 한다. HF 입력 머리줄(방법,
  FRAG 문자열, 전하·다중도)을 반응마다 비교한다.
- 라벨 y = F_pairs_cr2.csv (geo 라벨 파일의 D1 행은 byte 단위로 같음, d0d1_splits L-11′).

## DP-4 짝 집합
- 공통 짝 = D1 T1 시험 짝 중 dft와 geo 둘 다에서 쓸 수 있는(양 끝 반응 usable) 짝 = (b)가 예측한 짝. (a)와 (c)를
  이 집합으로 줄여서 비교한다. 주 지표는 P_main reference para이고, 모든 위치(all, p, m, o, direct)도 적는다.
- dft에서만 / geo에서만 쓸 수 있는 짝은 개수만 적는다. D0 짝은 쓰지 않는다(D1만).

## DP-5 선택 (시험 점수로 다시 고르지 않음)
- (a)·(b): dft 단위의 내부 검증 MAE(val_kcal_mae)로 고른 REP(LADDER 13-1)와 REP8(GEO2.md 9-1: 두 arm 통합, 동점이면
  SH 먼저, 그다음 AW→GNN→TF→XGB). REP는 LADDER E2 대표와 같아야 한다(다르면 STOP). (b)는 (a)의 선택을 그대로 쓴다.
- (c): GEO2b geo 단위의 검증 기록으로 고른 것(GEO2b 그대로). ENS4 = 4 모델 단순 평균. LINE = SH 기준선 열.

## DP-6 지표 (서술만)
- 경우 × 선택(모델·arm, REP·arm, ENS4·arm, REP8) × 타깃 × 위치: n, MAE, core bootstrap 95 % CI(2,000회, seed
  20261019, d1_eval.boot_diff 규칙).
- 짝지은 차이 d = MAE(b) − MAE(a)(주), MAE(b) − MAE(c)(설명용): core bootstrap CI와 p.
- LADDER 목표 검사(13-2)를 (b)에 적용: oi와 cpcm 모두 para MAE ≤ 0.13, 꺾임 민감도 ≥ 0.7, FPR ≤ 0.1(점추정;
  surely = CI 상한 ≤ 0.13). REP 주, ENS4 보조, REP8과 단일 모델은 같은 표에. (a)·(c)도 같은 집합에서 설명용으로.
- LADDER evaluate.py E1/E2 함수(model_frames, derived_frames, gate_pairs, check_line, annotate_all, metrics_e1,
  bends_e1, goal_e1)와 eval_geo2 / eval_d0d1 함수(reps, select, boot_diff, boot_mae_ci, goal_rep8)를 그대로 부른다.
  이 목표 검사 말고 새 판정 규칙은 없다.
- **DP-run-1 (11-03 12:33)**: compat+XGB(1059491) 통과(공통 짝 7,540, 누수 0, token·HF 설정 같음; XGB 재현 gate exact, 최대 차이 0). 원자 모델 job(1059490)은 A10(gpu2, gpu6)에서 23분 넘게 Priority로 기다렸다. 추천안: 대기 중인 job의 파티션을 GPU 전체(gpu1, gpu2, gpu3, gpu5, gpu6)로 넓힌다(scontrol update, 취소 아님). A10이 아니면 재현 gate는 DP-2대로 'within_fp32_bound'(≤ 1e-4)로 판정한다.
