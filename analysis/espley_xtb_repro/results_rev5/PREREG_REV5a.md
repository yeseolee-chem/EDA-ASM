# PREREG_REV5a — rev 5 사전 등록 (블록 선별 전, 2026-10-01)

명세: `docs/specs/REV5_FEATURES_FIGURES.md` (Phase C-1). 이 문서는 **블록 선별(Phase C-2)과 모든 학습 전에** commit 한다.
commit 이후 아래 규칙은 바꾸지 않는다. 바꿔야 하면 바꾸지 않고, 결과 문서에 "사전 등록과 다름"으로 적는다.
상수의 원본은 `rev5_common.py`(이 commit에 포함)이고, 아래 값은 그 파일과 같다.

이 시점까지 본 것은 학습과 무관한 입력 점검뿐이다: B-0 엔진 smoke(`B0_smoke.json`), B-7 feature 계산·병합 gate
(`B7_report.json`), Phase A 재생성 검사(`A_*.json`). rev 5 모델(확장 블록을 쓰는 모델)은 어느 것도 학습하지 않았다.
두 가지는 lockbox를 만들기 전에, 나중에 lockbox가 된 반응을 포함한 행에서 계산됐다. 둘 다 블록 선별에 쓰이지 않는다:
(i) Phase A가 다시 만든 rev 4 분석(헤드라인 표, Espley 비교, 기하 절제 arm A/C/C0/B_g1/O_g1 — 확장 블록 없음),
(ii) B-7의 feature 분포 요약(`B7_feature_quantiles.csv`, 서술용).

## 1. 입력

| 항목 | 값 |
|---|---|
| 라벨 | repo 루트 `labels_all.json` (sha256 `62c0e7845045d0810627e88f1f52a3f353059bd78ac12ed89f8cd0f053a08753`) |
| G1 feature (rev 4) | `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features_g1.parquet` (sha256 `64ee5da7dcc7e3b0b2c055fe5030135f255e8d30e487cc6d1ea456d64af4b1d8`) |
| 확장 feature B1–B6 | `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features_ext_g1.parquet` (sha256 `35d1a84ff5667965450d8f1d24c1aa42209cbc63bd4838c27de98f18919b0370`), 109열 (`rev5_common.BLOCKS`; B6 생성물 feature 3개는 사용자 결정으로 제외, §6) |
| 엔진 | xtb 6.7.1 (edcfbbe) — xtb가 출력하는 모든 값; tblite 0.7.0 — B1 궤도 overlap만 |
| 행 | `rows_rev5.csv` = `rows_rev4.csv` ∩ 확장 feature ok, **n = 4,838** (sha256 `a89a3d923d6f3daa81e6d17b1d2640cadabd087736bfb10dbaef06edaf70c8f8`) |

## 2. lockbox와 dev (C-1)

- lockbox = `sorted(np.random.default_rng(20261001).choice(rows_rev5 rxn_id 오름차순, size=round(0.15 n), replace=False))`
  → `lockbox_ids.csv` (**n = 726**, sha256 `97c055149e4c680d0522bf48ad16af4430ebffa81863611902e05da88a797f86`). dev = 나머지 → `dev_ids.csv` (**n = 4,112**, sha256 `19a8e2123c6d173d3b1975688deb3fb0c4d15cd8de33fdc7ce822385ae5a1117`).
- lockbox 행은 Phase D 전까지 어떤 학습·튜닝·선별에도 쓰지 않는다. D의 모든 스크립트는 lockbox를 처음 쓰기 전에,
  선별 캐시(`$R5_SCRATCH/select/`)에 lockbox id가 하나도 없고 dev id가 모두 있는지 검사한다.

## 3. 블록 선별 (C-2, dev 행만)

- **모델:** KRR(RBF) 하나. 파이프라인은 rev 4 Protocol A와 같다: `_make_pipe` = X StandardScaler + y 표준화
  (`TransformedTargetRegressor`), grid = `train_ml_single.GRIDS["KRR_rbf"]`.
- **CV:** dev 행(rxn_id 오름차순)에 바깥 `KFold(5, shuffle=True, random_state=20261001)`. 바깥 fold마다 train 안에서
  `GridSearchCV(cv=KFold(5, shuffle=True, random_state=20261001), scoring=neg MAE)`로 튜닝(nested).
- **지표:** fold test 행의 MAE, NMAE (= MAE / 그 fold test 타깃의 평균절대편차), r².
- **후보 arm:** `BASE` = ESPLEY73; `BASE+Bk` (k = B1…B6); `EXT_ALL` = BASE + B1…B6. 이 8개는 9 타깃 모두에 대해 돌려
  보고한다(C-2 표).
- **선별 기준:** elst, Pauli, OI의 fold NMAE 평균을 다시 5 fold로 평균한 값.
- **전진 선택:** 현재 arm(처음 BASE)에 남은 블록을 하나씩 더한 후보 가운데 기준이 가장 낮은 것을 고른다.
  채택 조건: 현재 대비 **상대 감소 ≥ 2 %** 이고 **바깥 5 fold 중 ≥ 4 fold에서 fold 기준이 낮을 것.** 못 맞추면 멈춘다.
  결과가 `EXT_SEL`이다. 채택 블록이 없으면 `EXT_SEL = BASE`(STOP 아님).
- **사전 고정 arm:** `EXT_ALL`은 선별 결과와 상관없이 최종 평가에 들어간다.
- 결과는 `prereg_rev5b.json`과 `PREREG_REV5b.md`에 적고 commit 한 뒤에만 Phase D를 시작한다(C-3).

## 4. 최종 평가 (Phase D) — 지금 고정하는 세부

- **D-1 lockbox (1차, 헤드라인):** dev 전체로 학습, `GridSearchCV(cv=KFold(5, shuffle=True, random_state=20261001))`
  nested 튜닝, lockbox로 채점. arm BASE / EXT_SEL / EXT_ALL, 모델 KRR(헤드라인) · Ridge · SVR · XGB(부록), 9 타깃.
  MAE·NMAE·r², lockbox 행 단위 bootstrap 10,000회(rng 20261001; 모든 arm·모델·타깃이 같은 재표본 인덱스를 공유)
  95 % percentile CI, 짝지은 차이 EXT_SEL − BASE와 EXT_ALL − BASE의 bootstrap CI.
- **D-2 Protocol A (rev 4 연속성):** `rows_rev5` 전체, 80/10/10, seed 22/23/14/1/2, nested, arm ESPLEY46 / ESPLEY73 /
  EXT_SEL / EXT_ALL × 4 모델 × 9 타깃. 표에 "EXT_SEL은 dev 행에서 골랐으므로 D-2 test 행과 겹친다; 편향 없는 값은 D-1"을 단다.
- **D-3(a) Espley 분할:** 행 = Espley ds3 ∩ 라벨 ∩ G1 사용 가능 ∩ `rows_rev5`; Espley 분할(test = 20 % hold-out의 첫 절반),
  seed 22/23/14/1/2; 모든 쪽을 같은 test 행으로 채점. 우리: ESPLEY46 / ESPLEY73 / EXT_SEL, KRR, seed마다 nested.
  Espley: 저장된 SVR 예측(Interaction, ΔE‡, ΔG‡), 역할 d1/d2는 그들의 프로토콜 SVR 재학습, Espley 46 feature × 우리
  파이프라인 XGB(모든 5 타깃; 그들의 행으로 학습). 검정: seed별 짝지은 MAE 차이(Espley − 우리)에 Nadeau–Bengio 보정
  t(J = 5, n_test/n_train = 실제 평균), 양측 p와 95 % CI. 비교 대상 (i) Espley SVR, (ii) 타깃마다 Espley 쪽에서 평균 MAE가
  가장 낮은 모델(저장 SVR/KRR/Ridge/2·4-layer NN, 그들의 프로토콜 SVR/KRR, 우리 파이프라인 Ridge/KRR/SVR/XGB).
- **D-3(b) lockbox 정면 비교 (EXT_SEL의 1차 Espley 비교):** test = lockbox ∩ Espley 행, train = dev ∩ Espley 행.
  Espley: 그들의 프로토콜 — ESI Table S3 grid(`compare_espley.THEIR_TUNE`), 이 train에서 `GridSearchCV(cv=5)`(그들 코드처럼
  shuffle 없음), X만 StandardScaler, 튜닝 열 47(그들의 hyp_tuning.py), 적합 열 46, SVR과 KRR(KRR은 poly로 튜닝하고 rbf로
  돌린다 — 그들 코드 그대로). 역할 타깃에는 AM1 변형 feature를 같은 swap 벡터로 재배열. 우리: EXT_SEL KRR,
  `GridSearchCV(cv=KFold(5, shuffle=True, random_state=20261001))`. 반응 단위 짝지은 bootstrap 10,000회(rng 20261001)
  MAE 차이 95 % CI. **주 비교는 Espley SVR(그들이 발표한 최고 모델) 대 EXT_SEL KRR**, Espley KRR은 부록.
- **D-4 학습 곡선:** Espley 분할·seed; 부분집합 = seed별 train 행을 rxn_id로 인덱싱하고 **그들의 `ml_analysis.py`
  X_train 순서(분할 후 순서)** 로 둔 DataFrame에서 `sample(frac=f, random_state=seed)`, f = 0.1 … 1.0; ds3 행 순서로 뽑은
  결과는 민감도 분석으로 따로 적는다. 두 쪽이 같은 부분집합; 하이퍼파라미터는 전체 train 튜닝값으로 고정(Espley SVR:
  Interaction/ΔE‡/ΔG‡는 `hps.pkl`, 역할 타깃은 그들의 프로토콜 재튜닝 값; EXT_SEL KRR: D-3(a)의 seed별 튜닝값); test 행 고정.

## 5. 그림 (Phase E) — 지금 고정하는 선택

- fig1 오차 막대 = 5 seed MAE의 SE(sd/√5); Espley 정의 SE(seed 평균 std(|오차|)/√n)는 CSV에 함께 저장.
- fig2 = Espley SVR 대 EXT_SEL KRR(D-3(b) 주 비교). fig6 기준선 = D-4에서 frac 1.0일 때 Espley SVR의 MAE.
- 모든 그림은 양단 폭(183 mm). 색은 `rev5_common.COLORS`.

## 6. 이미 알려진 사양과의 차이

- B-0 smoke 대상 5건 가운데 rxn 20은 G1 기하가 없어(rev 4 Phase 1: OptTS 미수렴) 평가할 수 없었다. 나머지 4건과
  하전 반응 2건(3312, 3323)을 추가로 평가했고, 세 gate 모두 통과했다.
- 코드 작성 에이전트 4개가 로그인 노드에서 빈 입력의 `python3`를 한 번씩 실행했다고 스스로 보고했다(계산 없음).
- **B6 생성물 feature 제외 (사용자 결정 2026-10-01).** 첫 B-7 실행(잡 997381/997382, 코드 r5-ext-2)이 STOP 했다:
  rows_rev4 실패 122 / 4,839 = 2.52 % > 1 %. 121건이 B6 생성물 단계(생성물 그래프 ≠ G1 TS + 형성 결합 66, xtb 최적화 뒤
  생성물 그래프 변화 54, 최적화 미수렴 1)였다. 사용자 결정으로 `xtb_dErxn`, `prog_ad`, `prog_be`를 빼고 B6는
  `nu_imag`, `mode_share`, `mode_async`만 둔다(109열). 재계산(잡 997666/997667, r5-ext-3)은 B-7 gate를 모두 통과했다
  (실패 1 / 4,839 = 0.02 %, B1 엔진 gate 1.9e-6 Eh).
