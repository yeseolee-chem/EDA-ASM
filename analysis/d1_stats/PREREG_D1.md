# PREREG_D1 — Phase D 사전 등록 (SPEC_D1_ALL §Phase D)

- 상태: **초안 — 사용자 승인 전. 승인 전에는 `d1_models.py` 가 어떤 학습도 하지 않는다.**
- 승인 방법: 사용자가 아래 내용을 확인한 뒤 이 줄 바로 아래에 `- 승인: YYYY-MM-DD 이름` 한 줄을 넣고 commit 한다. `d1_models.py` 는 그 줄이 있고, 이 파일이 commit 되어 있고 HEAD 와 같을 때만 돈다. 이 파일의 sha256 은 모든 결과 캐시의 지문에 들어간다(승인 뒤 바꾸면 캐시가 거부된다).
- 작성 2026-10-14, 결과를 보기 전(Phase C · D 미실행, 확정 데이터 미동결).

## 1. 데이터

| 항목 | 정의 |
|---|---|
| D0 학습 | rev5 dev 4,112행 (`results_rev5/dev_ids.csv`), feature = `train_ml_single.load_ml(rows_rev5.csv, ext=True)` 그대로 |
| D0 lockbox | `results_rev5/lockbox_ids.csv` 726행. **M4 에서 한 번만** 연다 (§5) |
| D1 행 | 확정 `labels_d1.json` 중 (a) `q_bad` 아님 (`d1s_prep.py`: `q_badref` ∪ `q_stepwise`, 통계와 같은 제외), (b) 게이트 C 의 `ext_sel_status == ok` (EXT_SEL 133열 전부 유한) |
| feature | EXT_SEL 133열 = `train_ml_single.resolve_arm("EXT_SEL")` (ESPLEY73 + B1 + B4), 순서 고정 |
| 타깃 | `rev5_common.TARGETS9` (ΔE‡ + 8채널), D1 은 Phase C parquet 의 `dft_*` (라벨에서 계산, 라벨 `barrier_kcal` 과 차이의 절댓값 ≤ 1e-6 kcal/mol 로 대조) |
| rev5 위생 필터 | (d1<0, d2<0, d2>50 제외) D1 에는 **적용하지 않는다** (SPEC 의 제외 규칙만). 걸리는 D1 행 수만 보고 |
| D0 와 같은 반응 | 원자 번호를 지운 정준 반응 SMILES (반응물 정렬 + 생성물; 입체 무시판과 입체 포함판). 입체 무시판으로 **D0 lockbox 반응과 같은 D1 행은 M4 학습에서 뺀다** (lockbox 를 미리 보는 것을 막는다). dev 반응과 같은 D1 행은 건수만 보고 (M0 · M1 의 test 에 남는다) |

## 2. 모델

- `KRR_rbf` = `train_ml_single.GRIDS["KRR_rbf"]` · `_make_pipe` (X StandardScaler, y StandardScaler, rbf), grid alpha {1e-5 … 1} × gamma {3e-4 … 1} (48점), 점수 = 평균 MAE, 같은 점수면 grid 순서의 앞.
- 탐색: 바깥 학습 집합 안의 5-fold. M0 = rev5 와 같은 `KFold(5, shuffle=True, random_state=20261001)` (sklearn `GridSearchCV` 그대로).
  M1 · M2 · M4 = `GroupKFold(5, shuffle=True, random_state=20261001)`, 그룹 = D1 행은 코어, D0 행은 반응 하나씩 (짝 · 형제가 한 fold 에 같이 있도록; 무작위 행 분할 금지 규칙).
- 계산 방식: M1 · M2 · M3 · M4 의 탐색은 `d1_models.py` 의 `fast_grid` (fold · gamma 마다 커널 한 번, alpha 마다 Cholesky 한 번, 9 타깃을 한 번에 풀기). sklearn `GridSearchCV` 와 수학적으로 같다. **실행 전 `check` 단계에서 sklearn 과 비교**: 같은 fold 로 두 타깃 이상에서 best_params 가 같고 예측 차이 ≤ 1e-6 이어야 다음 단계로 간다.
- 부록 모델 Ridge 는 이번 등록에 넣지 않는다 (시간이 남으면 같은 분할로 따로 등록).

## 3. 분할 (그룹 단위)

| 이름 | test | train | 비고 |
|---|---|---|---|
| LOCO | D1 코어 하나 (dipole × 골격) | D0 dev + 나머지 D1 코어 | 부모 · 자식 · 형제가 모두 test 쪽 |
| LODO | dipole 하나 | D0 dev + 나머지 4 dipole | |
| LOXO | 치환기 X 하나 (직결 X 와 p/m/o-X-C6H4 를 같은 X 로 묶음) | D0 dev + 나머지 X 의 D1 행 + **H 부모 전부** | H 부모(X = H)는 모든 문맥의 기준이라 test 로 뺄 수 없다 — 부모는 train 쪽에 두고 짝 지표의 부모 예측은 같은 모델의 학습 내 예측을 쓴다 (SPEC 과 다른 점, 보고서에 적는다) |

## 4. 실험과 지표

- **M0**: D0 dev 로 rev5 방식 그대로 학습 → D1 전체 예측. 재현 확인: 9 타깃의 best_params, cv_mae(상대 1e-6), train MAE 가 rev5 lockbox 단위 기록(`$R5_SCRATCH/lockbox/units/EXT_SEL__KRR_rbf__<타깃>.json`)과 같아야 한다 (lockbox 는 쓰지 않는다). 다르면 STOP.
- **M1**: 분할마다 D0 dev + D1 train 으로 탐색 · 학습 → D1 test.
- **M2**: 짝 Δ 모델. 짝 = `d1s_prep.pairs_vs_H` (자식 − H 부모; 두 행 모두 §1 의 D1 행). 입력 [x(자식) − x(부모), x(부모)] (266열), 타깃 = 9 타깃의 ΔΔ. 학습 = train 쪽 짝(부모 · 자식 모두 train). 예측: ŷ₂(자식) = ŷ₁(부모) + Δ̂, 짝 지표의 ΔΔ̂ = Δ̂.
- **M3**: 학습 곡선, LOCO 의 test 코어마다: 나머지 코어를 `default_rng(20261001 + s)` (s = 0…4) 순열로 놓고 앞의 k 개 (k ∈ {0, 5, 10, 20, 40}, 그리고 전체 = M1) + D0 dev 로 학습. 하이퍼파라미터는 그 LOCO 분할의 M1 값으로 고정 (탐색을 1,500번 다시 하지 않는다).
- **M4**: M1 최종 모델 = D0 dev + D1 전체 (§1) 로 탐색 · 학습 → D0 lockbox 726 예측, 한 번.
- 지표 (모두 train · test 함께; carbonyl ylide 포함 / 제외 둘 다 — 제외는 같은 모델에서 test 행만 뺀다):
  - 절대: MAE, NMAE (`rev5_common.nmae`), 타깃 · dipole 별
  - 짝: ΔΔ̂ = ŷ(자식) − ŷ(부모) vs ΔΔ 의 MAE, 부호 일치율, 문맥(`ctx` = 코어 | 방향 | 면 | 입체) 안 Spearman ρ 의 중앙값 (짝 ≥ 5 인 문맥만)
  - 선택성 (ΔE‡ 만): regio 형제 (`cell_id, face, stereo_index` 안 방향 a vs b) · exo/endo 형제 (`cell_id, orient` 안) 에서 이긴 TS(낮은 장벽)를 맞춘 비율. 형제 = `d1s_prep.sibling_pairs` (robustness.py 와 같은 정의)
  - train 지표 = 같은 모델의 학습 행 · 학습 짝에 대한 학습 내 예측
  - M2 의 절대 지표는 M2 가 예측하는 행(H 부모가 있는 자식 + 부모)에서만 정의된다. 같은 행에서 잰 M1 을 'M1 (M2 rows)' 로 함께 낸다 (M1 · M2 비교는 이 둘로).

## 5. 판정 기준 (숫자로 고정)

1. **M2 채택**: 주 판정 = LOCO, ΔE‡. test 짝을 코어 단위로 1,000번 다시 뽑아 (seed 20261014) 짝 MAE(M2) − 짝 MAE(M1) 의 95 % 백분위 CI 의 상한이 0 보다 작으면 "M2 채택". 나머지 (분할 × 타깃) 는 같은 방식으로 보고만 한다.
2. **M4 악화**: lockbox 726 에서 타깃마다 MAE(M4) − MAE(rev5 EXT_SEL KRR_rbf) 를 반응 단위 짝 부트스트랩 10,000번 (`default_rng(20261001)`, rev5 와 같은 방식, rev5 예측 = `$R5_SCRATCH/lockbox/lockbox_predictions.parquet`) 으로 95 % CI. **9 타깃 중 하나라도 CI 하한 > 0 이면 보고하고 멈춘다.** 기준값(rev5 EXT_SEL lockbox MAE): ΔE‡ 1.732, d1 1.607, d2 1.221, elst 3.776, Pauli 6.663, OI 3.988, disp 0.471, CPCM 1.610, CDS 0.211.
3. M0 의 짝 지표가 낮게 나와도 그대로 보고한다 (SPEC 주의 문단).

## 6. lockbox 규칙

- M4 만 lockbox 를 읽는다. 순서: (1) 학습 (lockbox 를 쓰지 않음) → lockbox **feature** 로 예측을 캐시 → (2) `LOCKBOX_OPENED.json` 을 쓰고 나서 lockbox 타깃을 읽어 지표를 낸다.
- (2) 가 끝난 결과가 있으면 M4 는 다시 돌지 않는다. (2) 도중 잡이 죽은 경우에만, 같은 예측 캐시(지문 일치)로 (2) 를 마저 한다 — 새 학습 · 새 평가가 아니다.
- M0 · M1 · M2 · M3 는 lockbox 행을 메모리에서 바로 버린다.
