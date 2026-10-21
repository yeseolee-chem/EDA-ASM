# PREREG ADD2 — D1PAIR 사전 등록 추가본 (FU2 원자 토큰 모델, 2026-10-21)

이 문서는 `PREREG_D1PAIR.md`(commit 10f99d8d)와 `PREREG_D1PAIR_ADD1.md`(commit 1de89a0f)에 더하는 추가본이다. 둘은 바꾸지 않는다.
근거: 명령서 `docs/specs/D1PAIR_FU2.md`, 사용자 코드 검토 `results/fu2/REVIEW_FU2_CODE_d1e4e3d8.md` 1–9절(9절: commit 35313e3a 승인;
이 문서와 같은 commit에 들어가며 sha256은 json `review_sha256`), 검토 기록 `results/fu2/C_review.md`.
아래 1–9는 검토 문서 9절 "ADD2에 꼭 들어갈 것" 1–9를 그대로 따른다.

Phase M 학습은 이 문서와 `prereg_d1pair_add2.json`이 commit된 뒤에만 한다. `fu2/train.py`의 `gate_add2`가 매 학습 전에 확인한다.
- 두 파일이 commit되어 있고 그 뒤 바뀌지 않았는지
- md sha256이 json `md_sha256`과 같은지
- json `inputs_sha256`의 **11개 파일**(코드 5 + 입력 6)이 지금 파일과 같은지

하나라도 다르면 exit 3이다.

**지금까지 본 것:** 이 문서를 쓰는 시점에 Phase M(학습)은 한 번도 돌지 않았다(`$D1PAIR_SCRATCH/fu2/units` 없음).
- Phase C 검사가 시험 짝 예측을 만들기는 했다. smoke(2 epoch)는 `$D1PAIR_SCRATCH/fu2/smoke`(Phase M이 읽지 않는 scratch)에, C14(1 epoch)는 임시 폴더에 썼다. 그러나 원자 모델의 시험 짝 점수(MAE 등)는 계산하지도 보지도 않았다. 검사는 예측끼리 또는 정답 정렬만 비교했다.
- 비교 상대 C-XGB의 T1 시험 결과는 잠정 결과(`results/interim_T1_noSVR/`, commit 5df20b68)로 이미 보았다.

## 1. 입력 고정

**코드:** commit `35313e3ae6282549a1e861933224cac39f802eda`(35313e3a, 사용자 승인). 아래 5개 파일은 json `inputs_sha256`에도 들어 있어 `gate_add2`가 매 학습 전에 확인한다.
각 unit은 실제로 돌린 코드의 sha256을 지문(`code_sha256`, import 때 한 번 계산)에 남긴다. `train.py collect`는 모든 unit 지문을 collect 시점의 코드·입력과 비교해 다르면 STOP한다.
Phase E는 모든 unit의 `code_sha256`이 아래 값과 같고 지문의 `add2`가 이 md의 sha256과 같은지 확인한다. 아니면 STOP이다.

| 파일 (`analysis/d1_pair/`) | sha256 |
|---|---|
| `fu2/models.py` | `de2065a7d39c1e008b31274ab1b4fc94fa123ce53eda14a3c2f426dbd0d6197a` |
| `fu2/train.py` | `2f1ab7a06bfec20901cfd9130a9784961070e6477728ade92fed9a1cce729469` |
| `fu2/fu2_common.py` | `66dcf72f5b23a015e1750e4b2ac458dd61a4722050a7be24a03b8269a628dfe2` |
| `fu2/tokens.py` | `f37207bbf987764cfbadd064ad4f9ffbdb33e54daa7563825b3966d0c5e95514` |
| `d1pair_common.py` | `65a6e2da81e4dedf20c51423cbfe9c97e8b8f2272b24454a739a09e8552aeedc` |

참고로 Q·T·C를 만든 `fu2/atoms.py`, `fu2/splits.py`, `fu2/tests/run_tests.py`, `fu2/tests/diag_c12.py`의 sha256도 json `aux_code_sha256`에 적는다. 이 값은 gate가 확인하지 않는다.

**입력 결과 파일:** `gate_add2`가 매 학습 전에 확인한다.

| 파일 | sha256 |
|---|---|
| `results/fu2/Q_report.json` | `12e6c5c08f17575ddb93abe698560856bd0d2d1587a4260fd7c8d98e45f88307` |
| `results/fu2/Q_manifest.csv` | `646e4f29a30f189b44e548fa9c8faa1c3f5dcf52492de84cf8143ce5324076a1` |
| `results/fu2/T_report.json` | `673ac6c0efc77b52d18f562a821fa19ed9180ad4bf0f1a1fb6ff011cc0d8a871` |
| `results/fu2/T_manifest.csv` | `15df2a7c4eba21e7bb8778dc3fc8772686693138394dece6c88730208c668cc4` |
| `results/fu2/S_splits.json` | `eefe495a775babf2604be6f21156e4ed2869a1988bd08d1542c0d7d4b7779413` |
| `$D1PAIR_SCRATCH/fu2/splits/pairs.parquet` | `265468b5d2a83c79fe83fa8c269830a1f56cd7c6f771adcb673deabe41ab28fe` |

토큰 파일(반응별 npz 8,336개)의 sha256은 `T_manifest.csv`에 있고, `train.Data`가 읽을 때마다 확인한다.
비교 상대 C-XGB의 예측 파일(`$D1PAIR_SCRATCH/M/units/T1/T1__C__XGB__<타깃>__f<k>.parquet`, 50개)의 sha256은 json `comparator_sha256`에 있다. Phase E가 읽기 전에 이 값을 확인한다.

**버전:** atom `fu2-atoms-3`, token `fu2-tokens-3`, 학습 코드 `d1pair-fu2-1`.
- 환경 `env_d1fu2`: python 3.11, torch 2.2.1+cu118, numpy 1.26.0, pandas 2.1.1, scikit-learn 1.3.1(`results/fu2/env.yaml`).
- 원자 대응 상수: 동점 허용 1e-6 Å, automorphism 한도 10,000(관측 최대 12).

**데이터:**
- 반응 8,336개. Q의 8,337개 중 1007141 하나를 뺐다. 이 반응은 확장 값의 자기 검사(TS json 쌍극자 불일치)에서 실패했다.
- 짝 8,389개 중 원자 데이터가 있는 짝은 8,092개다.
- 원자 데이터 없는 짝을 빼기 전 T1 fold 0–4의 학습 / 시험 짝 수:

  | fold | 학습 | 시험 |
  |---|---|---|
  | 0 | 6,272 | 1,714 |
  | 1 | 6,348 | 1,670 |
  | 2 | 6,525 | 1,446 |
  | 3 | 6,614 | 1,357 |
  | 4 | 6,284 | 1,688 |

## 2. 모델·학습 규칙 (명령서 M-1–M-3 그대로)

**구성:** 모델 3개(AW 원자 단위 가산, GNN, TF) × 변형 2개("A" = 원자 토큰만, "A+G" = 원자 토큰 + 전역 토큰 182열) = 6개.
- grid(모델마다 4칸):
  - AW: L ∈ {1, 2} × 폭 ∈ {64, 128}
  - GNN: L ∈ {4, 6} × 폭 ∈ {64, 128}
  - TF: L ∈ {2, 3} × 폭 ∈ {64, 128}, 4 head
- 구조(명령서 M-2 표, `fu2/models.py`):
  - AW: 5 Å 연속 필터, RBF 32개, cosine cutoff, E = Σ ε_i (+ φ(전역)).
  - GNN: 간선 종류 + 거리 RBF 32개(0–4 Å), residual + LayerNorm, [합, attention pooling].
  - TF: head마다 거리 RBF 32개(0–10 Å)·같은 조각·형성 결합 bias, 가중치를 공유하는 cross-attention, [평균, attention pooling].
- 반대칭: AW는 g = E(a) − E(b), GNN·TF는 g = ½[h(a,b) − h(b,a)].

**입력:**
- 원자 토큰 = 원소 one-hot 9개 + 표시 6개(frag_B, dip_a, dip_mid, dip_b, dph_a, dph_b) + 숫자 14열(bond_dist, q_rel, q_frag, q_TS, dq_dist, dq_int, v_rel, v_frag, v_TS, dv_dist, dv_int, mu_abs, f_plus, f_minus).
- TS 좌표(원자 사이 거리만 씀), 공유결합·형성 결합 간선.
- σ, 계열·core 번호, X 이름은 넣지 않는다.

**짝·분할:**
- `d1_train.masks(u)`의 arm C, T1 바깥 fold 0–4.
- 학습 = 학습 쪽 T_main(기준 짝 + anchor 짝), 시험 = 시험 쪽 P_sens ∪ A_main.
- a·b 원자 데이터가 없는 짝은 뺀다. fold 0 시험에서 1짝, fold 1–4 학습에서 1짝씩 빠지며, 모두 같은 짝이다.

**표준화:**
- 학습마다 그 학습 짝에 나오는 반응만으로 숫자 열과 전역 토큰의 평균·표준편차를 구한다.
- 표준편차가 1e-12보다 작은 열은 표준편차를 1.0으로 두어 열을 남기고 평균만 뺀다. rev 9는 이런 열을 버렸다.
- y는 그 학습 짝의 표준편차(ddof 0)로 나누기만 한다.

**손실:** 타깃 10개의 |ŷ − y| 평균(표준화) + **λ = 0.1** × 평균 |ŷ_barrier·s_barrier − Σ_9 ŷ_c·s_c| / s_barrier.

**학습:**
- AdamW(lr 1e-3, weight decay 1e-4), cosine 감소(T_max 300 epoch), dropout 0.1, batch 64짝.
- 최대 300 epoch, early stopping patience 30.
- 매 epoch 짝마다 (a,b)/(b,a) 순서를 무작위로 바꾸고, 정답 부호도 함께 바꾼다.

**튜닝:**
- 바깥 학습 짝 안의 core별 GroupKFold(4) **첫 안쪽 fold**를 검증으로 쓴다(rev 9 NN 규칙; `S_splits.json`에 고정).
- grid 칸마다 검증 MAE(표준화, 9 주 타깃 평균)가 가장 낮은 칸과 그 best epoch를 고른다. 동점이면 grid 순서를 따른다.

**최종:**
- 고른 칸으로 바깥 학습 짝 전부를 best epoch만큼 학습한다(같은 cosine 일정).
- **seed 5개**(20261019 + 0..4). 예측은 5개 평균이고, seed 표준편차도 저장한다.

**단위:**
- 구성 6 × fold 5 = 30 unit, unit당 학습 9번(튜닝 4 + 최종 5).
- 예측은 `results/fu2/M_predictions.parquet`(`train.py collect`)에 모은다. 6개 구성의 결과를 모두 보고한다.

**GPU:**
- **RTX A6000만** 쓴다(`-p gpu4,gpu5 --gres=gpu:a6000:1`). 제출 옵션으로 정한다.
- GPU 이름은 지문에 들어간다. `run_unit`은 다른 GPU로 만든 캐시를 만나면 STOP한다. `collect`는 GPU 이름을 비교하지 않는다.
- 그래서 Phase E가 모든 unit 지문의 GPU 이름을 확인한다. RTX A6000(A6000 Ada 아님) 하나로 같아야 하고, 아니면 STOP이다.
- GPU 비결정성은 허용하고 기록한다(seed 고정, `cudnn.deterministic = True`, `benchmark = False`).

## 3. 대표 구성 (주 비교에 쓰는 원자 모델)

타깃마다(10개; 주 비교는 9개) 다음 순서로 정한다.
1. 구성 6개 각각, 바깥 fold k마다 **고른 칸의 안쪽 검증 MAE(그 타깃, best epoch)**를 kcal/mol로 바꾼다.
   - 곱하는 s_y = 그 검증 MAE를 표준화할 때 쓴 값이다.
   - 곧 그 fold 튜닝 학습 짝(첫 안쪽 fold의 학습 쪽, 원자 데이터 있는 짝)에서 그 타깃의 표준편차(ddof 0)이며, split 파일로 다시 계산한다.
2. 바깥 fold 5개 평균이 가장 낮은 구성이 그 타깃의 대표다.

- 동점 = float64 평균이 정확히 같은 경우다. 이때 AW-A → AW-A+G → GNN-A → GNN-A+G → TF-A → TF-A+G 순서로 안정 정렬해 앞의 것을 쓴다.
- **시험 점수는 보지 않는다.** rev 9 `d1_eval.representatives`와 같은 방식이다.

## 4. 주 비교 (확증)

- **대상:** 타깃 9개(8채널 d1, d2, elst, pauli, oi, disp, cpcm, cds + barrier). barrier는 모델이 직접 낸 값이며 9개의 합이 아니다.
- **통계량:** 타깃마다 **d = MAE(C-XGB) − MAE(원자 대표)**.
  - C-XGB = 기존 arm C의 XGB(원래 rev 9 grid). 예측은 `$D1PAIR_SCRATCH/M/units/T1/T1__C__XGB__*.parquet`(1절 sha256)에서 읽는다.
- **시험 짝:** T1 fold 0–4의 시험 짝 중 **P_main**을 다섯 fold 합쳐서 쓴다. 원자 데이터가 없는 짝은 두 모델 모두에서 똑같이 뺀다.
- **신뢰구간과 p:** `d1_eval.boot_diff`와 같은 함수로 계산한다(x = C-XGB, b = 원자 대표).
  - 그 짝들의 core를 복원 추출하는 cluster bootstrap 2,000회(seed 20261019)
  - MAE는 합친 짝 전체에서 계산
  - **95 % percentile CI**(2.5 / 97.5)
  - 양측 p = min(1, 2 min((1 + #{d* ≤ 0})/(B + 1), (1 + #{d* ≥ 0})/(B + 1)))
- **다중 비교:** **Holm 9개**(`rev6_common.holm` = `d1pair_common.holm`).
- **판정:** Holm p < 0.05이고 CI 하한 > 0인 타깃을 센다.
  - **5개 이상**: "원자 입력이 182개 feature보다 낫다"
  - **2–4개**: "일부 채널에서만"
  - **1개 이하**: "이득 없음"

## 5. 부 비교 (보정 없음)

모두 같은 분할 안에서는 같은 시험 짝을 쓰고(원자 데이터 없는 짝은 똑같이 뺌), bootstrap도 주 비교와 같으며, 보정은 하지 않는다.
- 모델 사이(AW 대 GNN 대 TF, 같은 변형)
- 변형 사이: "A+G" 대 "A"(원자 정보 위에 182개가 더하는 몫), "A+G" 대 C-XGB(182개 위에 원자가 더하는 몫)
- para 짝만
- C 대표(rev 9 대표 규칙): SVR이 끝나기 전에는 SVR을 뺀 5개 중, 끝나면 6개 중에서 다시 고른다. 어느 쪽인지는 SUMMARY에 적는다.
- XGBW(FU1)
- null(ΔΔ = 0)
- LODO·LOSO(6절)

설명용으로 함께 낼 지표는 MAE(전체·para·위치별·anchor)와 7절의 Hammett 수준이다.
해석 분석은 탐색이다: 원자별 기여 지도, TF cross-attention 지도, 작용기 원자의 Δq·Δv를 0으로 바꾸는 검사. 그림 설명에 "attention·기여는 설명이 아니라 진단"이라고 적는다.

## 6. M-4 (LODO·LOSO, T1이 끝난 뒤)

- **모델마다 대표 변형 하나:** 그 모델의 두 변형 중 **9 타깃 표준화 평균 검증 MAE(칸을 고를 때 쓴 값)의 T1 5 fold 평균**이 낮은 변형. 동점이면 A를 먼저 고른다.
- LODO(5 fold)·LOSO(10 fold)에서도 같은 규칙(2절)으로 칸을 다시 튜닝한다. 검증은 그 분할의 첫 안쪽 fold다(`S_splits.json`).
- 그 분할의 원자 대표는 타깃마다 3개 구성 중 3절 규칙(그 분할의 안쪽 검증 MAE)으로 정한다. 구성 3개의 결과도 모두 보고한다.
- 비교 상대는 같은 분할의 C-XGB다. 결과는 부 비교다.
- M-4 실행 코드는 실행 전에 따로 검토받는다.

## 7. Hammett 수준 (부)

rev 9 부록 D 규칙(`hammett_eval.calls`, `d1_eval`과 같은 함수)을 따른다: R²_fam, ρ 오차, 꺾임·V 판정, 숨은 V 6계열, 숨은 꺾임 30계열.
채널별 목표 MAE에 대한 배수도 내지만, 이 목표는 **사후 정의**임을 적는다(꺾임 크기 중앙값 × 0.28, FU1 L-3 표).

| 타깃 | barrier | d1 | d2 | elst | pauli | oi | cpcm | cds | disp |
|---|---|---|---|---|---|---|---|---|---|
| 목표 MAE (kcal/mol) | 0.16 | 0.19 | 0.04 | 0.21 | 0.41 | 0.29 | 0.22 | 0.01 | 0.05 |

## 8. 계획과 다르게 한 점

1. **C12 판정 제외:** AW 가장 작은 칸(L1·폭 64)만 64짝 학습 가능성 기준(< 0.05)을 넘지 못했다.
   - `C_tests.json` C12(fu2-tokens-3, job 1030202) 결과: 0.064. 같은 64짝에서 폭 128은 0.0075, 1,000 epoch(T_max 1,000)는 0.0007이다.
   - 앞서 3개 표본으로 돌린 진단(`C12_diag.json`, fu2-tokens-2)도 같은 결론이었다.
   - 결론: 용량·학습량 문제이고 버그가 아니다. 사용자 결정(2026-10-21): "크기를 키우면 없어지니 무시".
   - grid와 학습 규칙은 바꾸지 않았다. `C_tests.json`의 `all_pass: false`는 이 C12 하나 때문이다.
2. **원자 대응:** 명령서 Q-1의 `inv0`/`inv1`은 heavy 원자만 담는다. 여기에 세 가지를 더했다.
   - (가) H 원자 규칙: 자기 heavy 원자를 거쳐 대응한다. 값은 그 heavy 원자에 붙은 반응물 H들의 평균이고, 간선은 같은 heavy 원자 안에서 일대일로 옮긴다.
   - (나) 대칭 보정(사용자 검토 F1): 반응 원자 5개를 고정한 반응물 heavy 그래프 automorphism 중 Kabsch RMSD가 가장 작은 것을 쓴다. 동점이면 원래 VF2 대응을 유지한다.
   - (다) **반사 허용(O(3))**(사용자 결정 2026-10-21): 후보마다 min(고유 회전, 비고유 회전) RMSD로 비교한다.
   - xTB 재계산 없이 저장 값에서 다시 만들었다(fu2-atoms-3). Q-a 0, Q-c(원소 불일치·반응 원자 재대응·자기 검사) 모두 0.
3. **반응 1007141 제외(확장 값 자기 검사 실패):** T1 fold 0 시험에서 1짝이 빠진다. 같은 짝이 fold 1–4에서는 학습 쪽 1짝이다.
4. **최종 학습 step 약 1.33배:** 최종 학습은 튜닝에서 고른 epoch 수를 그대로 쓰지만, 짝이 약 1.33배라 optimizer step도 약 1.33배가 된다. 명령서 규칙 그대로이며 고치지 않는다.

명령서가 정하지 않아 정한 값(모두 Phase C 실행 전에 정했다; `C_review.md` 1-1):
- C12 판정 실행의 설정: dropout 0, batch 8.
- C9 기준의 읽기: "손실 안의 벌점 항(λ 포함) ≤ 1e-3".
  - 정답 y에서 λ 포함 값은 2.0e-4로 통과한다. λ 없이 읽으면 2.0e-3으로 1e-3을 넘는다.
  - 원인은 라벨 자체의 항등식 잔차(최대 0.096 kcal/mol)다.
- GNN 거리 RBF 0–4 Å, TF 0–10 Å, pooling 모양.
- 표준편차 < 1e-12 열의 처리(2절).

## 9. 멈춤 규칙

**Phase M:** 다음이면 STOP(exit 3)이다.
- **gate M-1′ 실패:** 학습한 모델마다 무작위 시험 짝 200개(입력만 사용)를 float64로 다시 계산해, |g(a,b) + g(b,a)| 또는 |g(a,a)| > 1e-4 kcal/mol
- **손실·검증 MAE가 NaN/inf, 또는 best_epoch < 1**
- **캐시 지문 불일치:** 다른 코드·입력·GPU로 만든 캐시를 만남
- ADD2 gate 실패(11개 sha256 포함)
- CUDA 장치가 없거나 env_d1fu2·torch 고정판이 아님
- 튜닝에 시험 짝을 씀. C8이 같은 코드 경로에서 확인했다.

**Phase E:** 다음이면 STOP이다.
- 비교하는 모델 사이 시험 짝 목록이 다름(원자 대표, C-XGB, 부 비교 상대 모두; 원자 데이터 없는 짝을 똑같이 뺀 뒤)
- unit 지문의 `code_sha256`·`add2`·GPU 이름이 1·2절과 다름
- C-XGB 예측 파일의 sha256이 1절과 다름

**원칙:**
- 시험 결과를 본 뒤 grid·학습 규칙·대표 규칙·판정 규칙을 바꾸지 않는다. 바꾸면 이탈로 기록한다.
- 다음 코드는 실행 전에 따로 검토받는다: Phase E 평가 코드(C-XGB와 같은 시험 짝, bootstrap, Holm, 꺾임 판정), 대표 구성 고르기 코드, M-4 실행 코드.

## 알려진 점

**반사 허용 뒤에도 heavy RMSD가 1 Å을 넘는 조각:** **A 103개**(그중 대칭 원자 없음 49), **B 510개**(302개).
- 수와 RMSD의 출처: `Q_manifest.csv` `qc_rmsd_A_v2` / `qc_rmsd_B_v2` / `qc_n_auto_*`.
- 원인은 **확인하지 않은 추정**이다. 판단 근거는 SMILES(`$D1PAIR_SCRATCH/G/smiles_d1.csv`)와 `pairs_d1.csv`의 치환기·위치 열이다. 유연한 큰 치환기의 회전 이성질체가 완화된 반응물과 TS 조각에서 다를 것으로 본다.
- 예시 10개:

  | rxn_id | 조각 | RMSD (Å) | 내용 |
  |---|---|---|---|
  | 1006793 | B | 2.92 | 토실-아릴 sulfone이 달린 cyclooctyne |
  | 1006274 | B | 2.53 | 1006793과 같은 dipolarophile, dipole은 다름 |
  | 1010591 | A | 2.51 | o-토실-아릴 nitrone |
  | 1010590 | A | 2.49 | 1010591과 같은 dipole |
  | 1009491 | B | 2.50 | o-토실-아릴 norbornene |
  | 1008902 | B | 1.74 | o-벤질페닐 norbornene |
  | 1007736 | B | 1.33 | m-포밀페닐 norbornene |
  | 1010866 | B | 1.22 | o-포밀페닐 cyclooctyne, 고리 주름 포함 추정 |
  | 1006029 | B | 1.13 | m-CF₃-페닐 norbornene |
  | 1006157 | B | 1.01 | m-아세틸페닐아세틸렌 |

- 이 경우 Δq_dist에는 이형태 변화도 섞인다. 다만 라벨 d1·d2도 같은 기준(완화된 반응물)이라 정의는 일관된다.
- Phase E 해석 그림에서 이 반응들을 표시한다.

**기타:**
- Phase M 시간 상한(조기 종료 없음 가정)은 smoke 기준 약 132 A10 GPU-시간이다. A6000에서는 더 짧을 것으로 예상한다.
- 짝 차이의 항등식 잔차(barrier − Σ9)는 평균 0.005, 최대 0.096 kcal/mol이다(ORCA 표 마감 오차). 그래서 손실의 벌점 항은 정답에서도 약 2e-4다.
