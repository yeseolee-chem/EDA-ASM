# D1PAIR FU2 — xTB 원자 토큰 모델 3종 (원자 단위 가산 · 그래프 신경망 · Transformer) — Claude Code 실행 지시서

**대상:** 브랜치 `d1-pair` `ae9f36b6` (`analysis/d1_pair/`). 이어서 commit한다.
**새 코드:** `analysis/d1_pair/fu2/` (`atoms.py`, `tokens.py`, `models.py`, `train.py`, `evaluate.py`) · **결과** `analysis/d1_pair/results/fu2/` · **scratch** `$D1PAIR_SCRATCH/fu2/`
**이 파일:** `docs/specs/D1PAIR_FU2.md`로 commit한다. rev 9·FU1의 규칙은 여기서 바꾸는 것 말고는 그대로다.
**보고:** Phase마다 보고. STOP이 나오면 멈추고 사용자에게 묻는다. 버그를 찾으면 고치기 전에 보고한다.
**순서:** Q(원자별 xTB, CPU) → T(토큰·그래프) → **C(코드 점검, 여기서 멈추고 사용자 승인)** → A(사전 등록 ADD2 commit) → M(학습, GPU) → E(평가) → R(보고)
**지금 SVR이 계산 중이다.** SVR job과 `M/units/` 캐시는 건드리지 않는다. CPU 작업은 `squeue` 개수 규칙 안에서 제출한다.

---

## 0. 사용자 결정 (2026-10-21)

| # | 결정 | 이유 (`results/followup/SUMMARY_FU1.md`) |
|---|---|---|
| F4 | 182개 feature 모델 대신 **원자 단위 입력**으로 바꾸고, 세 구조를 같은 입력·분할·평가로 비교한다: **AW**(원자 단위 가산), **GNN**(그래프 신경망), **TF**(Transformer) | learning curve에서 목표 MAE까지 학습 짝이 barrier·d1·cpcm 2.5–4.7배, elst·oi·pauli 12–64배 필요(외삽, 참고용) → 데이터를 그만큼 늘릴 수 없음. 넓은 XGB grid의 개선은 para MAE 0–6 % |
| F5 | 학습 대상은 arm C와 같다: 정답 = D1 짝 차이 ΔΔ(D0 모델 f0를 빼지 않음) | 잠정 T1에서 C가 B보다 모든 타깃에서 낮았고, C에 Δf0를 더해도 좋아지지 않았음 |
| F6 | 한 모델이 10개 타깃(8채널 + c_ghost + barrier)을 함께 예측한다. 보고의 중심은 8채널 + barrier | c_ghost는 항등식(barrier = 9개 합)을 맞추는 데만 쓴다 |

---

## Phase Q — 원자별 xTB 값 (CPU)

### Q-1. 대상과 계산
- 대상: `X182_d1.parquet`에 있는 D1 반응 **8,337개**(G1·feature 성공). 구조 = G1 최종 구조(`d1pair_common.G1_ROOT_D1F/<rxn_id>/{ts,rel1,rel2}.xyz`).
- 조각·반응 원자·원자 대응은 **`xtb_slice.ts_fragments(label, "g1", meta)`를 그대로** 쓴다(`ESPLEY_PARTITION=label`, rev 6·D1PAIR와 같은 환경 변수, `results/ENV_D1.md`).
  - TS 원자 i가 조각 A(dipole)면 반응물 원자 = `rel1`의 `inv0[i]`, B면 `rel2`의 `inv1[i]`(`P.map0`, `P.map1`의 역).
- 단일점 5개는 **`xtb_slice.xtb_sp`를 그대로** 부른다(`process_one`과 같은 순서와 전하): `gs1 = rel1`, `gs2 = rel2`, `d1 = fA`, `d2 = fB`(TS 모양 그대로 떼어 낸 조각), `ts = TS 전체`. 원자별 `charges`(Mulliken)와 `valence`(원자별 Wiberg 합)를 **버리지 않고 저장**한다.
- 확장(기본 포함): TS 하나에 `ext_features.xtb_run`으로 `--json`과 `--vfukui`를 돌려 `parse_json_sp`(원자 쌍극자 |μ|)와 `parse_fukui`(f⁺, f⁻)로 읽는다. ext_features의 자기 검사(json 전하 = `charges` 파일 5e-4 e 등)를 통과한 반응만 쓴다. 통과율이 99 % 미만이면 확장 값을 모든 모델에서 빼고 보고한다.
- 출력: `$D1PAIR_SCRATCH/fu2/atoms/<rxn_id>.npz`(원자적 쓰기, idempotent) + `results/fu2/Q_manifest.csv`(rxn_id, 원자 수, 상태, sha256).
- 실행: `sbatch` 배열(예: `--array=0-7`, `--cpus-per-task=48`, 반응 단위 병렬, `XTB_THREADS=1`). 반응당 xtb 7회, 전체 약 5.8만 회.

### Q-2. gate
- **Q-a (재현):** 저장한 원자별 값으로 기존 30열(`mulliken_*` 15, `wbo_valence_*` 15)을 `process_one`과 같은 인덱스로 다시 만들어 `X182_d1.parquet`와 비교한다. 최대 |차| ≤ 1e-6. 실패 반응이 0.5 %(42개)를 넘으면 **STOP**, 아니면 실패 반응을 빼고 목록을 남긴다.
- **Q-b:** 조각별 전하 합 = 조각 전하(D1은 모두 0) ± 1e-3, 모든 값 유한.
- 보고: 성공 수, 실패 사유, 확장 값 통과율.

---

## Phase T — 토큰과 그래프 (CPU)

반응 하나 = **원자 토큰 n개**(8–63, 중앙값 35) + (변형 "+G"에서만) **전역 토큰 1개**(그 반응의 `X182_d1` 182열).

### T-1. 원자 토큰 (TS 원자 순서)
| 묶음 | 값 |
|---|---|
| 정체 | 원소(데이터에 있는 원소의 one-hot), 조각(A = dipole / B = dipolarophile), 반응 원자 표시 5개(dip_a, dip_mid, dip_b, dph_a, dph_b) |
| 위치 | 가장 가까운 반응 원자까지의 결합 거리(공유결합 그래프 최단 경로, 8에서 자름) |
| 전하 | q_rel(따로 떨어진 반응물), q_frag(TS 모양 조각), q_TS(TS 전체), **Δq_dist = q_frag − q_rel**(변형이 만든 분극), **Δq_int = q_TS − q_frag**(상대가 만든 분극·전하 이동) |
| 결합 차수 | v_rel, v_frag, v_TS, Δv_dist, Δv_int (같은 정의) |
| 확장 | TS의 원자 쌍극자 |μ|, Fukui f⁺, f⁻ (Q-1 확장이 통과한 경우) |
| 좌표 | TS 직교좌표(Å). 모델은 원자 사이 거리만 쓴다(회전·이동에 무관) |

### T-2. 그래프 (GNN, AW용)
- 공유결합 간선 = `rel1`, `rel2`의 결합(`fragmenter.bond_matrix`)을 TS 번호로 옮긴 것. 형성 결합 2개(`formed_pairs_ts`)는 **다른 종류의 간선**으로 더한다.
- 간선 정보: 종류(공유 / 형성), TS 거리.

### T-3. 저장과 스케일
- `$D1PAIR_SCRATCH/fu2/tokens/`에 반응별 배열(패딩 없이), `results/fu2/T_report.json`(원소 목록, 열 순서, 개수, sha256).
- 숫자 열의 표준화(평균·표준편차)는 **바깥 학습 쪽 반응으로만** 계산하고 시험 쪽에 적용한다(학습 코드 안에서, fold마다).
- 목표 y: 학습 짝 표준편차로 나누기만 한다(평균 빼지 않음, rev 9 §3-2와 같음).
- **gate T-a:** 토큰 수 = 원자 수, NaN 없음, Q-a 실패 반응은 빠짐. 짝 중 a·b 하나라도 빠진 짝 수를 보고한다.

---

## Phase C — 코드 점검 (A 전에; 전체 학습 전에 반드시)

모델·학습 코드(`fu2/models.py`, `train.py`)를 다 쓴 뒤, 전체 학습 전에 아래 자동 검사를 `fu2/tests/`에 만들어 GPU 노드에서 sbatch로 돌린다. 결과 `results/fu2/C_tests.json`. **하나라도 실패하면 STOP.**

| # | 검사 | 통과 기준 |
|---|---|---|
| C1 | 순서 바꾸기: 세 모델 × 두 변형에서 g(b,a) = −g(a,b), g(a,a) = 0 | ≤ 1e-4 kcal/mol (float64 재계산) |
| C2 | 원자 순서 섞기: 반응 안 원자 순서를 무작위로 바꿈 | 예측 차이 ≤ 1e-4 |
| C3 | 회전·이동: TS 좌표를 무작위 회전·이동 | 예측 차이 ≤ 1e-4 |
| C4 | 패딩: 같은 batch에 원자가 많은 반응을 섞어 패딩을 늘림 | 예측 차이 ≤ 1e-4 (mask 확인) |
| C5 | batch 독립: 짝 하나만 batch로 돌린 값 = 큰 batch 안에서 돌린 값 | ≤ 1e-4 (cross-attention·pooling이 다른 짝 원자를 보지 않음) |
| C6 | 정답 정렬: 무작위 짝 50개에서 loader의 y = `F_pairs.csv`의 dd(a − b), 순서 바꾸기 증강에서 y 부호가 같이 바뀜 | 50/50 |
| C7 | 스케일 누출: 원자 feature·전역 토큰·y 표준화 통계를 만든 반응·짝 = 바깥 학습 쪽만 | 시험 쪽 0개 (assert) |
| C8 | 튜닝 누출: 튜닝 검증·early stopping에 쓴 짝 = 바깥 학습 쪽의 첫 안쪽 fold | 시험 pair_id 0개 |
| C9 | 항등식 벌점: 정답 y를 넣으면 벌점 항이 0에 가까움 | ≤ 1e-3 (구현·단위 확인) |
| C10 | AW 국소성: L = 1에서 어떤 원자로부터 5 Å 밖 원자를 움직여도 그 원자의 기여 ε_i가 그대로이고, E = Σε_i | ≤ 1e-6 |
| C11 | GNN 간선: 형성 결합 2개는 '형성' 종류로만 있고, 공유결합 간선 수 = 두 반응물의 결합 수 합 | 모든 반응 |
| C12 | 학습 가능성: 학습 짝 64개로 300 epoch → 학습 MAE가 null의 5 % 미만 | 세 모델 모두 |

- 그다음 코드와 `C_tests.json`을 **commit·push하고, 사용자에게 알린 뒤 멈춘다.** 사용자가 별도 코드 검토 후 승인하면 Phase A로 간다. 검토에서 고친 점은 `results/fu2/C_review.md`에 적는다.
- smoke test(M-3)도 이 단계에서 함께 돌려 보고한다.

---

## Phase A — 사전 등록 추가본 ADD2 (M 학습 전)

`results/PREREG_D1PAIR_ADD2.md` + `prereg_d1pair_add2.json`을 쓰고 **commit·push**. `fu2/train.py`는 commit과 sha256을 확인한 뒤에만 학습한다(`d1_split.gate_add1`과 같은 방식). 담을 것: §M의 모델·grid·학습 규칙·대표 모델 규칙, §E의 비교·판정 규칙, Q·T 결과 sha256. 기존 PREREG·ADD1은 바꾸지 않는다.

---

## Phase M — 세 모델 (GPU)

### M-1. 공통 규칙
- **짝·분할:** `d1_train.masks(u)`로 arm C unit과 **같은 학습·시험 짝**을 얻는다(T1: `split="T1"`, fold 0–4). 학습 = 학습 쪽 T_main(기준 짝 + anchor 짝), 시험 = 시험 쪽 P_sens ∪ A_main. a·b 원자 데이터가 없는 짝은 빼고 수를 기록한다.
- **입력 금지:** σ, 계열·core 번호, X 이름(원소와 구조는 토큰에 있음).
- **출력:** 10개(d1, d2, elst, pauli, oi, disp, cpcm, cds, c_ghost, barrier), 표준화된 단위.
- **손실:** 타깃별 |ŷ − y|의 평균(표준화 단위) + λ · |ŷ_barrier − Σ 9채널 ŷ| / s_barrier, **λ = 0.1**.
- **반대칭 (모델마다 구조로 보장):** 아래 M-2. **gate M-1′:** 학습한 모델마다 무작위 시험 짝 200개에서 |g(a,b) + g(b,a)| ≤ 1e-4 kcal/mol, |g(a,a)| ≤ 1e-4 kcal/mol(float64로 다시 계산). 넘으면 **STOP**.
- **학습:** AdamW(lr 1e-3, cosine 감소, weight decay 1e-4), dropout 0.1, batch 64짝, 최대 300 epoch, early stopping patience 30. 매 epoch 짝마다 (a,b)/(b,a) 순서를 무작위로 바꾼다(정답 부호도 같이).
- **튜닝:** 바깥 학습 짝 안에서 core로 묶은 `GroupKFold(4)`의 **첫 안쪽 fold**를 검증으로 쓴다(rev 9 NN 규칙과 같음). grid 칸마다 검증 MAE(표준화 단위, 9 주 타깃 평균)가 가장 낮은 칸과 그 best epoch를 고른다.
- **최종 모델:** 고른 칸으로 **바깥 학습 짝 전부**를 best epoch 수만큼 학습, **seed 5개**(20261019 + 0..4), 예측 = 5개 평균.
- **변형 2개:** "A"(원자 토큰만), "A+G"(원자 토큰 + 전역 토큰 182). 모델 3 × 변형 2 = **구성 6개**.

### M-2. 모델 구조 (모두 순수 PyTorch, 반응별 패딩 + mask; PyG 등 추가 의존성 없이)

| 모델 | 구조 | 반대칭 | grid (4칸) |
|---|---|---|---|
| **AW** 원자 단위 가산 | 원자 토큰 임베딩 → 거리 5.0 Å 안 이웃과 연속 필터 상호작용층 L개(거리 RBF 32개, cosine cutoff) → 원자마다 10개 기여 ε_i. 반응 값 E = Σ_i ε_i ("A+G"는 + φ(전역 토큰)) | **g = E(a) − E(b)** (빼기라 정확) | L ∈ {1, 2} × 폭 ∈ {64, 128} |
| **GNN** 그래프 신경망 | T-2 그래프에서 메시지 전달 L층(간선 종류·거리 RBF, residual + layer norm) → 반응 벡터 r = [합, attention pooling] ("A+G"는 전역 토큰 투영을 r에 이어 붙임) → h(a,b) = MLP([r_a, r_b]) | **g = ½[h(a,b) − h(b,a)]** | L ∈ {4, 6} × 폭 ∈ {64, 128} |
| **TF** Transformer | 원자 토큰(+ "A+G"의 전역 토큰) → self-attention L층(4 head; attention 점수에 원자 간 거리 RBF와 같은 조각·형성 결합 여부를 head마다 학습한 bias로 더함) → cross-attention 1블록(a의 토큰이 b를, b가 a를 봄, 가중치 공유) → [평균, attention pooling] → h(a,b) | **g = ½[h(a,b) − h(b,a)]** | L ∈ {2, 3} × 폭 ∈ {64, 128} |

- **대표 구성:** 타깃마다 6개 구성 중 튜닝 검증 MAE(그 타깃)가 가장 낮은 것. 시험 점수로 고르지 않는다. 모든 구성의 결과를 보고한다.

### M-3. 실행·캐시
- 단위 = 구성 6 × 바깥 fold 5. 튜닝 4칸 + 최종 seed 5개 = 단위당 9번 학습, 전체 270번.
- 캐시: `$D1PAIR_SCRATCH/fu2/units/T1__<model>__<variant>__f<k>.{json,pt,parquet}`. json에 입력 지문(Q·T sha256, 짝 목록 sha256, grid, 코드 버전 `d1pair-fu2-1`, torch·CUDA 버전, GPU 이름)을 넣는다. 다른 지문의 캐시를 만나면 exit 3.
- 예측 저장 `results/fu2/M_predictions.parquet`: pair_id, ref_kind, fold, split, model, variant, target, y, yhat(seed 평균), yhat_seed_sd.
- **GPU:** UBAI GPU 파티션은 `sinfo`로 확인한다. 새 환경 `env_d1fu2`(torch·CUDA 버전 고정, `env.yaml`에 기록). 첫 제출 전 **smoke test**(fold 0, 구성마다 2 epoch, gate M-1′ 확인)를 sbatch로 돌려 보고한다.
- 시간 예상: 짝 약 6,400개 × 원자 약 35개로 작아 학습 1회 수 분–수십 분, 전체 GPU 수십 시간.
- GPU 비결정성은 허용하고 기록한다(seed 고정, `torch.backends.cudnn.deterministic=True` 시도).

### M-4. 보조 분할 (T1이 끝난 뒤)
- LODO(5)·LOSO(10)는 **모델마다 T1 대표 구성 하나씩**(구성 3개)만, 같은 규칙으로 돌린다. 결과는 부 비교다.

---

## Phase E — 평가 (`results/fu2/`)

- **비교 상대 (같은 시험 짝):** 기존 arm C의 XGB, XGBW, 대표 모델(지금은 SVR을 뺀 5개 중 대표, SVR이 끝나면 6개로 다시 계산), null(ΔΔ = 0). 원자 데이터가 빠진 짝은 모든 모델에서 똑같이 뺀다.
- **지표:** `d1_eval`과 같은 함수로 MAE(전체·para·위치별·anchor), R²_fam, ρ 오차, 꺾임·V 판정(부록 D `hammett_eval.calls`), 채널별 목표 MAE 대비 배수(FU1 L-3 표; 사후 정의임을 적는다), 숨은 V 6계열·숨은 꺾임 30계열.
- **주 비교 (ADD2에 고정):** 타깃 9개(8채널 + barrier)마다 **원자 모델 대표 대 C-XGB**, d = MAE(C-XGB) − MAE(원자 대표), P_main 시험 짝, core cluster bootstrap 2,000회(seed 20261019), **Holm 9개**.
  - 판정: Holm p < 0.05이고 CI 하한 > 0인 타깃이 **5개 이상**이면 "원자 입력이 182개 feature보다 낫다". 2–4개면 "일부 채널에서만", 1개 이하면 "이득 없음".
- **부 비교 (보정 없음):** 모델 사이(AW 대 GNN 대 TF, 같은 변형), 변형 사이("A+G" 대 "A" = 원자 정보 위에 182개가 더하는 몫, 그리고 "A+G" 대 C-XGB = 182개 위에 원자가 더하는 몫), para 짝만, C 대표 대비, LODO·LOSO.
- **해석 (탐색):** rev 9 fig3의 계열 6개에서 AW의 원자별 기여 지도(a와 b의 원자별 ε를 각각 분자 그림에 색으로, 채널 oi·elst·pauli), TF의 cross-attention 지도. 검증: 작용기 원자의 Δq·Δv를 0으로 바꿨을 때 예측 변화. 그림은 "attention·기여는 설명이 아니라 진단"이라고 설명에 적는다.
- **그림:** `figF2_1_mae`(타깃 × [null, C-XGB, C-XGBW, AW, GNN, TF] 상대 MAE, 전체·para), `figF2_2_hammett_level`(R²_fam·꺾임 민감도·위양성), `figF2_3_examples`(rev 9 fig3 계열 3개 × barrier·oi·elst, DFT vs C-XGB vs 원자 대표), `figF2_4_atoms`(해석 그림).

---

## Phase R — `results/fu2/SUMMARY_FU2.md` (한국어)
- 맨 앞 세 문장: Q·T 결과(반응 수), 주 비교 판정, 가장 나은 구조.
- 절: Q → T → A(commit hash) → M(구성별 표, gate M-1′) → E(주 비교·부 비교·Hammett·해석) → 이탈 → 한계.
- 반드시 적을 한계: T1은 이미 여러 번 본 분할이다(새 모델은 시험 짝을 튜닝에 쓰지 않았음) · 구조 출발점이 DFT TS(G1) · 원자별 기여와 attention은 해석이 아니라 진단 · SVR이 끝나기 전 결과면 그 사실.

---

## 규칙
- HPC: python은 모두 `sbatch`(`--time=48:00:00`). 로그인 노드에서는 편집·git·`squeue`·`scancel`·`sbatch`만. MaxJobs 10, MaxSubmit 20.
- SVR job, `M/units/`, `M/units_fu/`, PREREG·ADD1, rev 4–6 파일은 바꾸지 않는다. 새 파일은 `fu2/`, `results/fu2/`, `$D1PAIR_SCRATCH/fu2/`에만 쓴다.
- idempotent, 원자적 쓰기. 수치는 파일에서 옮긴다.
- 결과를 본 뒤 모델·grid·변형·판정 규칙을 바꾸지 않는다. 바꾸면 이탈로 기록한다.

## STOP

| Phase | 조건 |
|---|---|
| Q | Q-a 재현 실패 반응 > 0.5 %; 조각 전하 합 불일치 |
| T | 토큰 수 ≠ 원자 수; NaN |
| C | C1–C12 중 하나라도 실패; 사용자 승인 전 Phase A·M 진행 |
| A | ADD2 commit 전 M 학습 시도 |
| M | gate M-1′ 실패; 캐시 지문 불일치; smoke test 실패; 튜닝에 시험 짝 사용 |
| E | 비교하는 모델 사이 시험 짝 목록 불일치 |
