# DEPLOY-MODEL-W — 같은 계열 안의 치환체끼리 짝을 지어 함께 학습해 꺾임 판정 오차를 줄이기 (방법 B) — Claude Code 실행 지시서

**저장소:** 브랜치 `deploy` (DEPLOY_MODEL 명령서와 같은 브랜치, 같은 worktree `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy`).
**이 파일:** `docs/specs/DEPLOY_MODEL_W.md`로 commit한다.
**새 코드:** `analysis/deploy/w/` · **결과:** `analysis/deploy/results/w/` · **scratch:** `$DEPW=/gpfs/tmp_cpu2/yeseo1ee/deploy/w`
**읽기만 하는 것:** DEPLOY_MODEL의 모든 산출물(`$DEP` 아래, `analysis/deploy/results/`), `$LAD`, `$D1PAIR_SCRATCH`,
`analysis/d1_pair/results/S_folds.csv`, 기존 `analysis/` 코드 전부. **DEPLOY_MODEL 코드는 고치지 않는다**(13절의 `predict_tf.py`
확장만 예외, 기존 검사 I-1 · I-2를 다시 통과해야 함).

**순서:** 0 → B(계열 안 짝) → C(코드 검사) → R(사전 등록) → K(교차검증) → P(기준 모델의 그래프 읽기 예측) → E(평가)
→ D(판정) → [채택 시] W2(최종 모델) → I2(추론기 확장 · 검사) → S(요약 후 **멈춤**)
- 15절 STOP 조건에서만 멈추고 묻는다. 버그를 찾으면 고치기 전에 보고한다.
- 모든 계산은 `sbatch`, `--time=48:00:00`(저장소 `CLAUDE.md`).

---

## 0. 무엇을, 왜

### 0-1. 지금까지 확인한 것 (LADDER g2 · S · TF 시험 예측 원자료, 2026-11-03)

| 항목 | 값 |
|---|---|
| 숨은 꺾임 계열(장벽 직선 & 채널 ≥ 1 꺾임) 판정, para 89계열 | 민감도 0.65, 위양성률 0.21, PPV 0.57 (기저율 0.29) |
| 계열 안 예측 오차 중 σ에 대한 직선 성분 (꺾임 판정에 무해) | ≈ 60 % |
| 나머지 비선형 성분 = **유해 오차** | ≈ 40 %, 신호(DFT 꺾임 크기) / 유해 오차 = 0.9–1.9 |
| 유해 오차만 축소했을 때(모양 고정 가정) | −30 % → 민감도 0.85, PPV 0.73 · −70 % → PPV 0.81 · −80 % → PPV 0.93 |
| DFT 구조로 학습한 모델 (상한 참고) | 민감도 0.78, PPV 0.66 |
| 효과 없음 확인 | seed 증가(유해 분산의 ≈ 12–20 %만 seed 잡음), ENS4(PPV 0.38), HF 추가(0.41), p+m 결합(0.49) |

### 0-2. 방법 B의 생각

지금 모델은 **(치환 반응, 기준 반응)** 짝만 배운다. 꺾임은 같은 계열 안에서 **치환기 X1과 X2 사이의 차이**가 σ에 따라
어떻게 변하는지로 정해진다. 모델 `g(a, b) = ½[h(a,b) − h(b,a)]`는 어떤 두 반응이든 받으므로, 같은 기준 반응을 공유하는
치환 반응끼리 짝 **(X1, X2)**를 학습에 더한다.

- 라벨은 **새 계산 없이 정확히** 정해진다: `dd_t(a1,b) − dd_t(a2,b) = (y_a1 − y_b) − (y_a2 − y_b) = y_a1 − y_a2`
  (`analysis/d1_pair/d1_pairs.py` 9행: `dd_<t> = y_a − y_b`).
- 근거: 짝으로 직접 차이를 학습하면 데이터가 조합적으로 늘고 체계적 오차가 상쇄된다고 보고됨 (DeepDelta, Fralish et al.,
  *J. Cheminform.* 2023, 15, 101, DOI 10.1186/s13321-023-00769-x). 우리 과제에서는 검증되지 않았다 → 이 명령서가 검증한다.
- 규모(D1 S_folds 기준, 분할 전): 같은 (기준 반응, 계열, 위치) 묶음 353개, 묶음 크기 중앙값 19, 가능한 짝 ≈ 79,600개
  (지금 학습 짝의 ≈ 11배).

### 0-3. 비교하는 세 모델 (입력은 모두 lf, S arm, TF)

| 이름 | 학습 짝 | 설정 선택 지표 | 출처 |
|---|---|---|---|
| `lf` | 기준 짝만 | 내부 검증 MAE (FU2 규칙) | DEPLOY_MODEL K (이미 있음) |
| `lfV` (대조) | 기준 짝만 (r = 0) | 내부 검증 **유해 오차 V_nl** | 이 명령서 K (lfW 튜닝의 r = 0 부분) |
| `lfW` (방법 B) | 기준 짝 + 계열 안 짝 (r ∈ {0, 1, 4} 중 선택) | V_nl (+ MAE 보호 조건) | 이 명령서 K |

`lfW` 대 `lf` = 배포 판정. `lfW` 대 `lfV` = 개선이 계열 안 짝 때문인지, 선택 지표 때문인지 가르는 해석용.

### 0-4. 읽기 방식 두 가지 (추론만 다름, 학습은 같음)

| 이름 | 반응 i의 예측 | 비고 |
|---|---|---|
| **direct** | `g(i, ref)` | 지금 방식 |
| **graph** | `(1/m) Σ_{j∈G} [g(i, j) − g(ref, j)]`, G = 그 계열의 반응 전부(ref 포함, m = \|G\|) | 모든 경로의 평균. 모델이 완전히 가법적이면 direct와 같다. 2차 지표로 평가 (DeepDelta의 가법성 `g(x,y)+g(y,z)=g(x,z)` 활용) |

---

## 1. 시작 조건 (하나라도 아니면 STOP)

| # | 조건 |
|---|---|
| 0-1 | DEPLOY_MODEL `results/C_tests.json` `all_pass`, `PREREG_DEPLOY.md` commit됨, `E_report.json`의 G-C 통과 |
| 0-2 | `$DEP/units/D__lf__S__TF__f{0..4}.{json,parquet}`, `__s{0..4}.pt`, `results/K_predictions.parquet` 있음 |
| 0-3 | `analysis/d1_pair/results/S_folds.csv`가 있고, 열 `pair_id, ref_kind, a_rxn, b_rxn, core, fam, pos, Xb, sigma, in_T_main, in_P_main, fold_T1`을 가진다. DeployData의 짝 표와 `pair_id`로 합쳤을 때 `a_rxn`, `b_rxn`이 모두 같다 |
| 0-4 | DEPLOY_MODEL의 입력 sha256(`labels_lf.json`, `X182_lf.parquet`, lf `T_manifest.csv`, `F_pairs_cr2.csv`, T1 분할 5개)이 `prereg_deploy.json`과 같다 |
| 0-5 | 디스크 여유 20 GB 이상 |

DEPLOY_MODEL의 W(최종 lf 모델) · I는 끝나 있어도 되고 아니어도 된다. 끝나 있으면 lf 최종 모델은 **대체용**으로 남는다.

---

## 2. 공통 (`analysis/deploy/w/w_common.py`)

- `dep_common`을 import (경로 · 쓰기 가드 · sha256 · 원자적 쓰기). 쓰기 허용 = `$DEPW`, `analysis/deploy/results/w`.
- 상수: `R_GRID = (0, 1, 4)`, `MAE_GUARD = 1.05`, `MIN_GROUP_VAL = 6`(V_nl에 쓰는 묶음의 최소 짝 수),
  `MIN_FAM_EVAL = 12`(평가 계열의 최소 점 수, H 포함 — `hammett_eval.family_series`와 같음), `PRIMARY = 9개 타깃`
  (d1 d2 elst pauli oi disp cpcm cds barrier), `SEED = 20261019`, `N_BOOT = 2000`.
- **묶음(group) 키 = (`b_rxn`, `fam`, `pos`)** — 같은 기준 반응, 같은 계열, 같은 위치. `S_folds.csv`에서 읽는다.
  `ref_kind`는 reference와 anchor 모두 허용(둘 다 같은 b를 공유하는 묶음이 있음).

---

## 3. Phase B — 계열 안 짝 (`analysis/deploy/w/within.py`)

### 3-1. 정의

fit 하나의 학습 짝 목록 `tr`(DeployData 행 번호, 사용 불가 짝 제거 후)에 대해:
- 같은 묶음 키를 가진 학습 짝 i < j마다 계열 안 짝 하나: `ra = D.ra[i]`, `rb = D.ra[j]`, `y = D.y[i] − D.y[j]` (10 타깃 전부).
- 두 반응이 모두 그 fit의 학습 반응이므로 새 반응은 들어오지 않는다(누출 없음 — C-W2로 확인).
- 샘플링 가중치: 묶음 g(학습 짝 n_g개, 계열 안 짝 C(n_g, 2)개)의 각 짝에 **w = 2 / (n_g − 1)** → 묶음마다 계열 안 짝의 기대
  출현 수가 그 묶음의 기준 짝 수에 비례(큰 묶음이 짝 수의 제곱으로 지배하지 않게).

### 3-2. 산출 (fold k = 0..4, 그리고 `all`)

`$DEPW/within/<split>__f<k>__{train,inner_train}.npz` — `ra, rb, y, w, gid`(정수 묶음 번호) + `meta.json`(묶음 수, 짝 수, sha256).
`$DEPW/within/test_groups__f<k>.parquet` — 시험 짝의 묶음(평가 · graph 읽기용): `pair_id, gid, b_rxn, fam, pos, Xb, sigma`.
`results/w/B_report.json` — fold별 기준 짝 수, 계열 안 짝 수, 묶음 크기 분포.

```
python within.py build      # 5 fold + all, 멱등
```
CPU, `espley` 또는 `env_d1fu2` 환경. 수 분.

---

## 4. 학습 함수 `fit_w` (`analysis/deploy/w/train_w.py`)

`fu2/train.py`의 `fit`을 **복사해서 아래만 바꾼다.** `Std`, `batch`, `orient`, `loss_fn`, `predict`, `gate_m1`,
`Mo.build`는 import해서 그대로 쓴다.

| # | 바꾸는 것 | 내용 |
|---|---|---|
| W-1 | 표준화 | `std = Std(D, tr_base)` — **기준 짝만으로** 토큰 수치열 · 전역 토큰 · `sy` 통계. 계열 안 짝은 표준화에 영향 없음 (r에 따라 `sy`가 바뀌지 않게) |
| W-2 | epoch 구성 | epoch마다 계열 안 짝 `n_w = r × len(tr_base)`개를 가중치 w로 **비복원** 추출(풀이 작으면 전부) → 기준 짝과 이어 붙여 `epoch_batches`에 넘김 (순서 섞기, (a,b)/(b,a) 뒤집기 그대로). y는 `/ std.sy` |
| W-3 | 난수 | 계열 안 추출은 **별도 생성기** `rng_w = default_rng(seed + 7919)`. r = 0이면 `rng_w`를 쓰지 않는다 → r = 0 · MAE 선택일 때 `fit`과 같은 난수 흐름 (C-W3) |
| W-4 | 검증 지표 | epoch마다 내부 검증 짝 예측으로 **V_mae**(FU2 `rec["val"]`과 같은 식)와 **V_nl**(5절)을 모두 기록. 조기 종료 · 최적 epoch 기준 J = `select` 인자 (`"nl"` 또는 `"mae"`) |
| W-5 | 손실 | 바꾸지 않음: 표준화 평균 \|ŷ − y\| + 0.1 × 장벽 일관성 벌점 (계열 안 짝 라벨도 차이이므로 장벽 = Σ₉ 관계가 그대로 성립) |
| W-6 | 그 외 | AdamW 1e-3 / 1e-4, cosine T_max 300, dropout 0.1, batch 64, patience 30, 최대 300 epoch, 반대칭 관문 — 모두 FU2 그대로 |

반환: `fit`과 같은 dict + `best_val_mae`, `best_val_nl`, `history`에 epoch별 `val_mae`, `val_nl`, `n_within`.

---

## 5. 선택 지표 V_nl (내부 검증 유해 오차)

내부 검증 짝(기준 짝, 사용 가능한 것)에서:
1. 묶음 키별로 모은다. `pos ∈ {p, m}`이고 짝 수 ≥ `MIN_GROUP_VAL`(6)인 묶음만 쓴다.
2. 묶음 g, 타깃 t마다 점 = 짝들의 (σ, e = ŷ − y) [kcal/mol] + 기준점 (0, 0). e를 σ에 대해 최소제곱 직선으로 맞춘 잔차의 RMS = `r_gt`.
3. `V_nl = mean_{t∈PRIMARY} [ mean_g r_gt / sy_t ]` (`sy_t` = 그 fit의 `std.sy`).

σ는 **평가에만** 쓴다(모델 입력 아님 — DEPLOY_MODEL C-6 유지). 내부 검증 묶음 수를 `history`에 기록한다(fold마다 대략 20–40개 예상).

---

## 6. Phase C — 코드 검사 (`analysis/deploy/w/tests_w.py`) — 하나라도 실패하면 STOP

| # | 검사 | 기준 |
|---|---|---|
| C-W1 | 라벨 정확성: `F_pairs_cr2.csv`의 dd가 묶음 안에서 **공통 y_b**를 뺀 값인지 확인. 반응별 값(`d1pair_common.load_d1_labels()`, 열 = `d1_pairs.py`의 `TARGETS` 대응)으로 묶음마다 `y_a − dd(a, b)`를 계산 → 묶음 안 표준편차. 그리고 무작위 계열 안 짝 2,000개의 y = `y_a1 − y_a2`(반응별 값) 비교 | 10 타깃 모두 묶음 안 표준편차 ≤ 1e-6, 짝 \|차\| ≤ 1e-6 kcal/mol. 벗어나면 먼저 cr2가 a쪽 반응 값을 고쳤는지(`analysis/d1_ladder/results/V_report.json`, `refs_cr2.parquet` 기록) 확인해 고친 값으로 다시 계산. 그래도 벗어나면 어느 타깃 · 묶음인지 보고하고 STOP (짝마다 다른 보정 = 계열 안 라벨을 뺄셈으로 만들 수 없음) |
| C-W2 | 누출: 모든 계열 안 짝의 두 반응 ∈ 그 fit의 학습 반응. 튜닝용(inner_train)에는 inner_val · 시험 반응 없음, 바깥 fold용에는 시험 반응 없음 | 위반 0 |
| C-W3 | **r = 0 환원**: CPU float64, 작은 부분집합(짝 300, 칸 2층 × 64, 5 epoch), 같은 seed로 `fit_w(r=0, select="mae")`와 `FT.fit` | epoch별 손실 · 검증값 · 최종 가중치가 비트 단위로 같음 |
| C-W4 | 표준화: fold k의 `fit_w`가 만든 `sy`, `mu`, `sd`, `gmu`, `gsd` = `D__lf__S__TF__f<k>__s0.pt`에 저장된 값 | ≤ 1e-12 |
| C-W5 | V_nl 구현: 합성 자료 — 정확한 직선 오차 → 0; 알려진 꺾임을 넣은 오차 → 해석적 RMS와 ≤ 1e-9 | 통과 |
| C-W6 | 가중 추출: 10,000 epoch 모의 추출에서 묶음별 출현 비율이 n_g 비례와 2 % 안 | 통과 |
| C-W7 | graph 읽기: (a) 완전 가법적인 합성 g → graph = direct (≤ 1e-12) (b) 반대칭 `g(i,j) = −g(j,i)` 이용한 식과 직접 합이 같음 | 통과 |
| C-W8 | 반대칭 관문 M-1′: 모든 튜닝 · 최종 fit | DEPLOY_MODEL C-4와 같은 기준 |
| C-W9 | smoke: fold 0, r = 4, 칸 최소 · 최대, 2 epoch, seed 1개 (GPU) | 끝까지 돌고 C-W8 통과, epoch당 시간 기록 |

결과 `results/w/C_tests_w.json` (`all_pass`).

---

## 7. Phase R — 사전 등록 (`results/w/PREREG_W.md` + `prereg_w.json`)

K 학습 전에 commit한다. 학습 명령은 두 파일이 git에 있고 변경이 없고 json의 `md_sha256` = md의 sha256일 때만 돈다.

**md에 적을 것**
1. 0절 설계, 3–5절 정의, 아래 판정 규칙
2. **지금까지 본 것**: 0-1절 표(LADDER g2 분석), DEPLOY_MODEL의 lf 결과(K · E). lfV · lfW 결과는 0개
3. 입력 sha256: DEPLOY_MODEL 입력 전부 + `S_folds.csv` + B 산출물(`meta.json`들)

**평가 대상 계열 (고정):** lf의 시험 예측에서 para 계열(`hammett_eval.family_series` 정의, 점 ≥ 12, H 포함). 세 모델 모두
같은 사용 가능 규칙이므로 계열 · 점 집합이 같다(다르면 STOP).

**1차 지표 H (시험 유해 오차):**
`H = mean_{t∈PRIMARY} mean_f [ r_ft / s_t ]`, `r_ft` = 계열 f, 타깃 t의 (σ, e = ŷ − y) + (0, 0) 직선 잔차 RMS,
`s_t` = 평가 계열 전체의 DFT y의 SD(모델과 무관한 고정 상수). **R = H_lfW / H_lf**, core 부트스트랩(2,000회, seed 20261019) 95 % CI.

**채택 규칙 (둘 다 만족하면 lfW를 배포 모델로 채택)**

| # | 조건 |
|---|---|
| A-1 | R의 95 % 상한 < 1.00 (유해 오차가 유의하게 작음) |
| A-2 | 비열등: 9개 주요 타깃마다 para P_main 상대 MAE 차 (MAE_lfW − MAE_lf) / MAE_lf의 95 % 상한 ≤ +5 % (DEPLOY_MODEL G-C와 같은 기준) |

등급: R ≤ 0.70 = "목표 달성"(0-1절 모의에서 PPV ≈ 0.73에 해당), 0.70 < R이고 A-1 통과 = "부분 개선", A-1 실패 = "개선 없음".

**2차 (graph 읽기, 추론 전용):** 채택된 모델(lfW 또는 lf)에서 R_g = H_graph / H_direct의 95 % 상한 < 1.00이고 A-2와 같은
비열등(graph 대 direct)을 만족하면 배포 추론을 graph로 한다. 아니면 direct.

**기술 지표 (판정에 쓰지 않음, 모두 보고):** 숨은 꺾임 민감도 · 위양성률 · PPV(+ 계열 부트스트랩 CI, 짝지은 차이 CI), 채널별
꺾임 민감도 · 위양성률, 연속 g AUC, 타깃별 H, lfW 대 lfV의 R, 0-1절 모의표와 비교한 기대 PPV.

---

## 8. Phase K — 교차검증 (`train_w.py run`)

### 8-1. 단위와 튜닝

- 단위 `D__lfW__S__TF__f{0..4}` (한 작업이 대조 `D__lfV__S__TF__f{k}` 기록도 함께 쓴다).
- 튜닝 grid: 칸 {2, 3}층 × {64, 128} **×** r ∈ {0, 1, 4} = 12 fit. 모두 `select="nl"`, `inner_train`(+ 그 계열 안 짝)으로 학습,
  `inner_val`로 평가, 튜닝 seed 20261019.
- **lfW 선택**: 적격 = `V_mae ≤ MAE_GUARD × min_{r=0 칸} V_mae` (모두 각자 최적-J epoch 값). 적격 중 V_nl 최소.
  동률은 r 작은 쪽, 그다음 칸 grid 순서.
- **lfV 선택**: r = 0인 4칸 중 V_nl 최소 (동률 grid 순서). 가드 없음.
- 최종(fold): 선택된 (칸, r)로 바깥 학습 짝 전부(+ 그 계열 안 짝), 최적 epoch에서 멈춤, seed 20261019 + 0..4, 예측 평균.
  lfW와 lfV가 같은 (칸, r=0)을 고르면 최종 fit을 한 번만 하고 두 기록이 같은 `.pt`를 가리킨다.

### 8-2. 저장

`$DEPW/units/<unit>.{json,parquet}`, `__s{0..4}.pt`(state dict + 표준화 통계 `mu, sd, gmu, gsd, sy` + 칸 + r + `d_in`, `d_glob`).
json: 지문(코드 sha256, 입력 sha256, PREREG_W sha256, grid, 규칙, GPU 이름), 튜닝 12개 기록(`val_mae`, `val_nl`, best epoch,
초, n_within), 선택(lfW · lfV 각각), 최종 seed별 기록, 관문 결과.
parquet 열 = DEPLOY_MODEL `K_predictions.parquet`와 같음(`rung` = `"lfW"` 또는 `"lfV"`).

### 8-3. 명령 · 자원

```
python train_w.py smoke              # = C-W9
python train_w.py run <i|@I> <n>     # 단위 [i::n], PREREG_W 관문 통과해야 실행, 멱등(지문 비교)
python train_w.py collect            # → results/w/K_predictions_w.parquet, K_val_w.csv, K_report_w.json
```
GPU `env_d1fu2`, RTX 3090 우선. 배열 `--array=0-4%5`. 예상: r = 0 fit ≈ 100 s, r = 1 ≈ 2배, r = 4 ≈ 5배(LADDER g2 TF 실측
unit 465–884 s 기준) → fold당 ≈ 1–1.6 h, 5 fold ≈ 5–8 GPU-h. C-W9에서 epoch당 시간을 재서 `results/w/K_estimate.json`에
기록하고, 예상이 fold당 20 h를 넘으면 STOP.

---

## 9. Phase P — graph 읽기 예측 (`analysis/deploy/w/within_predict.py`)

fold k의 시험 묶음(`test_groups__f<k>.parquet`)마다 묶음 안 모든 순서 없는 짝 (i, j)(기준 반응 포함)를 **그 fold의 모델**로 예측한다
(5 seed 각각, 평균은 나중에).

| 모델 | 단위 |
|---|---|
| lf | `$DEP/units/D__lf__S__TF__f<k>__s*.pt` (DEPLOY_MODEL) |
| lfV, lfW | `$DEPW/units/...` |

산출 `$DEPW/within_pred/<unit>.parquet`: `gid, a_rxn, b_rxn, seed, target, g`. graph 읽기는 E에서 계산한다.
`predict_tf.py`의 입력 검사 · 표준화(각 `.pt`의 통계 사용)를 그대로 쓴다. 예상 짝 수 fold당 ≈ 1.5–3만 → GPU 수 분.

```
python within_predict.py run <i|@I> <n>     # (모델 × fold) 15개 중 [i::n]
```

---

## 10. Phase E — 평가 (`analysis/deploy/w/eval_w.py`)

평가 조합 5개: lf-direct, lf-graph, lfV-direct, lfW-direct, lfW-graph. lfV-graph도 계산하되 해석용.

| 산출 | 내용 |
|---|---|
| `E_H.csv` | 조합 × 타깃 × 계열: `r_ft`, 그리고 조합별 H (+ 타깃별 H), core 부트스트랩 CI |
| `E_ratio.csv` | R = H_lfW / H_lf, R_g(채택 후보 각각), lfW / lfV, lfV / lf — 점추정과 95 % CI. 같은 부트스트랩 표본으로 짝지어 계산 |
| `E_mae.csv` | 조합 × 타깃: para P_main MAE, 상대 차와 CI (`analysis/geo2/eval_geo2.py` `boot_diff`와 같은 core 부트스트랩) |
| `E_bends_w.csv` | 계열 × 타깃: DFT 판정 · 예측 판정 · g (`hammett_eval.calls`, `bent_gain`) — 조합마다 |
| `E_bend_rates_w.json` | 조합별 숨은 꺾임 민감도 · 위양성률 · PPV(CI), lf 대비 짝지은 차이(CI), 채널별 민감도 · 위양성률, 연속 g AUC |
| `E_report_w.json` | A-1 · A-2 판정, 등급, 2차 판정, 입력 sha256 |

DFT 정답 판정은 lf 평가와 같은 점 집합에서 만든다(LADDER 분석 `hk.py`와 같은 방식: 시험 예측이 있는 점 + H).

---

## 11. Phase D — 판정 (`eval_w.py decide`)

`E_report_w.json`을 읽어 `results/w/D_decision.json`을 쓴다:
- 채택 모델 = A-1 · A-2 통과 시 `lfW`, 아니면 `lf`
- 읽기 방식 = 채택 모델의 2차 판정 결과 (`graph` 또는 `direct`)
- 등급과 근거 수치

**채택 모델이 lf이고 읽기 방식이 direct이면** W2를 건너뛰고 I2의 검사만 한 뒤 S로 간다(배포 모델 = DEPLOY_MODEL의 lf 최종).
**채택 모델이 lf이고 읽기 방식이 graph이면** W2 없이 I2에서 lf 최종 모델에 graph 읽기를 붙인다.

---

## 12. Phase W2 — 최종 배포 모델 (`train_w.py final`) — lfW 채택 시에만

| 항목 | 규칙 |
|---|---|
| 학습 짝 | DeployData `split("all")` (사용 가능한 `in_T_main` 짝 전부) + 그 계열 안 짝 (`$DEPW/within/all__train.npz`) |
| r | 5 fold lfW 선택 r의 **최빈값** (동률 → 작은 r) |
| 칸 | 선택 r이 최빈값인 fold들의 칸 최빈값 (동률 → 그 fold들의 튜닝 V_nl 평균이 낮은 칸, 그다음 grid 순서) |
| epoch | 선택 r이 최빈값인 fold들의 최적 epoch **중앙값** (개수가 짝수면 아래쪽 중앙값). `best_epoch`는 1-기반, 최종 fit은 `n_epochs = best_epoch` (DEPLOY_MODEL 13절과 같은 규약) |
| seed | 20261019 + 0..4 |
| 표준화 | 기준 짝 전체 반응 (W-1) |
| 관문 | C-W8 (무작위 학습 짝 200 + 계열 안 짝 200) |
| 산출 | `$DEPW/model/D__lfW__S__TF__all__s{0..4}.pt`, `$DEPW/model/MODEL_CARD_W.json`, `results/w/W2_report.json` |

`MODEL_CARD_W.json` = DEPLOY_MODEL `MODEL_CARD.json`의 모든 항목 + r, 계열 안 짝 수 · sha256, 읽기 방식, 교차검증 성능(E 요약:
H, R, MAE, 숨은 꺾임 지표), 적용 범위 문턱(lfW 시험 짝 `yhat_seed_sd`의 타깃별 95 백분위).

---

## 13. Phase I2 — 추론기 확장과 검사

### 13-1. `analysis/deploy/predict_tf.py` 확장 (이 명령서에서 유일하게 DEPLOY_MODEL 파일을 고치는 곳)

```
python predict_tf.py run <tokens_root> <x182.parquet> <pairs.csv> <out.parquet> \
       [--model final|cv] [--model-dir <dir>] [--readout direct|graph]
```
- `--model-dir`: 기본 `$DEP/model` (lf). lfW 채택 시 `$DEPW/model`. `MODEL_CARD*.json`을 읽어 칸 · r · 읽기 방식 기본값을 정한다.
- `--readout graph`: `pairs.csv`에 `group_id` 열이 필수(같은 기준 반응 · 계열 · 위치). 묶음마다 모든 짝을 예측해 0-4절 식으로
  계산. 묶음 크기 > 80이면 그 묶음은 direct로 하고 사유를 기록.
- 출력 열에 `readout` 추가. 나머지는 DEPLOY_MODEL 14-1절과 같음.

### 13-2. 검사 (`results/w/I2_selftest.json`)

| # | 검사 | 기준 |
|---|---|---|
| I2-0 | DEPLOY_MODEL I-1 · I-2를 확장된 `predict_tf.py`로 다시 실행 (기본 옵션) | 이전과 같은 기준 통과 |
| I2-1 | `--model cv --model-dir $DEPW/units --readout direct`로 lfW 시험 짝 → `K_predictions_w.parquet`(fold별 seed 평균)과 비교 | ≤ 1e-4 kcal/mol |
| I2-2 | `--readout graph`로 시험 묶음 → E의 graph 값과 비교 | ≤ 1e-4 |
| I2-3 | (W2 했을 때) `--model final`로 학습 짝 1,000개 → W2 학습 직후 기록한 예측 | ≤ 1e-6 (같은 GPU) |
| I2-4 | DEPLOY_MODEL I-3 연쇄 smoke(SMILES부터 10반응)를 채택 모델 · 읽기 방식으로 다시 | 끝까지 돌고 형식이 맞음 |

---

## 14. Phase S — 요약 `results/w/SUMMARY_W.md` (한국어) 후 멈춤

포함: 계열 안 짝 규모, fold별 선택(칸, r)과 lfV 선택, **R과 CI · 등급**, A-2 표, lfW 대 lfV(개선의 원인), graph 읽기 결과,
숨은 꺾임 민감도 · 위양성률 · PPV를 lf와 나란히(0-1절 모의 기대치와 비교), 채널별 변화, 최종 채택 모델과 읽기 방식,
GPU 사용량, 배포 실행 명세에 넘길 것(`group_id` 열 필수 여부, 묶음 크기 상한).

---

## 15. STOP 조건

| 단계 | 조건 |
|---|---|
| 0 | 1절 시작 조건 하나라도 실패 |
| B | `S_folds.csv`와 짝 표의 a_rxn · b_rxn 불일치, 묶음 키 결측 |
| C | 하나라도 실패 |
| R | PREREG_W가 git에 없거나 변경됨 |
| K | 지문 불일치 캐시, C-W8 실패, 예상 시간 > fold당 20 h, 손실 · 검증값 비유한 |
| E | 세 모델의 평가 계열 · 점 집합이 다름 |
| I2 | I2-0 ~ I2-3 실패 |

---

## 16. 실행 규칙 (저장소 `CLAUDE.md`)

- `sbatch`만, `--time=48:00:00`. 로그인 노드 상주 프로세스 없음. 실행 10 / 대기 20 상한(배열 원소마다 1개), 제출 전
  `squeue -u $USER -h -r | wc -l`.
- 체인(`--dependency=afterany`): B(CPU) → C의 CPU 검사 → C-W9(GPU) → (사용자 PREREG_W 확인) → K 배열 0-4 → collect →
  P 배열(15개, `%10`) → E → D → [W2] → I2 → S.
- 분할: CPU는 `sinfo -p cpu1,cpu2`에서 빈 쪽, GPU는 RTX 3090이 있는 분할 우선(`sinfo -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6`).
- 모든 단계 멱등 · 원자적 쓰기. 48 h에 잘리면 같은 스크립트 재제출.

| 단계 | 환경 |
|---|---|
| B, E, D | `env_d1fu2` (CPU로 실행 가능) 또는 `espley` |
| C-W3 · C-W5 ~ C-W7 | `env_d1fu2` CPU |
| C-W4 · C-W8 · C-W9, K, P, W2, I2 | `env_d1fu2` GPU (torch 2.2.1 + cu118) |
| I2-4 | DEPLOY_MODEL 17절의 단계별 환경 |

**GPU 예산:** K ≈ 5–8 h, P ≈ 0.3 h, W2 ≈ 0.5–1.5 h(r에 따라), C · I2 ≈ 1 h → 합 ≈ 7–11 GPU-h.

---

## 17. 이 명령서의 범위 밖

- 능동학습(배포 후보의 DFT 결과를 다시 학습에 넣기) — 별도 명령서
- 학습 데이터 크기 축소 실험(학습 곡선) — 별도 명령서
- 새 치환기 추가, 새 골격 열거, 배포 실행
