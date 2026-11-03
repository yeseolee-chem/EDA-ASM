# PREREG W — 계열 안 짝 학습(lfW)과 대조(lfV) 사전 등록 (2026-11-04)

이 문서는 DEPLOY-MODEL-W의 사전 등록이다(`docs/specs/DEPLOY_MODEL_W.md` 7절, sha256 `b27f8bb4f7de782e2a13991ad33dd58bbf6d907b6e7cefb48208f6dd0bde474f`). `analysis/deploy/w/prereg_w.py`(deploy-w-prereg-1)가 결과 파일만 읽어 만들었다(git HEAD `84323e1f3f68b8ef678248d9aac297c7614a7ae0`, 작성 2026-11-04 08:19:27). 인용 블록 = 명령서 원문. **K(`train_w.py run`)와 W2(`train_w.py final`)는 이 md와 `prereg_w.json`이 commit되고 바뀌지 않았을 때만 돈다**(`prereg_w.py check`).

## 1. 설계 (0절)
> ## 0. 무엇을, 왜
>
> ### 0-1. 지금까지 확인한 것 (LADDER g2 · S · TF 시험 예측 원자료, 2026-11-03)
>
> | 항목 | 값 |
> |---|---|
> | 숨은 꺾임 계열(장벽 직선 & 채널 ≥ 1 꺾임) 판정, para 89계열 | 민감도 0.65, 위양성률 0.21, PPV 0.57 (기저율 0.29) |
> | 계열 안 예측 오차 중 σ에 대한 직선 성분 (꺾임 판정에 무해) | ≈ 60 % |
> | 나머지 비선형 성분 = **유해 오차** | ≈ 40 %, 신호(DFT 꺾임 크기) / 유해 오차 = 0.9–1.9 |
> | 유해 오차만 축소했을 때(모양 고정 가정) | −30 % → 민감도 0.85, PPV 0.73 · −70 % → PPV 0.81 · −80 % → PPV 0.93 |
> | DFT 구조로 학습한 모델 (상한 참고) | 민감도 0.78, PPV 0.66 |
> | 효과 없음 확인 | seed 증가(유해 분산의 ≈ 12–20 %만 seed 잡음), ENS4(PPV 0.38), HF 추가(0.41), p+m 결합(0.49) |
>
> ### 0-2. 방법 B의 생각
>
> 지금 모델은 **(치환 반응, 기준 반응)** 짝만 배운다. 꺾임은 같은 계열 안에서 **치환기 X1과 X2 사이의 차이**가 σ에 따라
> 어떻게 변하는지로 정해진다. 모델 `g(a, b) = ½[h(a,b) − h(b,a)]`는 어떤 두 반응이든 받으므로, 같은 기준 반응을 공유하는
> 치환 반응끼리 짝 **(X1, X2)**를 학습에 더한다.
>
> - 라벨은 **새 계산 없이 정확히** 정해진다: `dd_t(a1,b) − dd_t(a2,b) = (y_a1 − y_b) − (y_a2 − y_b) = y_a1 − y_a2`
>   (`analysis/d1_pair/d1_pairs.py` 9행: `dd_<t> = y_a − y_b`).
> - 근거: 짝으로 직접 차이를 학습하면 데이터가 조합적으로 늘고 체계적 오차가 상쇄된다고 보고됨 (DeepDelta, Fralish et al.,
>   *J. Cheminform.* 2023, 15, 101, DOI 10.1186/s13321-023-00769-x). 우리 과제에서는 검증되지 않았다 → 이 명령서가 검증한다.
> - 규모(D1 S_folds 기준, 분할 전): 같은 (기준 반응, 계열, 위치) 묶음 353개, 묶음 크기 중앙값 19, 가능한 짝 ≈ 79,600개
>   (지금 학습 짝의 ≈ 11배).
>
> ### 0-3. 비교하는 세 모델 (입력은 모두 lf, S arm, TF)
>
> | 이름 | 학습 짝 | 설정 선택 지표 | 출처 |
> |---|---|---|---|
> | `lf` | 기준 짝만 | 내부 검증 MAE (FU2 규칙) | DEPLOY_MODEL K (이미 있음) |
> | `lfV` (대조) | 기준 짝만 (r = 0) | 내부 검증 **유해 오차 V_nl** | 이 명령서 K (lfW 튜닝의 r = 0 부분) |
> | `lfW` (방법 B) | 기준 짝 + 계열 안 짝 (r ∈ {0, 1, 4} 중 선택) | V_nl (+ MAE 보호 조건) | 이 명령서 K |
>
> `lfW` 대 `lf` = 배포 판정. `lfW` 대 `lfV` = 개선이 계열 안 짝 때문인지, 선택 지표 때문인지 가르는 해석용.
>
> ### 0-4. 읽기 방식 두 가지 (추론만 다름, 학습은 같음)
>
> | 이름 | 반응 i의 예측 | 비고 |
> |---|---|---|
> | **direct** | `g(i, ref)` | 지금 방식 |
> | **graph** | `(1/m) Σ_{j∈G} [g(i, j) − g(ref, j)]`, G = 그 계열의 반응 전부(ref 포함, m = \|G\|) | 모든 경로의 평균. 모델이 완전히 가법적이면 direct와 같다. 2차 지표로 평가 (DeepDelta의 가법성 `g(x,y)+g(y,z)=g(x,z)` 활용) |
>
> ---

## 2. 정의 (3–5절)
> ## 3. Phase B — 계열 안 짝 (`analysis/deploy/w/within.py`)
>
> ### 3-1. 정의
>
> fit 하나의 학습 짝 목록 `tr`(DeployData 행 번호, 사용 불가 짝 제거 후)에 대해:
> - 같은 묶음 키를 가진 학습 짝 i < j마다 계열 안 짝 하나: `ra = D.ra[i]`, `rb = D.ra[j]`, `y = D.y[i] − D.y[j]` (10 타깃 전부).
> - 두 반응이 모두 그 fit의 학습 반응이므로 새 반응은 들어오지 않는다(누출 없음 — C-W2로 확인).
> - 샘플링 가중치: 묶음 g(학습 짝 n_g개, 계열 안 짝 C(n_g, 2)개)의 각 짝에 **w = 2 / (n_g − 1)** → 묶음마다 계열 안 짝의 기대
>   출현 수가 그 묶음의 기준 짝 수에 비례(큰 묶음이 짝 수의 제곱으로 지배하지 않게).
>
> ### 3-2. 산출 (fold k = 0..4, 그리고 `all`)
>
> `$DEPW/within/<split>__f<k>__{train,inner_train}.npz` — `ra, rb, y, w, gid`(정수 묶음 번호) + `meta.json`(묶음 수, 짝 수, sha256).
> `$DEPW/within/test_groups__f<k>.parquet` — 시험 짝의 묶음(평가 · graph 읽기용): `pair_id, gid, b_rxn, fam, pos, Xb, sigma`.
> `results/w/B_report.json` — fold별 기준 짝 수, 계열 안 짝 수, 묶음 크기 분포.
>
> ```
> python within.py build      # 5 fold + all, 멱등
> ```
> CPU, `espley` 또는 `env_d1fu2` 환경. 수 분.
>
> ---
>
> ## 4. 학습 함수 `fit_w` (`analysis/deploy/w/train_w.py`)
>
> `fu2/train.py`의 `fit`을 **복사해서 아래만 바꾼다.** `Std`, `batch`, `orient`, `loss_fn`, `predict`, `gate_m1`,
> `Mo.build`는 import해서 그대로 쓴다.
>
> | # | 바꾸는 것 | 내용 |
> |---|---|---|
> | W-1 | 표준화 | `std = Std(D, tr_base)` — **기준 짝만으로** 토큰 수치열 · 전역 토큰 · `sy` 통계. 계열 안 짝은 표준화에 영향 없음 (r에 따라 `sy`가 바뀌지 않게) |
> | W-2 | epoch 구성 | epoch마다 계열 안 짝 `n_w = r × len(tr_base)`개를 가중치 w로 **비복원** 추출(풀이 작으면 전부) → 기준 짝과 이어 붙여 `epoch_batches`에 넘김 (순서 섞기, (a,b)/(b,a) 뒤집기 그대로). y는 `/ std.sy` |
> | W-3 | 난수 | 계열 안 추출은 **별도 생성기** `rng_w = default_rng(seed + 7919)`. r = 0이면 `rng_w`를 쓰지 않는다 → r = 0 · MAE 선택일 때 `fit`과 같은 난수 흐름 (C-W3) |
> | W-4 | 검증 지표 | epoch마다 내부 검증 짝 예측으로 **V_mae**(FU2 `rec["val"]`과 같은 식)와 **V_nl**(5절)을 모두 기록. 조기 종료 · 최적 epoch 기준 J = `select` 인자 (`"nl"` 또는 `"mae"`) |
> | W-5 | 손실 | 바꾸지 않음: 표준화 평균 \|ŷ − y\| + 0.1 × 장벽 일관성 벌점 (계열 안 짝 라벨도 차이이므로 장벽 = Σ₉ 관계가 그대로 성립) |
> | W-6 | 그 외 | AdamW 1e-3 / 1e-4, cosine T_max 300, dropout 0.1, batch 64, patience 30, 최대 300 epoch, 반대칭 관문 — 모두 FU2 그대로 |
>
> 반환: `fit`과 같은 dict + `best_val_mae`, `best_val_nl`, `history`에 epoch별 `val_mae`, `val_nl`, `n_within`.
>
> ---
>
> ## 5. 선택 지표 V_nl (내부 검증 유해 오차)
>
> 내부 검증 짝(기준 짝, 사용 가능한 것)에서:
> 1. 묶음 키별로 모은다. `pos ∈ {p, m}`이고 짝 수 ≥ `MIN_GROUP_VAL`(6)인 묶음만 쓴다.
> 2. 묶음 g, 타깃 t마다 점 = 짝들의 (σ, e = ŷ − y) [kcal/mol] + 기준점 (0, 0). e를 σ에 대해 최소제곱 직선으로 맞춘 잔차의 RMS = `r_gt`.
> 3. `V_nl = mean_{t∈PRIMARY} [ mean_g r_gt / sy_t ]` (`sy_t` = 그 fit의 `std.sy`).
>
> σ는 **평가에만** 쓴다(모델 입력 아님 — DEPLOY_MODEL C-6 유지). 내부 검증 묶음 수를 `history`에 기록한다(fold마다 대략 20–40개 예상).
>
> ---

## 3. 사전 등록 규칙과 판정 (7절)
> ## 7. Phase R — 사전 등록 (`results/w/PREREG_W.md` + `prereg_w.json`)
>
> K 학습 전에 commit한다. 학습 명령은 두 파일이 git에 있고 변경이 없고 json의 `md_sha256` = md의 sha256일 때만 돈다.
>
> **md에 적을 것**
> 1. 0절 설계, 3–5절 정의, 아래 판정 규칙
> 2. **지금까지 본 것**: 0-1절 표(LADDER g2 분석), DEPLOY_MODEL의 lf 결과(K · E). lfV · lfW 결과는 0개
> 3. 입력 sha256: DEPLOY_MODEL 입력 전부 + `S_folds.csv` + B 산출물(`meta.json`들)
>
> **평가 대상 계열 (고정):** lf의 시험 예측에서 para 계열(`hammett_eval.family_series` 정의, 점 ≥ 12, H 포함). 세 모델 모두
> 같은 사용 가능 규칙이므로 계열 · 점 집합이 같다(다르면 STOP).
>
> **1차 지표 H (시험 유해 오차):**
> `H = mean_{t∈PRIMARY} mean_f [ r_ft / s_t ]`, `r_ft` = 계열 f, 타깃 t의 (σ, e = ŷ − y) + (0, 0) 직선 잔차 RMS,
> `s_t` = 평가 계열 전체의 DFT y의 SD(모델과 무관한 고정 상수). **R = H_lfW / H_lf**, core 부트스트랩(2,000회, seed 20261019) 95 % CI.
>
> **채택 규칙 (둘 다 만족하면 lfW를 배포 모델로 채택)**
>
> | # | 조건 |
> |---|---|
> | A-1 | R의 95 % 상한 < 1.00 (유해 오차가 유의하게 작음) |
> | A-2 | 비열등: 9개 주요 타깃마다 para P_main 상대 MAE 차 (MAE_lfW − MAE_lf) / MAE_lf의 95 % 상한 ≤ +5 % (DEPLOY_MODEL G-C와 같은 기준) |
>
> 등급: R ≤ 0.70 = "목표 달성"(0-1절 모의에서 PPV ≈ 0.73에 해당), 0.70 < R이고 A-1 통과 = "부분 개선", A-1 실패 = "개선 없음".
>
> **2차 (graph 읽기, 추론 전용):** 채택된 모델(lfW 또는 lf)에서 R_g = H_graph / H_direct의 95 % 상한 < 1.00이고 A-2와 같은
> 비열등(graph 대 direct)을 만족하면 배포 추론을 graph로 한다. 아니면 direct.
>
> **기술 지표 (판정에 쓰지 않음, 모두 보고):** 숨은 꺾임 민감도 · 위양성률 · PPV(+ 계열 부트스트랩 CI, 짝지은 차이 CI), 채널별
> 꺾임 민감도 · 위양성률, 연속 g AUC, 타깃별 H, lfW 대 lfV의 R, 0-1절 모의표와 비교한 기대 PPV.
>
> ---

## 4. 구현 결정 (코드 작성 시, 결과를 보기 전; 사용자 지시: 모든 분기 = 추천안)

1. B: group key (b_rxn, fam, pos) from S_folds.csv over every pair (ref_kind reference and anchor); S_folds itself has 610 groups (579 with >= 2 pairs), median size 17, 67,636 within pairs - not the spec 0-2 estimate (353 / 19 / 79,600), which no key on S_folds reproduces; recorded in results/w/B_report.json, no STOP rule applies.
2. W-2 sampling: 'w = 2/(n_g-1), without replacement' is realised as inclusion probability pi_i = min(1, c w_i) with sum pi = n_w (systematic sampling on a random order): every uncapped group gets exactly c n_g expected pairs per epoch; groups whose pool is exhausted are drawn whole ('all when the pool is small'). numpy's weighted choice(replace=False) misses the n_g-proportionality by 19-26 % at these sampling fractions (C-W6).
3. C-W1: labels_d1.json fails the within-group identity for d1 / d2 / barrier where the LADDER cr2 correction changed an a-side reaction; with labels_d1_cr2.json (V_report all_pass) every group passes (<= 1e-6) - the spec fallback.
4. C-W3: fu2 fit computes in float32 (train.py 155, 226-227), so 'CPU float64' is not possible without editing it; both fits run on CPU in float32 with 1 thread (multi-threaded CPU reductions are not bitwise reproducible) and must agree bit for bit (history, best epoch, validation, weights).
5. V_nl / H points: pairs with a finite sigma only; H point (0, 0) appended; RMS over n + 1 points; sigma = S_folds sigma (sigma_p for p, sigma_m for m). V_nl groups: inner_val pairs, pos p / m, >= 6 such pairs.
6. H: s_t = SD (ddof 0) of the DFT y over every evaluation point (para families, H points excluded); core bootstrap = eval_geo2.boot_diff resampling (default_rng(20261019), 2,000), family means weighted by core counts.
7. grade: A-1 fails -> 'no improvement' (also when the point R <= 0.70); else R <= 0.70 -> 'goal reached', else 'partial improvement'. Secondary graph readout for the adopted model only (lf and lfW both reported).
8. W2 epoch = lower median of the best epochs of the folds whose lfW r is the mode (spec 12); C-W8 for K = 200 random test pairs + 200 random within pairs of the fit (r > 0); W2 = 200 training + 200 within pairs.
9. lfV = lfW selection (same cell, r = 0) -> one set of final fits; the lfV .pt files are hard links of the lfW ones.
10. E: the Phase P (a, ref) predictions must equal the K predictions (<= 1e-3 kcal/mol, else STOP: wrong model files).

## 5. 지금까지 본 것
- 0-1절 표(LADDER g2 · S · TF 분석; 위 1절 인용 그대로)
- DEPLOY_MODEL lf 결과 (K · E): ```{"K": {"D__lf__S__TF__f0": {"chosen": {"cell": {"n_layers": 3, "width": 64}, "best_epoch": 77, "index": 2, "grid_index": 2}, "n_test": 1590}, "D__lf__S__TF__f1": {"chosen": {"cell": {"n_layers": 3, "width": 64}, "best_epoch": 46, "index": 2, "grid_index": 2}, "n_test": 1460}, "D__lf__S__TF__f2": {"chosen": {"cell": {"n_layers": 2, "width": 128}, "best_epoch": 92, "index": 1, "grid_index": 1}, "n_test": 1346}, "D__lf__S__TF__f3": {"chosen": {"cell": {"n_layers": 3, "width": 64}, "best_epoch": 41, "index": 2, "grid_index": 2}, "n_test": 1300}, "D__lf__S__TF__f4": {"chosen": {"cell": {"n_layers": 2, "width": 128}, "best_epoch": 17, "index": 1, "grid_index": 1}, "n_test": 1583}}, "K_sha256": "17bc5ea2faf15f73a34f0cc2a2b0ae8292273ecc2bb274bea297f23a6b4ce9d1", "E": {"G_C": false, "para_mae_own": {"d1": 0.37098913025077096, "d2": 0.30188742148175257, "elst": 0.8896401434721166, "pauli": 1.7411314758405272, "oi": 1.0199895408124122, "disp": 0.13337053593112816, "cpcm": 0.315073386416521, "cds": 0.028319847998803247, "c_ghost": 0.041488252529403476, "barrier": 0.33016706615180136}, "auc_g_own": {"barrier": 0.8264738598442715, "d1": 0.8915254237288136, "d2": 0.7980952380952381, "elst": 0.7323943661971831, "pauli": 0.7464985994397759, "oi": 0.8155467720685112, "disp": 0.5676470588235294, "cpcm": 0.7514492753623189, "cds": 0.549367088607595}, "hidden_own": {"n_fam": 89, "tp": 14, "fp": 17, "fn": 12, "tn": 46, "base": 0.29213483146067415, "sens": 0.5384615384615384, "fpr": 0.2698412698412698, "ppv": 0.45161290322580644}, "n_para_families": 89}, "E_sha256": "2aa9a49de1f730dbbf48c669003ca62d37ed5eb07876ce6fd7a0e0baad334532"}```
- lfV · lfW 결과: **0개** ({"units": [], "results": [], "within_pred": [], "model": []}; 2026-11-04 08:19:27)
- Phase B 규모: {"0": {"train": {"n_base": 5756, "n_groups": 419, "n_groups_ge2": 396, "n_within": 45259, "group_size_ge2": {"n": 396, "min": 2.0, "q25": 11.0, "median": 17.0, "q75": 19.0, "max": 20.0, "mean": 14.477272727272727}, "within_over_base": 7.862925642807506}, "inner_train": {"n_base": 4312, "n_groups": 317, "n_groups_ge2": 295, "n_within": 33972, "group_size_ge2": {"n": 295, "min": 2.0, "q25": 12.0, "median": 17.0, "q75": 19.0, "max": 20.0, "mean": 14.542372881355933}, "within_over_base": 7.87847866419295}, "test": {"n_pairs": 1590, "n_groups": 133, "group_size": {"n": 133, "min": 1.0, "q25": 7.0, "median": 14.0, "q75": 18.0, "max": 20.0, "mean": 11.954887218045112}, "n_graph_pairs": 12981}}, "1": {"train": {"n_base": 5912, "n_groups": 464, "n_groups_ge2": 436, "n_within": 44776, "group_size_ge2": {"n": 436, "min": 2.0, "q25": 9.0, "median": 16.0, "q75": 19.0, "max": 20.0, "mean": 13.495412844036696}, "within_over_base": 7.573748308525034}, "inner_train": {"n_base": 4419, "n_groups": 341, "n_groups_ge2": 321, "n_within": 33807, "group_size_ge2": {"n": 321, "min": 2.0, "q25": 9.0, "median": 16.0, "q75": 19.0, "max": 20.0, "mean": 13.70404984423676}, "within_over_base": 7.6503733876442634}, "test": {"n_pairs": 1460, "n_groups": 88, "group_size": {"n": 88, "min": 1.0, "q25": 15.0, "median": 18.0, "q75": 19.0, "max": 20.0, "mean": 16.59090909090909}, "n_graph_pairs": 13706}}, "2": {"train": {"n_base": 5989, "n_groups": 455, "n_groups_ge2": 430, "n_within": 45774, "group_size_ge2": {"n": 430, "min": 2.0, "q25": 9.0, "median": 16.0, "q75": 19.0, "max": 20.0, "mean": 13.869767441860466}, "within_over_base": 7.643012189013191}, "inner_train": {"n_base": 4440, "n_groups": 333, "n_groups_ge2": 315, "n_within": 34267, "group_size_ge2": {"n": 315, "min": 2.0, "q25": 10.0, "median": 16.0, "q75": 19.0, "max": 20.0, "mean": 14.038095238095238}, "within_over_base": 7.7177927927927925}, "test": {"n_pairs": 1346, "n_groups": 97, "group_size": {"n": 97, "min": 1.0, "q25": 9.0, "median": 17.0, "q75": 19.0, "max": 20.0, "mean": 13.876288659793815}, "n_graph_pairs": 12096}}, "3": {"train": {"n_base": 6035, "n_groups": 464, "n_groups_ge2": 437, "n_within": 45988, "group_size_ge2": {"n": 437, "min": 2.0, "q25": 9.0, "median": 16.0, "q75": 19.0, "max": 20.0, "mean": 13.748283752860411}, "within_over_base": 7.620215410107705}, "inner_train": {"n_base": 4460, "n_groups": 340, "n_groups_ge2": 318, "n_within": 34185, "group_size_ge2": {"n": 318, "min": 2.0, "q25": 10.0, "median": 16.0, "q75": 19.0, "max": 20.0, "mean": 13.955974842767295}, "within_over_base": 7.664798206278027}, "test": {"n_pairs": 1300, "n_groups": 88, "group_size": {"n": 88, "min": 1.0, "q25": 12.0, "median": 17.0, "q75": 20.0, "max": 20.0, "mean": 14.772727272727273}, "n_graph_pairs": 11836}}, "4": {"train": {"n_base": 5753, "n_groups": 424, "n_groups_ge2": 399, "n_within": 44608, "group_size_ge2": {"n": 399, "min": 2.0, "q25": 11.0, "median": 17.0, "q75": 19.0, "max": 20.0, "mean": 14.355889724310776}, "within_over_base": 7.753867547366592}, "inner_train": {"n_base": 4350, "n_groups": 324, "n_groups_ge2": 303, "n_within": 33742, "group_size_ge2": {"n": 303, "min": 2.0, "q25": 10.0, "median": 17.0, "q75": 19.0, "max": 20.0, "mean": 14.287128712871286}, "within_over_base": 7.756781609195403}, "test": {"n_pairs": 1583, "n_groups": 128, "group_size": {"n": 128, "min": 1.0, "q25": 5.75, "median": 15.0, "q75": 19.0, "max": 20.0, "mean": 12.3671875}, "n_graph_pairs": 13507}}, "all": {"train": {"n_base": 7334, "n_groups": 552, "n_groups_ge2": 521, "n_within": 56507, "group_size_ge2": {"n": 521, "min": 2.0, "q25": 10.0, "median": 16.0, "q75": 19.0, "max": 20.0, "mean": 14.017274472168905}, "within_over_base": 7.70479956367603}}}
- 시작 조건(1절): {"0-1": true, "0-2": true, "0-3": true, "0-4": true, "0-5": true}

## 6. 입력 sha256

| 파일 | sha256 |
|---|---|
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/labels_lf.json` | `e7fdf9b57371f7c8e2851cd7fef4ef131604b6257047bd5e75f55ee326b5f720` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/input_meta_lf.csv` | `104ab323de72fb7e4a999744e5957914e3d4dfedefbeddafa8c1bf4941451a09` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/X182_lf.parquet` | `8917a728c231a48baff5bafe697dc88f745e9c0fc8d807690805c47dbf42bd18` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/fu2_lf/T_manifest.csv` | `72e0523c2ac14f2312431f86109191858de314eed875f74486d6eefeb5db2b92` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/fu2_lf/T_report.json` | `7d36292c0078f5fb32d94e89894ef79ff4e7258c52de58b835066d4520d183d5` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/fu2_lf/Q_report.json` | `d9d657fbbc781d78cf3fcf504a0fc14092cd61faeb999217189909eb9eb680c0` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/fu2_lf/Q_manifest.csv` | `b886ca126df69b8c8e905ad7d61dd11d2600a2288922c9c7c5b175b85606b4ad` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/F_pairs_cr2.csv` | `f2fe5cab1c61f082b7cf929ca1146d96f8e451d37ed7d4ee2b5b127b4a7fa6ce` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/d1_pair/results/fu2/S_splits.json` | `eefe495a775babf2604be6f21156e4ed2869a1988bd08d1542c0d7d4b7779413` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/d1_pair/results/S_folds.csv` | `944a1d1439eb0d54dff90f58f40476c21109cf3a5b71b4bb0f755651b216dcb7` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/d1_ladder/results/L_predictions.parquet` | `1de11e77ea82571b139d6070ad556386e2a74b482af5c357498614bee37b3e4d` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/d1_ladder/results/V_report.json` | `d4b2c0dfd340cb81596352dd6ca74452c1cce4a1d8601e05170abe1097a46f1f` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f0.json` | `8613360436df1cf55f4929c0e57d91e2b74a7a3579489315b0548c7f22f54521` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f1.json` | `28c40b0c4c2b8fa3f12bc753cf3976b5851e16327730527a26b3392dd3c8dc4c` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f2.json` | `0f1939c9e3b8194785205dacbc10eec725cdcd93ae89b2e784f726dac334735e` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f3.json` | `992a7ba21d4ac57948d78b83f5fc90efdcc82ea03a9667b548a8ea5c1665ff27` |
| `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f4.json` | `168b876bd111a3e7a078b493db50a75b4db7ae09b6cd0e6eb51eb32b64bafec6` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/prereg_deploy.json` | `9c747241f1f835bcae799acfdc72565f768b27cc288c782372b1953eec6624e3` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/PREREG_DEPLOY.md` | `1f576e5461dbaf51d928091cb55aa7ade9c259fa3550f77b266fa40cb1093d2e` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/K_predictions.parquet` | `a7c30339e46b5455972ba4bb93493d3f50f33db4b77d8bc6f3a83a5b53f58616` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/E_report.json` | `2aa9a49de1f730dbbf48c669003ca62d37ed5eb07876ce6fd7a0e0baad334532` |
| `/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-deploy/analysis/deploy/results/K_report.json` | `17bc5ea2faf15f73a34f0cc2a2b0ae8292273ecc2bb274bea297f23a6b4ce9d1` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/T1__f0__inner_train.npz` | `edcf572e0f0df414f255513a30303acf053456e78bb4b4e4e8e8dd60106928d9` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/T1__f0__train.npz` | `96c0a372f2291298ef050ad36849dff7189573075884aabe0e81404547d0fcba` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/T1__f1__inner_train.npz` | `24b2dbdc57f62d0724922151bc3eb0b87eea6aef74ae76fae0e9535bec263b9e` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/T1__f1__train.npz` | `78997b68b35a958876df6fabea6a96c365f1481cca847c0c4cc3bb03b0839a83` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/T1__f2__inner_train.npz` | `b81df4cd4f951e4f42a5d2eef395036b154ee843112f8b37f764c71b3ca56087` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/T1__f2__train.npz` | `c05e79b73ea28f380803d6db7a73f4ce059d60ac859944ccbda9961aa429c000` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/T1__f3__inner_train.npz` | `a974da9793c36f5f1f4a6342ace210d9e143ce8e9a1d94fa5cd905bcca65fd55` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/T1__f3__train.npz` | `e5176474b2a523a63aeaaebfe856678d4ce28c5a12b401e448c402df5bad4792` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/T1__f4__inner_train.npz` | `d871c571aef5ae0a0132535448478ddc96eca8d44f5b1a6d21484cfa0fffa64c` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/T1__f4__train.npz` | `e04f617d7ea44bcc10921c60916028b2ef48a12512d0fdad1493c5602330b7c1` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/all__train.npz` | `45ec61326a0e6586463c1047bda0baec2d04b9c3da51652756b828e2f525791d` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/test_groups__f0.parquet` | `b02614bd8002d56e966a69d6f208565057ca77ef012c536a55b7b9dfff13a859` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/test_groups__f1.parquet` | `93f46a708e5d4eb20f070a7c279dfbda468c7b4254b7ff0bab777039dfb42047` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/test_groups__f2.parquet` | `9277291135957ce49588a58ed560574720ad4cf004ebc7a9be4a592a9714320b` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/test_groups__f3.parquet` | `7d4cd37589971c966a276ce6fbb95a533e41f22d95bbb2e6d5a2f04cae447ffe` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/test_groups__f4.parquet` | `4e45994e974fc1f2e9cbe91b75e1be9271ae6c05d9915c3d5caa61eef8c04cae` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/w/within/meta.json` | `7acc856ee7617f4bb72b6fe13fb8bdd0eba55a6fd816e275a7cbb64c54664c5a` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f0.json` | `b5b8239dd8014af14da2daeb2c426b43e00c8df01be6284439d227bd4de64528` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f0.parquet` | `fa760bff135801f1688f26079db36423ca733186003b89a20b4b4f628beeac79` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f0__s0.pt` | `5f7424a8bfa8c31a94b1aa2ff7e1c41f02c0eed961d03fa605c83405e9402fca` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f0__s1.pt` | `f0a9e99ec5498343c3c1ad82e87f0ff30c4f66dcbc2e99cf47ad61fa5c326ada` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f0__s2.pt` | `196f4a36a479ec4ce47e20ebf400edabe9afabc61343b0fc7dea51246d5ac444` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f0__s3.pt` | `91a2acd94001877a0ec93b430eafe475708d0945ca59403b4c75d462afef8c64` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f0__s4.pt` | `3796621a967d30036013adfc97b4d2f90784a8a1d588542aa6ece067e7f84f48` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f1.json` | `b3a7bf5ba7337a39d0f64b752978916405a8b09c0f3ef02981e02e7661d67000` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f1.parquet` | `e799c5cc1d371e601153b910ce740a6e891a2962f5912f23e556bf2d790e3f6b` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f1__s0.pt` | `87c8875ebed256802f9a998bd1b7f021b1aed88efd0f5288f4112eeea24e2ef7` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f1__s1.pt` | `d3ce0937ddb0c58c39a89a8e50b0a7bece4f04b1c0e825caf9d06ed8ecef6c7e` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f1__s2.pt` | `9e9021d5a0b1c0740413d7085d411b885ab110994dcd1969898168b6b72cce19` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f1__s3.pt` | `9426cf8faf8452624f6edccdc495c585a3247b861bbcc67ec739994f5b714a01` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f1__s4.pt` | `e14516dcfdddebd295525fee8a35a6f0637869f8b34632beede483858db87b50` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f2.json` | `368d78256ce2ad331dae8f41be365149dab2ee2c2fb4e6c27b01e582bdd79bf0` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f2.parquet` | `7450580fbc32f8847b4f8000dc050d6853341c2d4603473311ca105676ac273c` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f2__s0.pt` | `bb5df7b41bbacd6ae91860a3c4177f5b0d5fb2c9d848972930b54e992d57fa2e` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f2__s1.pt` | `d3d60372d79005cf7da20366c11151679f59aa63a66111f2f9ce8da77c302e7c` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f2__s2.pt` | `2b883563d9ee17cadc178dee3d2b65c323adb65091d0598002b5141c1f4203d2` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f2__s3.pt` | `479ad29c6c7079676e0c1d03a6597ada3b8fac0f8f20c059a588a1d23aed6ce3` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f2__s4.pt` | `286d773d48bb8abe93b5ee195bfc48738a7d99ee8051d2e9a4a47f6b74c4a219` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f3.json` | `58c8aa0c84336516d6ba9eb170215b33542149d12401321442c82b5de59fb333` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f3.parquet` | `98fabb3f7ddcdabea49d98809ec9534312ba142a20c94814b3384d042a3a1361` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f3__s0.pt` | `e6f657b918089953f38411e9c6498cd770abdfec7418651e6e00649e54474f52` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f3__s1.pt` | `81e734bf9d4980dc9c25eff4f6094aa1b498c3898ee515a8846c5016a9f2ccf7` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f3__s2.pt` | `a9397923ef684c338143b82260c96ae989598a55871de9b06022617c991e11c4` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f3__s3.pt` | `41c3a3b01192474e8016debfc4cf00d265dad5f69b44a9ba98bb60fcf2e46240` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f3__s4.pt` | `941faf34d13d1be58ba15416496543ce8c9b8b4d1a24cfb3f7220eaf84b5f940` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f4.json` | `02c5fa354ef9508a449a02e3c581ab7aa45af425af75146c00adac9be38e1c0a` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f4.parquet` | `738da5fbf97e8dc66b517de143f1b0c40cfb88a76761642e39cb30f1f5ce7b5c` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f4__s0.pt` | `2f7a1523ba95b0a20ef9781db82e82ac45e4492b8adaa5462191d93acaeaa221` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f4__s1.pt` | `0ff0e9078f959ea033dacb5e5cecf03bcd13c90235a1f3cb6f73defd2c753c91` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f4__s2.pt` | `6b30316521f01aea1f00e32cb5361b2beb0375727f37620cdf044f9e5d0211a1` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f4__s3.pt` | `fca112e27c803c4bb546e966a855cf1e2ca2ace4f6f0309ae8b9a9837b8311cd` |
| `/gpfs/tmp_cpu2/yeseo1ee/deploy/units/D__lf__S__TF__f4__s4.pt` | `5ba5dc3a7f25b08cf80d04425bec7f77b3ae85d5e38883c641c35e06cb36f818` |

- C 검사: `results/w/C_tests_w.json` sha256 `75dbd7b90c9cf3a96a3c4993e8aa823cccc98365e8527e03360cebade397b1ad` (all_pass)

## 7. 코드 sha256 (작성 시)

| 파일 | sha256 |
|---|---|
| `analysis/deploy/w/w_common.py` | `bc29a2eeab58307e9fc04a038854a0b3fd85bdfbe278f7927cb5ec60d352e685` |
| `analysis/deploy/w/within.py` | `46989cfd945b5f88bdd919216295c7193b6913f188fe8d17606d902c44f59f82` |
| `analysis/deploy/w/train_w.py` | `5159f75336771fa0f390b187e3a5d3e38daff94ab20e43cb68e168259909d58c` |
| `analysis/deploy/w/within_predict.py` | `44455f2619e8d79abfae370ceb0b6ae7f32095a68fd657fb80b1c1642a3d0353` |
| `analysis/deploy/w/eval_w.py` | `315357772614741c6c1c54495ee9c0f2b29fd74eed707e600707f7459d9e3240` |
| `analysis/deploy/w/prereg_w.py` | `a031d9e881474c8b8fb0879a110d5f9df55f3403f9e1d0f256b0aa202e75dcfb` |
| `analysis/deploy/w/tests_w.py` | `a9f52002e2caf854012437a9dfee969643cabfed077fecf7799586915b07635d` |
| `analysis/deploy/c1_dft_free.py` | `d4aacd21135c867e7213897a7a05e494a42f684ca45b0f76dbeac955b1c7c152` |
| `analysis/deploy/check_start.py` | `647a1d829bef62c35308a054d17f783397cb9a9a464c51b9580e265ec6ee55ca` |
| `analysis/deploy/data_lf.py` | `2ffe3313898015a6ae749a5348ab774354722fccbef7c5454cf1e3c04e417eb8` |
| `analysis/deploy/dep_common.py` | `27d77566871d0e838345e92067d689c47d8459a9087c42cb79aafb5bc87f5f8a` |
| `analysis/deploy/eval_tf.py` | `8b1fe4e06e94e7ad094cf066ec45627823e3b6a772110ce20ba0d5fd3b90f9c3` |
| `analysis/deploy/feat_lf.py` | `28be59794a476a362c80babe92cd60c6457138b70ca1a58a6d3f4051e7c00a2a` |
| `analysis/deploy/final_dev.py` | `a9f15ec627aeae59755445a878db7b1625bb6c8c7652803caa5a687baa6e18f6` |
| `analysis/deploy/finalize_lf.py` | `7bf79b8d5670e255146a2834e76c5405b74cc9ee430a1859e4b29f38d2240ead` |
| `analysis/deploy/g2_deploy.py` | `cc0cc09f2f427ec464c944b6a0a252fb6546d40236da6d96cda091929bdc069a` |
| `analysis/deploy/gc_deviation.py` | `ba00ac0f28f5005997907000e9f316542a43b4fbca261010e37da6c7902b61c6` |
| `analysis/deploy/meta_lf.py` | `cb48851329f0855e5bcc45574ffe19eccb73dc3e0c01faeb3a2622148cd80bc9` |
| `analysis/deploy/null_gc.py` | `32651b352769a211f6ed747657e923f54daddbb54c08955fa055e89e366ea3ff` |
| `analysis/deploy/parity.py` | `af23a7836dcd2877331a37b56009f94f5ef4bb0b0920fdbc1790dd7f41ff0948` |
| `analysis/deploy/predict_tf.py` | `d5332129bc0bb19e2551ecc5de4945cc366531bb021ef5f9220ba4974e37e591` |
| `analysis/deploy/prereg_deploy.py` | `63fbc1e0bd4b5941bb7739def642611d36b3bde0b3d147759297b4543184c3dc` |
| `analysis/deploy/tests.py` | `85a0e448bbd5d54f809d953ce0587d7c7e1e52aef841f2cf28b707d2c9c9739e` |
| `analysis/deploy/tokens_lf.py` | `a24995927746c4c9fe7dd26ce78aa3aadf0269e9a4cc4f33105bec1354111815` |
| `analysis/deploy/train_tf.py` | `2755fafbd94332c16008f9ca93bb021f7f9b1af520081633ba0158f6e65624b7` |
