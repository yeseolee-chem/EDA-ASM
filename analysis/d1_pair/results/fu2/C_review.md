# FU2 Phase C — 코드 검토 기록

사양 `docs/specs/D1PAIR_FU2.md` Phase C: 사용자 검토에서 고친 점을 여기에 적는다. 아래 1절은 사용자 검토 **전**에 한 자체 검토다.
사용자 검토 결과는 2절에 이어 적는다.

## 1. 자체 검토 (2026-10-21, 사용자 검토 전)

방법: 기존 코드 지도(xtb_slice, ext_features, fragmenter, d1 파이프라인; 읽기 전용 reader 4개) → FU2 코드에 대한 적대적 검토
(관점 4개: 사양 일치, 계산 버그, HPC·캐시 안전, 검사 타당성; 관점마다 반박 검증자 1명). 검토자는 코드를 실행하지 않았다.
아래 수정은 모두 Q 전체 실행, C 검사 실행 **전**에 했다(Q smoke 24 반응은 수정 전후 두 번 돌렸고 둘 다 통과).

### 1-1. 사양의 빈틈 (추천 옵션으로 처리, 보고)

- **H 원자 대응.** 사양 Q-1은 TS 원자 i의 반응물 원자를 `inv0[i]` / `inv1[i]`(P.map0 / P.map1의 역)로 정하지만, 이 대응은 heavy 원자만
  담는다(fragmenter 부분그래프 대응의 노드 = heavy 원자, 표시 = (원소, H 수)). 처리:
  - H는 자기 heavy 원자(가장 가까운 heavy 원자; TS에서는 같은 조각 안에서)를 거쳐 대응한다.
  - q_rel / v_rel(H) = 대응된 반응물 heavy 원자에 붙은 H들의 평균(같은 heavy 원자의 H는 대응으로 구별되지 않으므로, 임의 선택에
    의존하지 않는 값). 조각별 합은 반응물 합과 같다(Q smoke 24 반응에서 차 ≤ 3e-16; C11이 모든 반응에서 ≤ 1e-9로 검사).
  - 공유결합 간선(T-2)은 heavy 원자별 H를 번호순으로 일대일 대응해 옮긴다. 반응물의 모든 H가 정확히 하나의 결합(자기 heavy 원자)만
    가질 때 간선 집합은 그 순서와 무관하다 → atoms.py가 이를 확인하고 아니면 그 반응을 `h_map:` 상태로 뺀다.
  - heavy 원자별 H 수가 TS와 반응물에서 다르면 `h_map:` 상태로 뺀다(보고).
- **C12의 batch 크기·dropout.** 사양은 "학습 짝 64개로 300 epoch"만 정한다. 판정 실행은 dropout 0, batch 8(300 epoch × 8 step)로 정했다
  (학습 가능성 점검). M-1 설정(dropout 0.1, batch 64 = epoch당 1 step) 실행은 참고값으로 같이 기록한다.
- **C9 기준의 읽기.** D1 라벨 자체가 항등식 barrier = 9개 합을 정확히 만족하지 않는다(ORCA 표 마감 오차; F_pairs.csv 짝 차이 기준
  |잔차| 평균 0.0054, 최대 0.096 kcal/mol; `P_report.json` P2_identity_resid와 일치). 그래서 정답 y의 벌점은 0이 아니다. 판정은 사양 문구
  그대로 **손실 안의 벌점 항**(λ · 평균 |ŷ_barrier − Σ9 ŷ| / s_barrier, λ = 0.1) ≤ 1e-3로 하고, 구현·단위 확인을 위해 (1) 항등식을 맞춘 y의
  벌점 ≤ 1e-12, (2) barrier에 0.7(표준화) 더하면 벌점 0.7, elst에 δ 더하면 δ·s_elst/s_barrier, (3) 무작위 ŷ에서 numpy 독립 계산과 1e-12
  일치, (4) 손실 = L1 + 0.1·벌점을 같이 판정한다. 라벨 잔차는 참고값으로 `C_tests.json`에 적는다.

### 1-2. 고친 버그 (실행 전)

| # | 파일 | 내용 | 수정 |
|---|---|---|---|
| B1 | splits.py | pair_id(`P00000`, `A00000`)를 int로 바꿔 내보내기가 멈춤(job 1029738 실패, 결과 없음) | 문자열 그대로 |
| B2 | atoms.py gate | 확장 값을 쓸 때 확장 검사 실패 반응이 'use'에 남아 NaN이 토큰으로 감 → T-a STOP | use = Q-a ∧ Q-b ∧ (ext_ok ∨ 확장 미사용); 통과율 분모 = Q-a·Q-b 통과 반응 |
| B3 | atoms.py | 확장 계산의 BlockFail 외 예외가 반응 전체를 실패시킴 | 확장 값만 잃음(이유 기록) |
| B4 | atoms.py | 코드 버그·시간 초과 등도 `.fail.json`으로 캐시되어 재실행 때 건너뜀(캐시 오염) | `ts_fragments:` / `h_map:`만 캐시; 다른 예외는 `.err.json`(재시도); ATOM_VERSION이 다르면 다시 계산; xtb 시간 초과·signal은 한 번 재시도 |
| B5 | atoms.py | pool에서 OSError가 나도 남은 반응을 다 돈 뒤 멈춤 | 즉시 남은 작업 취소, exit 3 |
| B6 | atoms.py | Q-a가 원시 배열만 확인 | 토큰이 쓰는 TS 순서 배열(q/v_rel, q/v_frag, q/v_TS)로도 30열을 다시 만들어 확인 |
| B7 | tests C10 | 원자 이동 식의 연산자 우선순위 버그(`+0.5`가 좌표 전체에 더해짐 → 5 Å 안으로 들어올 수 있음) | 바깥 방향 이동만; 이동 후 5 Å 밖 assert; 양성 대조(4 Å 안 원자 0.2 Å 이동 → ε_i 변화) 추가 |
| B8 | tests C9 | 위 1-1 | 위 1-1 |
| B9 | tests C8 | 다른 노드에서 GroupKFold를 다시 계산해 비교 → CPU 정렬 커널에 따라 동점 순서가 달라 거짓 실패 가능; core 분리를 직접 확인하지 않음 | 판정 = 검증·안쪽 학습 core 분리, 합 = 학습 쪽, 시험 짝과 분리, 비율 0.15–0.35; 재계산은 참고값 |
| B10 | tests C6 | 학습 루프의 y·부호 처리 경로를 검사하지 않음 | batch 구성을 `train.epoch_batches`로 분리(fit이 그대로 사용), C6가 그것으로 F_pairs dd 부호를 확인 |
| B11 | tests C1–C5 | 출력이 0이어도 통과하는 검사 | 가중치에 잡음(σ 0.3) 추가, 중앙값 \|g\| ≥ 1e-2 kcal 확인; 음성 대조(x만 섞기, 좌표 1.05배) |
| B12 | tests C4 | 손으로 0을 채운 padding만 확인 | 실제 batch 함수에 가장 큰 반응을 섞어 확인 + 무작위 값으로 채운 padding |
| B13 | tests C7 | Std를 검사 안에서 다시 만들어 확인(동어반복) | run_unit 안의 실제 fit이 쓴 통계 행 = 의도한 행, μ·σ 재계산 일치, 모델에 들어간 x 값 확인 |
| B14 | tests C11 | 간선 수를 같은 코드가 만든 수와 비교 | rel1/rel2 결합을 공유 반지름으로 독립 재계산; 간선 → 반응물 결합 확인; H마다 간선 하나 = 자기 heavy 원자 |
| B15 | train.py | GPU가 없으면 CPU 1 thread로 조용히 학습 | `require_gpu()`(CUDA, torch 2.2.1+cu118, env_d1fu2) — smoke·검사·학습 모두 |
| B16 | train.py | gate M-1′을 튜닝 모델은 안쪽 검증 짝에서 확인 | 사양대로 모든 모델을 시험 짝 200개에서(입력만 쓰고 라벨은 쓰지 않음) |
| B17 | train.py | ADD2 json에 입력 sha256이 없어도 gate 통과 | Q·T·분할 파일 6개의 sha256을 필수로 |
| B18 | train.py | `M_predictions.parquet`를 쓰는 곳이 없음 | `train.py collect` |
| B19 | train.py | smoke가 가장 작은 grid 칸만 실행(메모리·시간 상한 미확인) | 가장 작은 칸과 가장 큰 칸 모두 |
| B20 | 모든 FU2 쓰기 | 쓰기 경로 보호가 rev 4–6 위치만 막음 | `fu2_common.assert_path`: `$D1PAIR_SCRATCH/fu2`, `results/fu2` 밖이면 exit 3(M/units 보호) |
| B21 | env_build.slurm | 버전 확인 전에 stamp를 찍음 | 확인 통과 뒤 stamp(이번 빌드는 확인 통과) |

### 1-3. 운영 결정

- **GPU 기종 고정.** 사양 M-3은 캐시 지문에 GPU 이름을 넣고 다른 지문이면 exit 3이다. gpu3은 RTX A6000 Ada, gpu4·gpu5는 RTX A6000이라
  섞어 쓰면 다시 제출한 slice가 멈춘다 → Phase M 학습 job은 모두 `-p gpu4,gpu5 --gres=gpu:a6000:1`(48 GB). 지문 규칙은 사양 그대로 둔다.
  Phase C 검사·smoke는 GPU 기종과 무관(float64 성질 검사; GPU 이름은 기록)하다. gpu4/gpu5의 예상 시작이 10시간 뒤(우선순위 대기)여서
  바로 시작할 수 있는 gpu2/gpu6(NVIDIA A10, 24 GB)에서 돌렸다. smoke의 시간 추정은 A10 기준이라 A6000보다 보수적이다.
- **SVR helper 취소.** SVR job은 FU2 동안 건드리지 않는 것이 원칙이나, 사용자 승인(2026-10-21)으로 같은 unit을 겹쳐 계산하던 helper
  1029020_2–3만 취소했다(`jobs/submit_M.sh` 기록; 끝난 unit 손실 없음).

## 2. 사용자 검토 (2026-10-21, commit d1e4e3d8 대상; `REVIEW_FU2_CODE_d1e4e3d8.md`)

결론: 모델·학습 코드는 명령서와 일치, 막는 버그·누출 없음. 원자 대응(대칭 원자) 하나를 고치고 F1–F5를 반영한 뒤 다시 멈출 것.

### 2-1. 반영 (Phase A 전)

| # | 지적 | 조치 |
|---|---|---|
| F1 | 반응물의 모양이 같은 원자(페닐 ortho C, CF₃ F, NO₂ O, tBu·SiMe₃ C 등)는 VF2 첫 대응을 그대로 써서 q_rel·v_rel·Δq_dist·Δv_dist가 반대편 원자 값일 수 있다 | `atoms.sym_remap`: 반응물 heavy 그래프(fragmenter.heavy_graph 표시 (원소, H 수))의 automorphism 중 반응 원자(A: r1_dip 3개, B: r2_dph 2개)를 고정한 것을 모두 나열(반응 원자에 고유 표시를 붙여 GraphMatcher로), 저장된 대응에 합성, TS 조각 heavy 원자와 Kabsch RMSD가 가장 작은 대응을 씀(1e-6 Å 안의 동점은 저장된 대응 유지). H = 바뀐 heavy 원자의 owner H 평균, 간선 bijection도 같은 규칙. automorphism > 10,000이면 STOP(`AutoLimit`). 저장된 fu2-atoms-1 npz에서 xTB 없이 변환(`atoms.py remap`, `ATOM_VERSION = fu2-atoms-2`); 새 계산(`compute`)도 같은 함수. 공유결합 간선이 바뀌면 그 반응을 뺌(바뀌지 않아야 함). v1 값(q_rel_v1, v_rel_v1, rel_idx_v1)과 진단값(n_auto, rmsd_v1/v2, n_changed)을 npz에 남김 |
| F2 | 반응 원자 밖 대응을 독립적으로 확인하는 검사가 없다 | gate **Q-c**(`Q_report.json` `Q_c`): 모든 TS 원자와 대응 반응물 원자의 원소 일치(틀리면 STOP), 대응이 바뀐 반응·heavy 원자 수, 값이 바뀐 원자의 \|Δq_dist\|·\|Δv_dist\| 변화 분포, 조각별 heavy RMSD 전·후, automorphism 수 최대 |
| F3 | 캐시 지문에 코드가 없다 | 지문에 `models.py`·`train.py`·`fu2_common.py`·`tokens.py` sha256과 `S_splits.json` sha256 추가 |
| F4 | 검증 MAE가 NaN이면 best_epoch = 0 → 최종 학습이 1 epoch | `train.stop`: 학습 손실·검증 MAE가 NaN/inf, 또는 best_epoch < 1이면 exit 3 |
| F5 | A+G의 전역 토큰 사용, 저장 파일 정합을 확인하는 검사가 없다 | **C13**: 반응 a의 전역 토큰만 바꾸면 A+G 출력은 바뀌고(30짝 중 ≥ 90 %, > 1e-6) A는 그대로(≤ 1e-12). **C14**: C8의 `run_unit`이 쓴 parquet을 다시 읽어 y = F_pairs dd(pair_id·타깃 기준, ≤ 1e-12), yhat = 최종 모델 predict × s_y(≤ 1e-6), 행 수·시험 짝 일치 |

검증: F1–F5 수정을 따로 적대적 검토(관점 2개 + 반박 검증)하고, Q smoke(24 반응)에서 처음부터 v2로 계산한 값 = v1 npz 변환 값(24/24 동일)을 확인했다.
그다음 Q gate(Q-c) → T build → C1–C14 + smoke를 다시 돌렸다(결과는 2-3).

### 2-2. C12 진단 (사용자 결정 2026-10-21 "진단 후 판단"; 코드·기준 변경 없음, `C12_diag.json`)

C12(64짝, 300 epoch, dropout 0, batch 8)에서 AW 가장 작은 칸(L1·폭 64)만 학습 MAE / null = 0.066으로 기준(< 0.05)을 넘지 못했다.
아래 표는 F1 이전 데이터(fu2-tokens-1)로 돌린 진단이다(F1 뒤 데이터로 다시 돌린 값은 2-3).
같은 설정으로 64짝 표본 3개:

| AW 설정 | 표본 0 | 표본 1 | 표본 2 |
|---|---|---|---|
| L1·폭 64, 300 epoch (C12 조건) | 0.053 | 0.089 | 0.107 |
| L1·폭 128, 300 epoch | 0.010 | 0.014 | 0.038 |
| L2·폭 64, 300 epoch | 0.030 | 0.047 | 0.069 |
| L2·폭 128, 300 epoch | 0.006 | 0.006 | 0.024 |
| L1·폭 64, 1,000 epoch (T_max 300: 주기적 cosine, 600 epoch에 lr 1e-3로 돌아가 1,000에서 약 2.5e-4) | 0.025 | 0.021 | 0.037 |
| L1·폭 64, 1,000 epoch (T_max 1,000) | 0.0012 | 0.0010 | 0.011 |

해석: 폭을 늘리거나 학습을 길게 하면 기준보다 훨씬 낮게 맞추므로 구현 버그가 아니라 **가장 작은 AW 칸의 용량·학습량 문제**다.

**사용자 결정 (2026-10-21): "C12는 크기를 키우면 없어지니 무시."** → C12의 AW(가장 작은 칸) 실패는 Phase C STOP으로 보지 않는다.
C1–C11·C13·C14 통과와 smoke ok로 Phase C를 판정한다. `C_tests.json`의 `all_pass: false`는 이 C12 하나 때문이며, 기록을 그대로 둔다.
grid·학습 규칙은 바꾸지 않는다(AW 가장 작은 칸도 grid에 남고, 칸 선택은 검증 MAE로 한다).

### 2-3. 다시 돌린 결과 (fu2-atoms-2 / fu2-tokens-2)

F1–F5 수정 뒤 적대적 재검토(관점 2개 + 반박 검증)에서 나온 작은 지적도 반영했다: automorphism 한도를 계산 경로에서도 STOP,
Q-c의 "바뀐 원자"를 대응 기준으로, 반응 원자 재대응 STOP, 코드 sha256을 import 때 한 번(d1pair_common.py 포함), `Data()`가 T_report의
토큰 버전·Q_report·분할 sha256을 확인, collect가 모든 unit 지문을 현재 코드·입력과 비교, C_tests·C_smoke에 코드·입력 sha256,
C13 별도 난수열(C12 표본 = 처음과 같은 64짝) + 전역 토큰 = X182 확인, C14 NaN·ref_kind·unit 열 확인.

**Q (gate Q-a·Q-b·Q-c, `Q_report.json`).** 8,337 반응 v1 → v2 변환(xTB 없음, 실패 0). Q-a 최대 차 0(TS 순서 배열 포함), Q-b 최대 7e-8,
확장 통과 8,336 / 8,337(1 반응 제외, 그대로). **Q-c:** 원소 불일치 0, 반응 원자 재대응 0, automorphism 최대 A 12 · B 12(한도 10,000).
대응이 바뀐 반응 2,414개(heavy 원자 7,763, H 포함 19,555). 바뀐 원자의 |Δq_dist| 변화 중앙값 0.0004, p90 0.006, p99 0.024, 최대 0.048 e;
|Δv_dist| 중앙값 0, p99 0.016, 최대 0.045. 조각 heavy RMSD: 나빠진 조각 0. A 조각 970개가 바뀜(줄어든 폭 > 0.1 Å: 831, > 0.5 Å: 557,
중앙값 0.56 Å; v2 중앙값 0.132 Å). B 조각 1,584개가 바뀜(> 0.1 Å: 1,110, > 0.5 Å: 756, 중앙값 0.44 Å; v2 중앙값 0.134 Å).

**거울상 진단 (대응은 바꾸지 않음; 결정 필요).** F1 규칙의 Kabsch는 고유 회전만 쓴다. TS 조각이 반응물 구조의 거울상일 때(대칭인 반응
원자 쌍을 고정한 노르보넨류, 키랄 이형태 등) 바른 짝을 고르지 못할 수 있다. v2 RMSD가 1 Å을 넘는 조각은 A 175, B 1,901개.
반사를 허용하면 v2 RMSD가 0.1 Å 넘게 줄어드는 조각(거울상 맞춤)은 A 260(대칭 원자 있음 149), B 1,697(294)개. 같은 F1 규칙에 반사를
허용하면(O(3): automorphism × 고유/비고유 회전 중 최소) 대응이 바뀌는 조각은 A 539, B 1,031개이고, 그때 q_rel 변화는 중앙값 0.0002,
p90 0.004, p99 0.012, 최대 0.039 e(반응 원자 값·Q-a는 영향 없음). xTB 값은 반사에 대해 변하지 않으므로 O(3)가 물리적으로 일관된
선택이지만, 리뷰의 "Kabsch RMSD" 규칙을 바꾸는 일이라 사용자 결정으로 남긴다.

**T (`T_report.json`).** 8,336 반응, token_version fu2-tokens-2, gate T-a 통과, 원자 없는 짝 297(그대로).

**C1–C14 (`C_tests.json`, NVIDIA A10).** C1–C11, C13, C14 통과. C13: A 변형 차 0, A+G 30/30 변함, 전역 토큰 = X182 행(상대 차 1e-15).
C14: 17,130행, y = F_pairs dd(차 0), yhat = predict × s_y(차 0), ref_kind·unit 열 일치. **C12 실패(같은 64짝):** AW 0.066(기준 < 0.05),
GNN 0.008, TF 0.003; 참고(판정 아님, 같은 64짝): AW L1·폭 128·300 epoch 0.0085, AW L1·폭 64·1,000 epoch(T_max 1,000) 0.0008.
**smoke (`C_smoke.json`):** 6개 구성 × 가장 작은·큰 칸, gate M-1′ 모두 0.

### 2-3b. C12 진단 다시 (fu2-tokens-2, `C12_diag.json`)

같은 3개 표본, F1 뒤 데이터: L1·폭 64·300 epoch 0.056 / 0.086 / 0.105, L1·폭 128 0.010 / 0.015 / 0.039, L2·폭 64 0.033 / 0.045 / 0.071,
L2·폭 128 0.006 / 0.007 / 0.023, L1·폭 64·1,000 epoch(T_max 1,000) 0.0009 / 0.0010 / 0.0095 — F1 이전과 같은 결론.

### 2-3c. 사용자 결정: Kabsch 반사 허용 (2026-10-21, fu2-atoms-3)

거울상 진단(2-3)을 보고 **"반사 허용으로 변경"**. F1 규칙의 후보 점수를 min(고유 회전, 비고유 회전) Kabsch RMSD로 바꿨다(나머지 규칙 —
반응 원자 고정 automorphism, 동점은 저장된 VF2 대응 유지, H 규칙 — 그대로). `ATOM_VERSION = fu2-atoms-3`, `TOKEN_VERSION = fu2-tokens-3`.
저장된 fu2-atoms-1 값(npz의 *_v1)에서 xTB 없이 다시 변환했다. 고유 회전 규칙(fu2-atoms-2)과 다른 조각 수는 `Q_report` `vs_proper_rule`,
`mirror_diag`는 이제 같은 O(3) 선택을 따로 다시 계산하는 자기 검사(바뀌어야 할 대응 0이 아니면 STOP).
결과는 2-3d.

### 2-3d. O(3) 뒤 다시 돌린 결과 (fu2-atoms-3 / fu2-tokens-3; 지금 쓰는 입력)

**Q.** Q smoke 24 반응 통과(새로 계산 = 저장 값 변환, 24/24 동일). 8,337 반응 변환(xTB 없음, 실패 0). Q-a 최대 차 0(TS 순서 포함),
Q-b 최대 7e-8, 확장 8,336 / 8,337, 사용 8,336. **Q-c:** 원소 불일치 0, 반응 원자 재대응 0, O(3) 자기 검사 불일치 0, automorphism 최대 12.
VF2 대응 대비 바뀐 반응 2,280(heavy 7,377, H 포함 17,672). 바뀐 원자의 |Δq_dist| 변화 중앙값 0.0007, p90 0.007, p99 0.024, 최대 0.048 e;
|Δv_dist| p99 0.017, 최대 0.045. 조각 heavy RMSD(반사 허용, 전후 같은 기준): 나빠진 조각 0.
A: 바뀐 조각 1,116(줄어든 폭 > 0.1 Å 780, > 0.5 Å 457, 중앙값 0.42 Å), v2 중앙값 0.128 Å, 1 Å 넘는 조각 103(고유 회전 규칙 175).
B: 바뀐 조각 1,227(> 0.1 Å 944, > 0.5 Å 512, 중앙값 0.44 Å), v2 중앙값 0.079 Å, 1 Å 넘는 조각 510(고유 회전 규칙 1,901).
고유 회전 규칙(fu2-atoms-2)과 다른 조각: A 539(heavy 1,768), B 1,031(heavy 2,644). 최종 대응이 거울상 맞춤인 조각: A 723, B 2,258.

**T.** 8,336 반응, fu2-tokens-3, gate T-a 통과, 원자 없는 짝 297.

**C1–C14 (`C_tests.json`, NVIDIA A10, job 1030202).** C1–C11, C13, C14 통과(C13 전역 토큰 = X182 상대 차 1e-15; C14 y·yhat 차 0).
C12는 판정 제외(사용자 결정): 같은 64짝에서 AW 0.064, GNN 0.005, TF 0.004; 참고 AW 폭 128 0.0075, 1,000 epoch 0.0007.
`all_pass: false`는 C12 하나 때문이다. **smoke (`C_smoke.json`):** ok, gate M-1′ 모두 0, Phase M 상한 약 132 A10 GPU-시간(조기 종료 없음 가정).

**상태:** Phase C 끝(C12 제외). 다음은 사용자 코드 승인 → Phase A(ADD2: 2-4의 대표 구성 규칙 + Q·T·분할 sha256) → Phase M.
Phase M GPU는 gpu4/gpu5 RTX A6000에만(지문에 GPU 이름).

### 2-4. Phase A(ADD2)에 그대로 적을 것 (REVIEW 4절)

- 대표 구성: rev 9 `d1_eval.representatives`와 같은 방식. 타깃마다 고른 칸의 안쪽 검증 MAE를 kcal/mol로(× 그 fold의 s_y), 바깥 fold
  5개 평균이 가장 낮은 구성. 동점이면 AW → GNN → TF, A → A+G 순.
- M-4(LODO·LOSO)의 "모델마다 대표 하나": 그 모델의 두 변형 중 칸 고를 때 쓴 값(9 타깃 표준화 평균 검증 MAE)의 5 fold 평균이 낮은 쪽.
- 알려진 점: 최종 학습은 튜닝에서 고른 epoch 수를 그대로 쓰지만 짝이 약 1.33배라 optimizer step도 약 1.33배(명령서 규칙 그대로).
- Phase E 평가·M-4 실행·대표 고르기 코드는 실행 전에 따로 검토받는다.
