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

## 2. 사용자 검토

(사용자 검토 후 여기에 적는다.)
