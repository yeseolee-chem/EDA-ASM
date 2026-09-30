# D1 validation — VALIDATION_REPORT

## 1. 사전 등록과 실행

| file | sha256[:16] | commit | clean |
|---|---|---|---|
| config_val.yaml | 11c74e58cfaf1cc9 | 85214ce972c3c83da0e4a2411d301ad7d7dbaba4 | True |
| val_d0_set.csv | 82f29109066b62e5 | 43b962b4dafe5199128230a261761fd67c54809d | True |
| tasks.csv | 0bdfee54d2a45fdb | 43b962b4dafe5199128230a261761fd67c54809d | True |
| val_manifest.csv | e1d594e82d88bd80 | 43b962b4dafe5199128230a261761fd67c54809d | True |

worker job ids: 996236, 996238, 996239, 996240, 996241, 996242, 996243, 996244, 996245, 996246, 996557, 996558

tasks: 148 — {'done': 147, 'failed': 1}

## 2. 판정표 (A1–A10)

| id | value | 95% CI | result | note |
|---|---|---|---|---|
| A1 | 19/20 ok, pipeline bugs 0 | 0.764–0.991 | PASS | policy allowed; classes {"ok": 19, "gate": 0, "no_ts_unconfirmed": 1, "chemical_no_saddle": 0, "engine_abnormal": 0, "pipeline": 0, "incomplete": 0}; no-TS 1/20 (CI 0.009–0.236, Coley 12.3% within: True)  |
| A2 | 0 of 113 labels with a failed gate | | PASS | |
| A3 | 24/24, max \|Δ\| 0.0127 kcal/mol | | PASS |  |
| A4 | 24/24 agree | | PASS | disagreements: 0 |
| A5-eng | n = 24/24, TS RMSD median 0.0091 Å | see §3 | FAIL | |
| A5-run | NF max 3.452 | | report only | NF > 1: True |
| A5-e2e | n = 23; \|Δbarrier\| ≤ 1.0: 0.696 | see §3 | PASS | |
| A6 | (a) 0.000 Eh (b) 1.37e-09 Eh (c) 2.74e-15 Å | | PASS |  |
| A7 | 8/8 | | PASS | |
| A8 | ok hard fails 0; foreign {'3090': True, '3766': True, '4252': True}; no-bond {'3400': True, '5783': True} | | PASS | short-bond flags (ok): 1 |
| A9 | 10.74 core-h/row (n=28); 2,846 rows ≈ 30569 core-h, 15.9 d | 6.07–15.41 | report only | |
| A10 | J07 ok: True; J03: None | | not judged | |

**A1 engine_abnormal files** (ORCA outputs autodE 1.4.5 would call abnormal, no-TS rows): none

## 3. 가설 판정 자료 (H0 / H1 / H2)

**V1 (H0):** A6(a) runs [{"src_job": "J08", "max_abs_dE_eh": 0.0, "max_abs_dlabel_kcal": 0.0}, {"src_job": "J11", "max_abs_dE_eh": 0.0, "max_abs_dlabel_kcal": 0.0}]; A6(b) [{"src_job": "J08", "max_abs_dE_eh": 1.0231815394945443e-12}, {"src_job": "J11", "max_abs_dE_eh": 1.3749286154052243e-09}]; A6(c) [{"rxn_id": 20, "aligned_max_dev_A": 2.737549743767154e-15, "raw_max_dev_A": 0.0}, {"rxn_id": 105, "aligned_max_dev_A": 2.275280134513746e-15, "raw_max_dev_A": 0.0}]

**V2a:** {"L0": {"n": 24, "median_max": 0.00047401607849999997, "max_max": 0.000732416558, "median_rms": 0.0001547361893243665}, "L1": {"n": 24, "median_max": 0.000304818352, "max_max": 0.000607245417, "median_rms": 8.836171012860715e-05}, "L2": {"n": 24, "median_max": 0.000293881664, "max_max": 0.000579414607, "median_rms": 7.92033019786912e-05}}; rms gradient decreasing L0 > L1 > L2 in 21/24

**V2b L0 (A5-eng)** (n = 24)

| target | MAE_ref | mean Δ | 95% CI | RMS Δ | bias ok | RMS ok | R | R CI |
|---|---|---|---|---|---|---|---|---|
| barrier | 1.45 | 0.037 | 0.025–0.049 | 0.047 | PASS | PASS | — | —–— |
| d1 | 1.12 | 0.083 | -0.004–0.170 | 0.218 | PASS | PASS | — | —–— |
| d2 | 0.82 | 0.066 | -0.005–0.137 | 0.177 | PASS | PASS | — | —–— |
| elst | 1.52 | 0.156 | -0.162–0.475 | 0.755 | PASS | PASS | — | —–— |
| pauli | 2.02 | -0.319 | -1.006–0.367 | 1.622 | PASS | FAIL | — | —–— |
| oi | 1.16 | 0.064 | -0.270–0.397 | 0.776 | PASS | FAIL | — | —–— |
| disp | 1.25 | 0.021 | 0.012–0.030 | 0.030 | PASS | PASS | — | —–— |
| cpcm | 1.01 | -0.021 | -0.049–0.006 | 0.068 | PASS | PASS | — | —–— |
| cds | 0.23 | -0.003 | -0.009–0.003 | 0.014 | PASS | PASS | — | —–— |

**V2b L2** (n = 8)

| target | MAE_ref | mean Δ | 95% CI | RMS Δ | bias ok | RMS ok | R | R CI |
|---|---|---|---|---|---|---|---|---|
| barrier | 1.45 | -0.008 | -0.026–0.010 | 0.022 | PASS | PASS | — | —–— |
| d1 | 1.12 | 0.012 | -0.069–0.093 | 0.092 | PASS | PASS | — | —–— |
| d2 | 0.82 | 0.026 | -0.034–0.087 | 0.072 | PASS | PASS | — | —–— |
| elst | 1.52 | -0.013 | -0.143–0.117 | 0.146 | PASS | PASS | — | —–— |
| pauli | 2.02 | 0.031 | -0.252–0.313 | 0.318 | PASS | PASS | — | —–— |
| oi | 1.16 | -0.026 | -0.159–0.108 | 0.151 | PASS | PASS | — | —–— |
| disp | 1.25 | -0.005 | -0.022–0.011 | 0.019 | PASS | PASS | — | —–— |
| cpcm | 1.01 | -0.025 | -0.064–0.013 | 0.050 | PASS | PASS | — | —–— |
| cds | 0.23 | -0.011 | -0.019–-0.003 | 0.014 | PASS | PASS | — | —–— |

**V4 e2e (A5-e2e)** (n = 23)

| target | MAE_ref | mean Δ | 95% CI | RMS Δ | bias ok | RMS ok | R | R CI |
|---|---|---|---|---|---|---|---|---|
| barrier | 1.45 | 0.689 | -0.610–1.988 | 3.018 | PASS | not judged | 0.567 | 0.038–7.198 |
| d1 | 1.12 | 0.702 | -0.613–2.017 | 3.056 | PASS | not judged | 0.911 | 0.074–344.831 |
| d2 | 0.82 | 0.505 | -0.434–1.444 | 2.183 | PASS | not judged | 0.323 | 0.015–52.358 |
| elst | 1.52 | 1.169 | -0.125–2.462 | 3.150 | PASS | not judged | 0.213 | 0.013–6.965 |
| pauli | 2.02 | -1.252 | -4.071–1.567 | 6.497 | PASS | not judged | 0.289 | 0.032–17.045 |
| oi | 1.16 | 0.291 | -1.094–1.676 | 3.147 | PASS | not judged | 0.204 | 0.017–29.941 |
| disp | 1.25 | 0.006 | -0.131–0.143 | 0.311 | PASS | not judged | 0.515 | 0.078–3.140 |
| cpcm | 1.01 | -0.752 | -1.671–0.167 | 2.210 | PASS | not judged | 2.948 | 0.270–22.784 |
| cds | 0.23 | -0.007 | -0.055–0.041 | 0.108 | PASS | not judged | 0.515 | 0.119–3.117 |

**V3 / V6b (A5-run):**

| target | SD_run D0 | SD_run D1 | SD all | NF all |
|---|---|---|---|---|
| barrier | 2.835 | 2.591 | 2.752 | 1.898 |
| d1 | 2.259 | 3.160 | 2.608 | 2.328 |
| d2 | 2.714 | 0.158 | 2.194 | 2.675 |
| elst | 4.795 | 2.102 | 4.066 | 2.675 |
| pauli | 8.475 | 2.259 | 6.973 | 3.452 |
| oi | 4.897 | 0.918 | 3.992 | 3.441 |
| disp | 0.306 | 0.400 | 0.341 | 0.273 |
| cpcm | 0.909 | 1.332 | 1.075 | 1.064 |
| cds | 0.106 | 0.130 | 0.115 | 0.501 |

**TS 일치율:** V3 pairs 0.500 (n=22); V4 control = Coley 0.478 (n=23); lowest of 3 = Coley 0.250 (n=8)

**새 허수 규칙 영향:** 0/70 TS with extra modes in (−50, 0) cm⁻¹: []

**V2c refopt (refopt − label, kcal/mol):** d1_kcal mean -0.018 / max \|·\| 0.184, d2_kcal mean -0.030 / max \|·\| 0.060, barrier_kcal mean -0.048 / max \|·\| 0.147

## 4. 교수님 명제에 대한 답 (데이터만)

같은 ORCA 입력 → 최대 차이 0.000 Eh (nprocs 1 vs 4: 1.37e-09 Eh; 같은 OptTS 두 번: 2.74e-15 Å). Coley 구조에서 ORCA L0 재최적화 → TS 이동 중앙값 0.0091 Å, 채널 \|mean Δ\| 최대 0.319 kcal/mol. 반복 실행 → pooled SD 최대 6.973 kcal/mol.

## 5. §8 결정표

| 조건 | 해당 | 조치 |
|---|---|---|
| A3, A6(a)(b), A7 중 하나라도 FAIL | 아니오 | STOP. `$Q/STOP`, 본실험 불가 |
| A6 PASS, A5-eng PASS, A5-e2e PASS | 아니오 | GO (ORCA L0). A5-run은 따로 판단 |
| A5-eng FAIL (L0), L2 기준 충족 | **예** | 사용자 결정: L2 설정 / D0 재최적화 |
| A5-eng FAIL (L0, L2 모두) | 아니오 | 사용자 결정: G16 / D0 전체 ORCA 재라벨 |
| A5-e2e FAIL, A5-eng PASS | 아니오 | STOP 후 보고 (autodE 1.2→1.4.5 가능성) |
| A5-run: NF > 1 채널 있음 | **예** | 사용자 결정: 라벨 정의·잡음 하한 서술 |
| A1 FAIL | 아니오 | 실패 유형별 대책 후 재검증 |
| A2, A4, A8, A10 FAIL | 아니오 | 원인 보고, 수정안, 부분 재검증 |

**권고: 조건부 (해당 행의 사용자 결정 필요)**

## 6. 비용

D1 행당 10.74 core-h (중앙값 8.55, 최대 65.28, n=28); 2,846행 외삽 30569 core-h, 10 job × 8코어로 약 15.9일.

## 7. 한계

- 표본은 35원자 이하 (D0 중앙값 44원자)
- SMD 이산화 차이는 ORCA 안에서 격리할 수 없음 (L2도 ORCA SMD)
- D0 진동수 파일이 없어 V9에서 허수 개수 gate는 적용 불가

## 부록: 재실행 기록

# 재실행 기록 (VALIDATION_SPEC §2, 파일럿 SPEC §2: 실패를 다시 돌릴 때는 사유를 보고서에 남긴다)

2026-09-30 06:55 KST, 사용자 결정 (c). 1차 worker pool은 job 996236이다. 사전 등록 파일 4개는 수정하지 않았다.

| task | 1차 결과 | 원인 (로그 확인) | 분류 | 조치 |
|---|---|---|---|---|
| V7_J03 | `$Q/failed` (rc=1, 4.4 h) | ORCA relaxed scan이 100점 중 89점째에서 `orca_prop_mpi` "error termination in PROPERTIES"로 종료. 남은 11점이 없어 minimax 경로가 이어지지 않음 | 인프라 (MPI) | `$Q/failed/V7_J03` 삭제 → scan 처음부터 재실행 |
| V3_0020_r2 | `.fail_ts` "autodE exception: CouldNotGetProperty: Could not get energy" | `transition_states/neb/4/4_0_orca.out`: `orca_leanscf_mpi` "error termination in LEANSCF" | 인프라 (MPI) | `.fail_ts`와 `$Q/done/` 표시 삭제 → autodE checkpoint에서 재개 |
| V6b_J08_r3 | `.fail_ts` "autodE exception: CouldNotGetProperty: Could not find Hessian file" | `ts_guess_vvEPtl_template_2-3_7-8_hess_orca.out`: `orca_scfresp_mpi` "error termination in SCF RESPONSE" | 인프라 (MPI) | 같음 |
| V4_0047 | `.fail_ref` "CouldNotGetProperty: Could not get energy" | `ref/dipole_alt_hess_orca.out`: `orca_startup_mpi` "error termination in Startup" | 인프라 (MPI) | `.fail_ref`와 `$Q/done/` 표시 삭제 → ref 단계 재실행 |
| V6b_J09_r2 | `$Q/failed` (rc=1) | `assemble.py`가 `TypeError: dict() got multiple values for keyword argument 'sum_mismatch_promoted'`로 종료. 파일럿 번들에 원래 있던 버그로, `eda_sum_mismatch`에서 `ok`로 승격하는 라벨은 모두 이 단계에서 멈춘다 | 파이프라인 버그 | `assemble.py` 수정(키를 한 번만 넣음, 판정 기준은 그대로) → `$Q/failed` 삭제 → label 단계만 재실행 (ts, ref, inputs, sp는 이미 완료) |

**재실행하지 않은 것:**
- V4_0500: `.fail_ts` "ValueError: RMSD must be computed between atom lists of the same length: 31 =/= 0".
- ORCA 출력 18개는 autodE 종료 규칙상 모두 정상이다. autodE 내부 오류(원자 0개 conformer)이므로 인프라 실패가 아니다. 결과에 그대로 둔다.

**MPI 오류의 분포:**
- 노드는 n107, n108, n115로 흩어져 있어, 특정 노드 하나의 문제로 보이지 않는다.
- 검증 전체 ORCA 출력 중 MPI 오류 종료는 6건이다.
- 환경 변수는 CLAUDE.md와 같다(`OMPI_MCA_pml=ob1`, `hcoll` off, `UCX_TLS=tcp,self,sm`).

