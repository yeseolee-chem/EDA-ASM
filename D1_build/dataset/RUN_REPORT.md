# RUN_REPORT

생성 2026-10-18 22:34:10

대상 11596행 · 끝남 10928 · ok 9496 (끝난 것의 86.9%) · 진행 중 0 · 대기 668

## 분류

| 분류 | 건수 |
|---|---|
| ok | 9496 |
| no_ts | 805 |
| gate_ts | 101 |
| stereo_mismatch | 345 |
| ref_fail | 1 |
| gate_label | 2 |
| engine_abnormal | 8 |
| infra_fail | 10 |
| pipeline | 160 |

## 게이트 잔차 (|값|, kcal/mol)

| 항목 | 중앙값 | 최대 | n |
|---|---|---|---|
| identity_residual_kcal | 1.332e-15 | 2.854e-10 | 9496 |
| bond_reference_residual_kcal | 1.569e-08 | 3.199e-08 | 9496 |
| eda_sum_minus_bond_kcal | 2.007e-03 | 9.806e-02 | 9496 |
| channel_closure_residual_kcal | 2.007e-03 | 9.806e-02 | 9496 |

## 라벨 게이트 탈락 (독립 검사 포함)

- T07934: status=eda_sum_mismatch failed_gates=[] audit=[] uniformity=[]
- T11092: status=eda_sum_mismatch failed_gates=[] audit=[] uniformity=[]

## 플래그

- flag_async: 225
- d2_below_min: 100
- audit_flags: 18
- d1_below_min: 1

## dipole별 ok

| dipole | ok | 전체 |
|---|---|---|
| azide | 1930 | 2096 |
| azomethine ylide | 1820 | 2334 |
| carbonyl ylide | 1356 | 2334 |
| nitrile oxide | 1943 | 2096 |
| nitrone | 2447 | 2736 |

## 골격별 ok

| 골격 | ok | 전체 |
|---|---|---|
| A | 885 | 1045 |
| Ap | 792 | 808 |
| AppE | 623 | 645 |
| AppZ | 588 | 645 |
| B | 2383 | 2570 |
| Bp | 1237 | 1290 |
| C | 920 | 1045 |
| Cp | 396 | 405 |
| D | 793 | 1045 |
| Dp | 649 | 808 |
| DppC | 111 | 645 |
| DppT | 119 | 645 |

## 부착 단계별 ok

| 부착 단계 | ok | 전체 |
|---|---|---|
| 1 | 2336 | 2956 |
| 2 | 2405 | 2880 |
| 3 | 2411 | 2880 |
| 4 | 2344 | 2880 |

## 실행 요약

- 작업 11596 · 코어시간 합 1,394,444 (timing.tsv + label.sp_coreh)
- 거두어 간 행 10728

## 데이터셋 크기

- ok 행 9496 · 완결된 반응(형제가 모두 ok) 4818 / 6460

## 입체 (§10.3)

- stereo_mismatch 345: stereo_mismatch: requested C[C@@H]1CCCC[C@H](C)C2=C1[C@H](c1 (40); stereo_mismatch: requested C[C@@H]1CCCC[C@H](C)C2=C1[C@@H](c (38); stereo_mismatch: requested C[C@H]1CCCC[C@H](C)C2=C1CN(C)[C@H (31); stereo_mismatch: requested C[C@H]1CCCC[C@H](C)C2=C1ON(C)[C@H (22); stereo_mismatch: requested C[C@H]1CCCC[C@H](C)C2=C1ON(C)[C@@ (18)
- 형제 비교 3792쌍 · 같은 TS(flag_duplicate_sibling) 0쌍

## 끝나지 않은 시도 (FIX1 §3: attempts.tsv 에 시작만 있고 timing.tsv 에 끝이 없는 것)

- 398건 (227행) — 코어시간 합에는 들어가지 않았다 (timing.tsv 만 합산)
- 보류(held) 0행 (FIX1 §9)

## 기준 구조

- alt_used 3304/9496 · 음의 변형(d < −0.5) 101

## 재실행 기록

- RETRIES.md → results/RETRIES.md (68행 재실행)

## STOP 점검 (§9.4)

- **같은 사유 pipeline 실패 24건: autodE exception: ValueError: RMSD must be computed between **
- **같은 사유 pipeline 실패 65건: autodE exception: NoConformers:**
- **같은 사유 pipeline 실패 16건: autodE exception: CouldNotGetProperty: Could not get energy**
- **같은 사유 pipeline 실패 18건: cannot match reactant geometries to the mapped SMILES**
- **같은 사유 pipeline 실패 7건: ValueError: No matching found**
- **같은 사유 pipeline 실패 5건: AtomValenceException: Explicit valence for atom # 0 C, 5, is**
