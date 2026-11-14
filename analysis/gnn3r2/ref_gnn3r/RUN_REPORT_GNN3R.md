# GNN3R 실행 보고 — g2r 입력과 다시 만든 짝으로 GNN3 다시 학습

- 지시서: `docs/specs/GNN3R.md` · 브랜치 `gnn3r` · 결과 commit `b7401a56` · push한 HEAD `98cbe350204e71463db7252068973ac22feef1db`
- scratch `/gpfs/tmp_cpu2/yeseo1ee/gnn3r` (cv/, final/ = .pt), 결과 `analysis/gnn3r/results/`, manifest `results/gnn3r.json`
- 데이터셋 고정 파일: `results/F_pairs_r.csv` sha256 `5597ad11558efb1bc3fc6322f99b5920026166e161a1df383662c6905b95a467` (`B_label_report.json`에 기록, gnn3 로더가 이 값으로 확인)

## 1. 단계별 판정

| 단계 | 판정 | 비고 |
|---|---|---|
| 1 시작 조건 | 통과 | 1-1 … 1-11, RDKit 2026.03.2, 여유 111 GB |
| A 행 속성 | 통과 | A-1 … A-6; B′ need 922행 {'cis': 467, 'trans': 451, '': 4} (빈칸 4행 = X = H 기준 행, DEC-1); F 안 Ph 중복 6 → 0 |
| B 짝 | 통과 | B-0 … B-7, B-R |
| S 분할 | 통과 | S-1 … S-4, sklearn 1.3.1 |
| D 데이터 | 통과 | D-1 … D-4 |
| K 교차검증 | 통과 | K-1 … K-3 (RTX 3090) |
| E 평가 | 통과 | E-1: 8,170짝 평가 |
| W 최종 모델 | 통과 | W-1: 8,339짝 학습 |

## 2. 깔때기 (반응)

| 단계 | 빠짐 (기대 / 실제) | 남음 (기대 / 실제) |
|---|---|---|
| D1 비D″ 라벨 | — | 9,266 / 9,266 |
| short_forming 9 + regio_ambiguous 5 | 14 / 14 | 9,252 / 9,252 |
| g2 TS 없음·autodE 오류 등 (X, TS 단계) | 599 / 599 | 8,653 / 8,653 |
| g2 TS 검사·조각 분할 (X, 검사) | 19 / 19 | 8,634 / 8,634 |
| g2 기준 구조 복구 못 함 (R) | 17 / 17 | 8,617 / 8,617 |
| 짝 없음 | 13 / 28 | **8,604 / 8,589** |

사용 가능하지만 짝이 없는 반응 28개 (로컬 추정 13개와 비교):

| job_id | 서버 status | 로컬 추정 | fam |
|---|---|---|---|
| T01073 | singleton | singleton | `azomethine ylide/A/R1/t////anchor` |
| T01318 | p4_dropped | (짝 있음) | `azomethine ylide/B/R1/t//endo/cis/anchor` |
| T02374 | no_family | no_family | `carbonyl ylide/A/R1/c///` |
| T02510 | no_family | no_family | `carbonyl ylide/AppZ/R1/t///trans` |
| T02573 | no_family | no_family | `carbonyl ylide/B/R1/t//exo/trans` |
| T02606 | singleton | (짝 있음) | `carbonyl ylide/B/R1/c//exo/trans/anchor` |
| T02833 | singleton | no_family | `carbonyl ylide/D/R2/////anchor` |
| T08439 | singleton | p4 | `carbonyl ylide/Bp/R1/c//endo/trans/anchor` |
| T09850 | singleton | singleton | `azomethine ylide/AppE/R1/c///trans/anchor` |
| T09972 | p4_dropped | p4 | `azomethine ylide/B/R1/t//endo/cis` |
| T10047 | p4_dropped | (짝 있음) | `azomethine ylide/Bp/R1/t//endo/cis/anchor` |
| T10056 | p4_dropped | (짝 있음) | `azomethine ylide/Bp/R1/t//endo/cis/anchor` |
| T10067 | p4_dropped | (짝 있음) | `azomethine ylide/Bp/R1/t//endo/cis/anchor` |
| T10078 | p4_dropped | (짝 있음) | `azomethine ylide/Bp/R1/t//exo/trans` |
| T10092 | p4_dropped | (짝 있음) | `azomethine ylide/Bp/R1/t//endo/cis/anchor` |
| T10111 | p4_dropped | (짝 있음) | `azomethine ylide/Bp/R1/t//endo/cis/anchor` |
| T10523 | p4_dropped | (짝 있음) | `nitrone/B/R1/c//endo/cis/anchor` |
| T10560 | p4_dropped | (짝 있음) | `nitrone/B/R1/c//endo/cis/anchor` |
| T10596 | p4_dropped | (짝 있음) | `nitrone/B/R1/c//endo/cis/anchor` |
| T10683 | p4_dropped | (짝 있음) | `nitrone/Bp/R1/c//endo/trans/anchor` |
| T10719 | p4_dropped | (짝 있음) | `nitrone/Bp/R1/c//endo/trans/anchor` |
| T10744 | p4_dropped | (짝 있음) | `nitrone/Bp/R1/c//endo/trans/anchor` |
| T10748 | p4_dropped | (짝 있음) | `nitrone/Bp/R1/c//endo/trans/anchor` |
| T10756 | p4_dropped | (짝 있음) | `nitrone/Bp/R1/c//endo/trans/anchor` |
| T11185 | no_family | no_family | `carbonyl ylide/B/R1/t//exo/cis` |
| T11220 | no_family | no_family | `carbonyl ylide/B/R1/t//endo/trans` |
| T11346 | no_family | (짝 있음) | `carbonyl ylide/Bp/R1/c//exo/trans` |
| T11449 | singleton | (짝 있음) | `carbonyl ylide/D/R1/t////anchor` |
| T10679 | in_dataset | p4 | (서버에서는 데이터셋 안) |
| T10704 | in_dataset | p4 | (서버에서는 데이터셋 안) |
| T01507 | in_dataset | no_family | (서버에서는 데이터셋 안) |

9,252행 status: in_dataset 8,589, not_usable 635, p4_dropped 16, singleton 6, no_family 6 (in_dataset 중 F+Xb+pos 중복 직접 행 0)

## 3. 데이터셋

- 반응 **8,589** (기대 8,604, 허용 8,432–8,776), 짝 **8,339** (기대 8,355, 허용 8,188–8,522): 기준 짝 6,978, anchor 짝 1,361
- 계열: fam 문자열 244개, 짝이 있는 계열 묶음 250개 (anchor_family 81, reference 154, reference_not_usable 15). 지시서의 로컬 추정 '계열 323개'와는 세는 단위가 다를 수 있다. core 50개, 기준 반응(b) 154개, anchor b 96개
- anchor 짝의 출처: {'anchor_family': 1107, 'reference_not_usable': 268} (anchor_family = D1PAIR anchor 계열, reference_not_usable = 기준이 g2r에서 쓸 수 없어 F로 다시 묶은 계열)

**fold_T1별** (실제 / 로컬 추정)

| | anchor | 기준 | 합 |
|---|---|---|---|
| -1 | 169 / 170 | 0 / 0 | 169 / 170 |
| 0 | 267 / 269 | 1,415 / 1,417 | 1,682 / 1,686 |
| 1 | 301 / 305 | 1,399 / 1,401 | 1,700 / 1,706 |
| 2 | 171 / 171 | 1,243 / 1,243 | 1,414 / 1,414 |
| 3 | 36 / 36 | 1,308 / 1,308 | 1,344 / 1,344 |
| 4 | 417 / 427 | 1,613 / 1,608 | 2,030 / 2,035 |
| **합** | 1,361 / 1,378 | 6,978 / 6,977 | 8,339 / 8,355 |

**쌍극자별** (실제 / 로컬 추정)

| | anchor | 기준 | 합 |
|---|---|---|---|
| nitrone | 261 / 266 | 1,985 / 1,986 | 2,246 / 2,252 |
| azide | 261 / 260 | 1,535 / 1,536 | 1,796 / 1,796 |
| nitrile oxide | 44 / 44 | 1,663 / 1,663 | 1,707 / 1,707 |
| azomethine ylide | 484 / 488 | 964 / 966 | 1,448 / 1,454 |
| carbonyl ylide | 311 / 320 | 831 / 826 | 1,142 / 1,146 |
| **합** | 1,361 / 1,378 | 6,978 / 6,977 | 8,339 / 8,355 |

**위치별** (실제 / 로컬 추정)

| | anchor | 기준 | 합 |
|---|---|---|---|
| p | 326 / 328 | 1,839 / 1,839 | 2,165 / 2,167 |
| m | 326 / 325 | 1,879 / 1,879 | 2,205 / 2,204 |
| o | 388 / 405 | 1,708 / 1,705 | 2,096 / 2,110 |
| direct | 321 / 320 | 1,552 / 1,554 | 1,873 / 1,874 |
| **합** | 1,361 / 1,378 | 6,978 / 6,977 | 8,339 / 8,355 |

**참조 목록(`docs/specs/gnn3r_ref/`)과의 차이** (`results/B_diff_vs_est.csv`, 이유별 수)

| 종류 | 추정에 있음 | 서버에 있음 | 이유 | 수 |
|---|---|---|---|---|
| pair | False | True | estimate pairs a with | 154 |
| pair | True | False | server pairs a with | 154 |
| reaction | True | False | server p4_dropped | 15 |
| pair | True | False | a server p4_dropped | 14 |
| pair | True | False | a is a server anchor b | 10 |
| pair | False | True | a is an estimate anchor b | 9 |
| reaction | False | True | not in the estimate list | 3 |
| pair | False | True | a not in the estimate list | 2 |
| pair | True | False | a server singleton | 2 |
| reaction | True | False | server singleton | 2 |
| pair | True | False | a server no_family | 1 |
| reaction | True | False | server no_family | 1 |

test 크기 (실제 / 로컬 추정): f0 1,682 / 1,686, f1 1,700 / 1,706, f2 1,414 / 1,414, f3 1,344 / 1,344, f4 2,030 / 2,035, 늘 학습 169 / 170

## 4. P-4 입체 검사

- 검사한 짝 8,355 (반응 8,605, 고리 구성 실패 반응 0)
- v2: no_relative_info 3,841, match 2,422, mirror_match 2,076, mismatch 16
- mismatch로 버린 짝 16 (a는 다시 짝짓지 않음): T01318→T01264 (anchor), T09972→T01302 (reference), T10078→T01421 (reference), T10047→T10116 (anchor), T10056→T10116 (anchor), T10067→T10116 (anchor), T10092→T10116 (anchor), T10111→T10116 (anchor), T10523→T04796 (anchor), T10560→T04796 (anchor), T10596→T04796 (anchor), T10683→T04955 (anchor), T10719→T04955 (anchor), T10744→T04955 (anchor), T10748→T04955 (anchor), T10756→T04955 (anchor)
- D1PAIR와 같은 짝 6,592개의 v2 = `P4_stereo.csv` (B-3), cr2 라벨 차이 최대 1.4e-14 kcal/mol (6,591짝, B-4)

## 5. 학습

| fold | 고른 칸 (층×폭) | epoch | 검증 MAE strain / inter / solv / barrier | 학습 / test 짝 | 시간 (h) | GPU |
|---|---|---|---|---|---|---|
| 0 | 6×64 | 41 | 0.779 / 0.589 / 0.495 / 0.704 | 6,657 / 1,682 | 1.05 | NVIDIA GeForce RTX 3090 |
| 1 | 4×64 | 55 | 0.973 / 0.757 / 0.611 / 0.876 | 6,639 / 1,700 | 1.00 | NVIDIA GeForce RTX 3090 |
| 2 | 6×128 | 46 | 0.971 / 0.799 / 0.588 / 0.911 | 6,925 / 1,414 | 1.43 | NVIDIA GeForce RTX 3090 |
| 3 | 4×128 | 36 | 1.023 / 0.721 / 0.635 / 0.914 | 6,995 / 1,344 | 0.99 | NVIDIA GeForce RTX 3090 |
| 4 | 6×128 | 35 | 1.024 / 0.777 / 0.618 / 0.948 | 6,309 / 2,030 | 1.09 | NVIDIA GeForce RTX 3090 |

- 최종 setting: 6층 × 128폭, epoch 41 (fold best epoch [35, 36, 41, 46, 55], 칸 선택 수 {'{"n_layers": 6, "width": 64}': 1, '{"n_layers": 4, "width": 64}': 1, '{"n_layers": 6, "width": 128}': 2, '{"n_layers": 4, "width": 128}': 1})
- 최종 모델: seed 5개, 8,339짝, 0.92 h, NVIDIA GeForce RTX 3090; `.pt` = `/gpfs/tmp_cpu2/yeseo1ee/gnn3r/final/gnn3_final__seed{0..4}.pt`

## 6. 성능 (교차검증 test 예측, kcal/mol)

| 집합 | 위치 | target | n | MAE | r | 예전 GNN3 MAE (n) |
|---|---|---|---|---|---|---|
| P_main | all | strain | 6,978 | 0.811 | 0.855 | 0.808 (6,403) |
| P_main | all | inter | 6,978 | 0.635 | 0.779 | 0.621 (6,403) |
| P_main | all | solv | 6,978 | 0.506 | 0.690 | 0.501 (6,403) |
| P_main | all | barrier | 6,978 | 0.741 | 0.855 | 0.724 (6,403) |
| P_main | p | strain | 1,839 | 0.374 [0.301, 0.450] | 0.838 | 0.359 (1,658) |
| P_main | p | inter | 1,839 | 0.378 [0.306, 0.459] | 0.636 | 0.368 (1,658) |
| P_main | p | solv | 1,839 | 0.303 [0.241, 0.370] | 0.659 | 0.298 (1,658) |
| P_main | p | barrier | 1,839 | 0.348 [0.269, 0.430] | 0.803 | 0.318 (1,658) |
| test_all | all | strain | 8,170 | 0.852 | 0.842 | 0.824 (7,279) |
| test_all | all | inter | 8,170 | 0.638 | 0.780 | 0.617 (7,279) |
| test_all | all | solv | 8,170 | 0.500 | 0.701 | 0.491 (7,279) |
| test_all | all | barrier | 8,170 | 0.787 | 0.841 | 0.751 (7,279) |
| test_all | p | strain | 2,124 | 0.394 | 0.852 | 0.388 (1,881) |
| test_all | p | inter | 2,124 | 0.378 | 0.675 | 0.368 (1,881) |
| test_all | p | solv | 2,124 | 0.299 | 0.681 | 0.294 (1,881) |
| test_all | p | barrier | 2,124 | 0.366 | 0.816 | 0.349 (1,881) |

- 예전 GNN3(`gnn3/results/eval/metrics.csv`)는 다른 짝 목록(7,970짝, g2 입력)이라 같은 짝 비교가 아니다. 참고용이다.
- P_main = 기준 짝(ref_kind reference) 전부(D-10). [ ] = core bootstrap 95 % 구간(para).

