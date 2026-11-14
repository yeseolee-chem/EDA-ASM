# GNN3R2 실행 보고 — c3 판정 오류 2건을 고친 짝으로 GNN3 다시 학습

- 지시서: `docs/specs/GNN3R2.md` · 브랜치 `gnn3r2` · 결과 commit `2890f2c5`
- scratch `/gpfs/tmp_cpu2/yeseo1ee/gnn3r2` (cv/, final/ = .pt), 결과 `analysis/gnn3r2/results/`, manifest `results/gnn3r2.json`, GNN3R 대조 사본 `analysis/gnn3r2/ref_gnn3r/`
- 데이터셋 고정 파일: `results/F_pairs_r.csv` sha256 `9d75fae8b7ba39ab7d83112342a7ddd299017ad7eb3b05f97a4409deec4f3455` (`B_label_report.json`에 기록, gnn3 로더가 이 값으로 확인)
- 결정·편차: `results/DECISIONS.md`

## 1. 단계별 판정

| 단계 | 판정 | 비고 |
|---|---|---|
| 1 시작 조건 | 통과 | 1-1 … 1-5 (GNN3R 1-2 … 1-10 포함), RDKit 2026.03.2, 여유 111 GB |
| 2 복사 | 통과 | 코드·참조 결과·scratch 파일 sha256 = 원본 (`COPY.json`) |
| A c3 다시 판정 | 통과 | A-1 … A-6; c3가 바뀐 행 2개 |
| B 짝 | 통과 | B-0 … B-7, B-R |
| G GNN3R 대조 | 통과 | G-1 … G-4 |
| S 분할 | 통과 / S-5 통과 | S-1 … S-5, sklearn 1.3.1 |
| D 데이터 | 통과 | D-1 … D-4 |
| K 교차검증 | 통과 | K-1 … K-3 (RTX 3090) |
| E 평가 | 통과 | E-1: 8,175짝 평가 |
| W 최종 모델 | 통과 | W-1: 8,344짝 학습 |

## 2. c3 수정 (D-12)

- 규칙 `c3_core`: C3·X 첫 원자(xs)·C4·C4 기준 치환기는 예전과 같이 고른다(B′는 메틸 규칙). X 가지를 xs 하나로 줄이고 xs를 CH₃로 바꾼 뼈대(core, 다른 입체 표기 유지)를 정규 원자 순서로 놓고, seed 7·11·23·42·97로 각각 EmbedMolecule + MMFF 배치한다. 고리 5원자 법선(SVD)에서 u·v > 0이면 cis. 5개가 모두 같아야 한다. X = H 행은 빈칸.
- `need` 2,667행: C3 치환기 있음 2,653 (cis/trans {'trans': 1355, 'cis': 1298, '': 14}), X = H 14행(모두 `ref_direct`: True).
- 서로 다른 core 72개(지시서 로컬 72), seed 5개가 모두 같은 행 2,653 / 2,653, 서로 거울상인 core 쌍 36개(c3 충돌 0), 같은 core 안 c3 충돌 0.
- GNN3R c3와 다른 행 2개 (A-5 기대: T10104 cis→trans, T10116 cis→trans):

| job_id | 위치 · X | c3 GNN3R → GNN3R2 | seed 7 · 11 · 23 · 42 · 97 | core SMILES |
|---|---|---|---|---|
| T10104 | o · SMe | cis → trans | trans · trans · trans · trans · trans | `C[C@H]1N(C)C[C@@]2(C)[C@H]3CC[C@H](C3)[C@@]12C` |
| T10116 | o · tBu | cis → trans | trans · trans · trans · trans · trans | `C[C@H]1N(C)C[C@@]2(C)[C@H]3CC[C@H](C3)[C@@]12C` |

## 3. 데이터셋

| 항목 | GNN3R | GNN3R2 기대 (0-4) | GNN3R2 실제 |
|---|---|---|---|
| 반응 | 8,589 | 8,594 | 8,594 |
| 짝 | 8,339 | 8,344 | 8,344 |
| 기준 짝 / anchor 짝 | 6,978 / 1,361 | 6,978 / 1,366 | 6,978 / 1,366 |
| fold 1 test 짝 | 1,700 | 1,705 | 1,705 |
| `p4_dropped` 행 | 16 | 11 | 11 |

**새 짝 6개** (0-4절)

| a → b | 서버 | P-4 v2 |
|---|---|---|
| T10104 → T07195 | 데이터셋 안 (`GRA00587`) | mirror_match |
| T10116 → T07195 | 데이터셋 안 (`GRA00590`) | mirror_match |
| T10047 → T10056 | 데이터셋 안 (`GRA00536`) | mirror_match |
| T10067 → T10056 | 데이터셋 안 (`GRA00537`) | mirror_match |
| T10092 → T10056 | 데이터셋 안 (`GRA00538`) | match |
| T10111 → T10056 | 데이터셋 안 (`GRA00539`) | mirror_match |

**GNN3R 대조** (`results/G_report.json`, `G_diff_gnn3r.csv`)

| # | 판정 | 내용 |
|---|---|---|
| G-1 | 통과 | 계열 T 밖 짝 8,289개: GNN3R과 (a, b)·값 모두 같음 (GNN3R에만 0, GNN3R2에만 0, 값 차이 0) |
| G-2 | 통과 | 계열 T 안: 추가 6, 빠짐 [['T10104', 'T10116']], P-4로 빠진 기대 짝 [], 값 차이 0 (DEC-1 기준) |
| G-3 | 통과 | `B_row_status`가 다른 행: T10047, T10056, T10067, T10092, T10104, T10111, T10116 |
| G-4 | 통과 | 짝 8,344 (기대 8,344), 반응 8,594 (기대 8,594), 기준 짝 6,978 |
| S-5 | 통과 | fold 0·2·3·4 test 목록 = GNN3R |

**fold_T1별** (GNN3R → GNN3R2)

| | anchor | 기준 | 합 |
|---|---|---|---|
| -1 | 169 → 169 | 0 → 0 | 169 → 169 |
| 0 | 267 → 267 | 1,415 → 1,415 | 1,682 → 1,682 |
| 1 | 301 → 306 | 1,399 → 1,399 | 1,700 → 1,705 |
| 2 | 171 → 171 | 1,243 → 1,243 | 1,414 → 1,414 |
| 3 | 36 → 36 | 1,308 → 1,308 | 1,344 → 1,344 |
| 4 | 417 → 417 | 1,613 → 1,613 | 2,030 → 2,030 |

**쌍극자별** (GNN3R → GNN3R2)

| | anchor | 기준 | 합 |
|---|---|---|---|
| nitrone | 261 → 261 | 1,985 → 1,985 | 2,246 → 2,246 |
| azide | 261 → 261 | 1,535 → 1,535 | 1,796 → 1,796 |
| nitrile oxide | 44 → 44 | 1,663 → 1,663 | 1,707 → 1,707 |
| azomethine ylide | 484 → 489 | 964 → 964 | 1,448 → 1,453 |
| carbonyl ylide | 311 → 311 | 831 → 831 | 1,142 → 1,142 |

**위치별** (GNN3R → GNN3R2)

| | anchor | 기준 | 합 |
|---|---|---|---|
| p | 326 → 326 | 1,839 → 1,839 | 2,165 → 2,165 |
| m | 326 → 326 | 1,879 → 1,879 | 2,205 → 2,205 |
| o | 388 → 393 | 1,708 → 1,708 | 2,096 → 2,101 |
| direct | 321 → 321 | 1,552 → 1,552 | 1,873 → 1,873 |

9,252행 status: in_dataset 8,594, not_usable 635, p4_dropped 11, singleton 6, no_family 6

## 4. P-4 입체 검사

- 검사한 짝 8,355 (반응 8,605, 이번에 새로 계산 0, 고리 구성 실패 0; 링 정보는 GNN3R 캐시 사본)
- v2: no_relative_info 3,841, match 2,422, mirror_match 2,081, mismatch 11
- mismatch로 버린 짝 11: T01318→T01264 (anchor), T09972→T01302 (reference), T10078→T01421 (reference), T10523→T04796 (anchor), T10560→T04796 (anchor), T10596→T04796 (anchor), T10683→T04955 (anchor), T10719→T04955 (anchor), T10744→T04955 (anchor), T10748→T04955 (anchor), T10756→T04955 (anchor)

## 5. 학습

| fold | 고른 칸 (층×폭) | epoch | 검증 MAE strain / inter / solv / barrier | 학습 / test 짝 | 시간 (h) | GPU |
|---|---|---|---|---|---|---|
| 0 | 6×64 | 52 | 0.817 / 0.589 / 0.501 / 0.722 | 6,662 / 1,682 | 1.18 | NVIDIA GeForce RTX 3090 |
| 1 | 4×64 | 55 | 0.973 / 0.757 / 0.611 / 0.876 | 6,639 / 1,705 | 1.00 | NVIDIA GeForce RTX 3090 |
| 2 | 6×128 | 40 | 1.021 / 0.799 / 0.617 / 0.940 | 6,930 / 1,414 | 1.35 | NVIDIA GeForce RTX 3090 |
| 3 | 6×128 | 48 | 1.038 / 0.733 / 0.618 / 0.939 | 7,000 / 1,344 | 1.47 | NVIDIA GeForce RTX 3090 |
| 4 | 6×64 | 18 | 1.012 / 0.773 / 0.624 / 0.945 | 6,314 / 2,030 | 0.60 | NVIDIA GeForce RTX 3090 |

- 최종 setting: 6층 × 64폭, epoch 48 (fold best epoch [18, 40, 48, 52, 55], 칸 선택 수 {'{"n_layers": 6, "width": 64}': 2, '{"n_layers": 4, "width": 64}': 1, '{"n_layers": 6, "width": 128}': 2})
- 최종 모델: seed 5개, 8,344짝, 0.72 h, NVIDIA GeForce RTX 3090; `.pt` = `/gpfs/tmp_cpu2/yeseo1ee/gnn3r2/final/gnn3_final__seed{0..4}.pt`

## 6. 성능 (교차검증 test 예측, kcal/mol)

| 집합 | 위치 | target | n | MAE | r | GNN3R MAE (n) | GNN3R r |
|---|---|---|---|---|---|---|---|
| P_main | all | strain | 6,978 | 0.813 | 0.856 | 0.811 (6,978) | 0.855 |
| P_main | all | inter | 6,978 | 0.638 | 0.778 | 0.635 (6,978) | 0.779 |
| P_main | all | solv | 6,978 | 0.509 | 0.684 | 0.506 (6,978) | 0.690 |
| P_main | all | barrier | 6,978 | 0.747 | 0.857 | 0.741 (6,978) | 0.855 |
| P_main | p | strain | 1,839 | 0.387 [0.317, 0.461] | 0.832 | 0.374 (1,839) | 0.838 |
| P_main | p | inter | 1,839 | 0.387 [0.316, 0.467] | 0.612 | 0.378 (1,839) | 0.636 |
| P_main | p | solv | 1,839 | 0.314 [0.250, 0.383] | 0.610 | 0.303 (1,839) | 0.659 |
| P_main | p | barrier | 1,839 | 0.360 [0.288, 0.433] | 0.800 | 0.348 (1,839) | 0.803 |
| test_all | all | strain | 8,175 | 0.854 | 0.843 | 0.852 (8,170) | 0.842 |
| test_all | all | inter | 8,175 | 0.641 | 0.778 | 0.638 (8,170) | 0.780 |
| test_all | all | solv | 8,175 | 0.501 | 0.696 | 0.500 (8,170) | 0.701 |
| test_all | all | barrier | 8,175 | 0.790 | 0.843 | 0.787 (8,170) | 0.841 |
| test_all | p | strain | 2,124 | 0.404 | 0.851 | 0.394 (2,124) | 0.852 |
| test_all | p | inter | 2,124 | 0.387 | 0.655 | 0.378 (2,124) | 0.675 |
| test_all | p | solv | 2,124 | 0.309 | 0.637 | 0.299 (2,124) | 0.681 |
| test_all | p | barrier | 2,124 | 0.374 | 0.816 | 0.366 (2,124) | 0.816 |

- GNN3R 값(`ref_gnn3r/metrics.csv`)은 참고용이다. 짝 목록이 6짝 다르고(fold 1) 모든 fold를 다시 학습했다.
- P_main = 기준 짝(ref_kind reference) 전부(D-10). [ ] = core bootstrap 95 % 구간(para).

