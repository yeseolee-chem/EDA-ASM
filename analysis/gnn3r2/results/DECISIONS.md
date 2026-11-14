# GNN3R2 — 결정 · 편차 · 버그 기록

사용자 방침: 모든 분기는 추천 옵션으로 진행하고 여기에 적는다(지시서 STOP 지점 포함). 2026-11-14 지시: "재학습 후 push".
코드는 `analysis/gnn3r/`에서 복사했다(`results/COPY.json`, 13개 파일 + 참조 결과 7개, sha256 모두 원본과 같음).

## 실행 전 결정

### DEC-1 (G-2를 문자 그대로 적용하면 항상 실패 → 추천: 계열 T 안의 "변화"를 검사)

- 지시서 G-2: "a가 T에 속하는 짝 ⊆ 0-4절의 추가 짝 6개".
- 사실: trans 계열(azomethine ylide / Bp / R1 / cfg t / orient "" / endo / trans)은 GNN3R에서 이미 anchor 계열이며, b = T07195인 짝 48개(GRA00537 … GRA00585, T10104·T10116 제외)가 있다. 0-4절도 "b = T07195는 그대로"라고 적어 이 계열이 이미 있음을 전제한다. 그러면 a ∈ T인 짝에 그 48개가 늘 포함되므로 문자 그대로의 G-2는 반드시 실패한다.
- 선택지: (a, 추천) T 안의 변화를 검사한다. 새 − 옛 ⊆ 추가 짝 6개(빠진 것은 모두 P-4 v2 mismatch로 버려진 것), 옛 − 새 = {T10104 → T10116}, 양쪽에 있는 짝은 G-1과 같이 모든 열이 바이트 단위로 같다. (b) STOP.
- 결정: (a). `diff_gnn3r.py`가 문자 그대로의 판정 결과도 `G_report.json`에 함께 적는다(참고용).

### DEC-2 (2-3절 표 밖의 수정: `jobs/s0_gr.slurm`)

- 복사한 S0 job이 `origin/g2r`를 fetch하고 `analysis/gnn3r/s0_start.py`(예전 시작 조건)를 실행한다. 2-3절 표는 `jobs/`를 "바꾸지 않음"으로 두었지만, 이대로면 GNN3R2 1절이 아니라 GNN3R 1절을 검사한다.
- 결정: fetch 대상을 `origin/gnn3r`(1-1), 실행 파일을 `analysis/gnn3r2/s0_start.py`로 바꾼다. 다른 줄은 그대로다.

### DEC-3 (복사 순서: 1-5와 2-2의 순서)

- 1-5는 `$GR2`가 없거나 비어 있어야 하는데, 2-2의 복사는 `$GR2/ref_gnn3r/`, `$GR2/B/`에 파일을 쓴다.
- 결정: `setup_r2.py`를 두 번 부른다. `code`(코드와 `ref_gnn3r/`, worktree 안만)를 먼저 하고, S0 통과 뒤 `$GR2` 폴더를 만들고 `scratch`(rowattr, P-4 캐시)를 복사한다. 두 결과 모두 `results/COPY.json`에 적는다.

### DEC-4 (S-5의 자리)

- 2-3절은 `splits_r.py`를 "바꾸지 않음"으로 두었지만 5절에 새 검사 S-5(fold 1을 뺀 fold의 test 목록 = GNN3R)가 있다.
- 결정: S-5는 새 파일 `diff_gnn3r.py splits`로 한다(GNN3R과의 대조는 모두 이 파일). S job에서 `splits_r.py ::: diff_gnn3r.py splits`로 이어 돌린다. GNN3R test 목록은 `ref_gnn3r/S_folds_r.csv`의 fold_T1에서 만든다.

### DEC-5 (c3_core 구현 세부)

- core의 원자 순서를 정규화(`CanonicalRankAtoms` → `RenumberAtoms`)한 뒤 배치한다. 같은 core가 입력 원자 순서에 따라 다른 배치를 받지 않게 하고, 지시서 자원 표의 "같은 정규 core SMILES는 한 번만 계산"을 그대로 따른다.
- C3 치환기가 있는데 C4 기준 치환기를 하나로 정할 수 없거나(또는 X 가지가 고리에 닿으면) c3 = "?"로 두어 A-3에서 멈추게 한다(원래 코드는 ""를 냄).
- A-3에 두 조건을 더 적는다: 같은 core는 같은 c3, 서로 거울상인 core는 같은 c3(지시서 A-3 문구).

## 실행 전 코드 검토 (2026-11-14, 검토자 3 + 검증자 3)

확인된 결함 3개를 실행 전에 고쳤다(모두 새 코드).

| # | 파일 | 내용 | 수정 |
|---|---|---|---|
| R2-1 | `rowattr_r.py` `mirror_smiles` | 키랄 표기를 뒤집은 뒤 입체 인식을 다시 하지 않고 SMILES를 써서, 거울상 core의 정규 SMILES와 다를 수 있었다 → A-3 거울상 검사가 아무것도 비교하지 않고 통과할 수 있음 | 쓰고 다시 읽어 정규화(round trip) |
| R2-2 | `delete_gnn3r.py` | 결과가 commit·push되지 않아도(작업 트리의 보고서만 통과) 삭제 조건이 참이 될 수 있었다 | HEAD = origin/gnn3r2, `analysis/gnn3r2`·지시서에 commit 안 된 변경 없음, origin/gnn3r2에 RUN_REPORT·gnn3r2.json·F_pairs_r.csv·지시서가 있음, manifest 빠진 경로 0을 조건에 더함 |
| R2-3 | `delete_gnn3r.py` | 11절에 없는 `gr2_prep`(GNN3R2 S0·복사 로그)도 지우게 되어 있었고, 다시 제출하면 삭제 기록을 덮어쓸 수 있었다 | 대상은 `$GR`만. rmtree 전에 크기를 기록하고, 끝난 기록은 다시 쓰지 않음 |
