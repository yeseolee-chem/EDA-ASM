# GNN3R — 결정 · 편차 · 버그 기록

사용자 지시(2026-11-13): "명령서대로 진행. 모든 분기는 추천옵션대로 진행. claude.md 준수." 지시서 STOP 지점에서도 추천 옵션을 골라 여기에 적고 진행한다.

## 실행 전 (코드 검토, 2026-11-14)

새 코드(`analysis/gnn3r/`)를 실행 전에 검토(검토자 4 + 검증자 4)했다. 확인된 결함 4개를 실행 전에 고쳤다(기존 코드는 손대지 않음).

| # | 파일 | 내용 | 수정 |
|---|---|---|---|
| R-1 | `check_cv.py` | K-2가 학습 GPU를 확인하지 않음(지시서 12: K·W는 RTX 3090에서만) | fold json의 `environment.gpu`에 "RTX 3090"이 없으면 K-2 실패 |
| R-2 | `train_final_r.py` | W-1이 GPU를 확인하지 않고, 판정 전에 `final_report.json`을 `results/final/`로 복사 | W-1에 RTX 3090 조건 추가, 통과할 때만 복사 |
| R-3 | `evaluate_r.py` | 표를 제자리에 씀(지시서 12: 임시 파일 → `os.replace`) | `write_atomic`으로 씀 |
| R-4 | `check_data.py` | 출력이 있어도 다시 돎(지시서 12) | 입력 sha256이 같고 통과한 `D_check.json`이 있으면 건너뜀 |

## A 단계

### BUG-1 (A-2 판정 코드, 첫 실행 job 1076283)

- 증상: A-2가 2,667행 모두 "다름"으로 셌지만, 기록된 행마다 저장값과 재계산값이 같았고 job_id는 NaN이었다.
- 원인: 다른 행이 없어 선택 결과가 빈 DataFrame이었는데, 빈 DataFrame에 `.assign(stored=Series)`를 하면 pandas가 그 Series의 index로 행을 새로 만든다(2,667행, job_id NaN). 실제로는 재계산값 = 저장값 100 %(A-2 통과)였다.
- 수정: 비교를 정렬된 numpy 배열로 하고, 다른 행만 골라 표를 만든다. 데이터·다른 판정에는 영향 없음.

### DEC-1 (A-3 STOP → 추천 옵션: X = H 기준 행은 c3 빈칸이 맞음)

- 상황: Bp `need` 922행 중 4행(T01395, T01396, T02051, T02686)이 수정 뒤에도 c3 = "". 지시서 A-3은 "922개 모두 cis/trans".
- 사실: 네 행 모두 X = H 기준 행(R1 = H, pos `ref_direct`; azomethine ylide/Bp exo·endo, nitrone/Bp endo, carbonyl ylide/Bp endo)이다. C3에 치환기가 없으므로 c3는 정의되지 않는다. 원래 `c3_orientation`도 ""를 내고, 참조 목록 `ref_reactions_est.csv`도 이 네 행의 `c3_est`를 비워 두었다(`family_basis` D1PAIR_family, 직접 계열의 H 기준). 지시서 B-2도 "H 행의 c3·orient는 비어 있을 수 있음"이라고 적고 있다.
- 선택지: (a, 추천) A-3을 C3 치환기가 있는 Bp 행 918개에 적용하고(전부 cis/trans), H 기준 4행은 c3 = ""여야 통과로 판정한다. 데이터는 바뀌지 않는다. (b) H 행에도 c3를 억지로 붙인다(정의 없음, D1PAIR 직접 짝 규칙과 어긋남). (c) STOP 유지.
- 결정: (a). `rowattr_r.py` A-3 판정에 반영(`n_bp_substituted`, `n_bp_h_reference` 기록).
