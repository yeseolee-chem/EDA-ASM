# ESPB — 결정 · 편차 · 버그 기록

사용자 지시(2026-11-16): "이대로 진행. 모든 분기는 추천옵션으로 진행(질문 금지), claude.md 준수." 지시서 STOP 지점에서도 추천 옵션을 골라 여기에 적고 진행한다.

## 실행 전 결정 (2-3절 표 밖의 세부, 실행 전에 기록)

| # | 내용 | 결정과 이유 |
|---|---|---|
| DEC-1 | E-A NN의 cv 단계 | inner fold마다 새 프로세스 하나(seed 22)로 돈다. 12절 자원 표의 "tune 1 + cv 4 + final 5 프로세스씩"을 따른 것이다. 원본 `d1_f0.py`는 cv를 한 프로세스에서 돌렸다. E-B의 cv는 지시서대로 `d1_train`과 같이 한 프로세스다 |
| DEC-2 | S0 job 스크립트 | `jobs/s0.slurm`을 더 만들었다(2-2절 표에 없음). 계산 노드에서 디스크 여유를 재고 env_espley로 `s0_start.py`를 부른다. GNN3R2의 `s0_gr.slurm`과 같은 방식이다 |
| DEC-3 | E-A unit 파일 | 지시서대로 test와 OOF를 한 parquet(role 열)에 둔다. C-4를 지키려고 `collect.py`는 pyarrow 행 필터(role == "oof")로 OOF 행만 먼저 읽고, `C_selection.csv`를 쓴 뒤에야 test 행을 읽는다 |
| DEC-4 | P-9 세부 | 무작위 500 반응(seed 20261019)으로 fit하고 fold 0 test 반응을 예측해 비교한다. Ridge는 alpha = 1, SVR은 C·epsilon·gamma 기본값이다 |
| DEC-5 | 합성 모델 ESPB | 타깃마다 고른 후보의 test 짝 행을 `results/C_espb.parquet`(model "ESPB", candidate 열)로 저장한다 |
| DEC-6 | `eb_train.py`에서 뺀 것 | XGB 경로, arm P·B·B_ref(f0 차감), group·SKIP 처리, `cmd_collect`. 모두 0-3절과 5-2절에서 쓰지 않는 부분이다. 남긴 함수(`Scale`, 커널, grid 작업자, `_tune_grid`의 Ridge·KRR·SVR, `fit_final`, NN 모델과 tune·cv)는 원본 그대로다 |
| DEC-7 | `eb_common.write_atomic` | `gr_common`의 helper에 callable(tmp_path) 형태를 더했다(`rev5_common.write_atomic`과 같음). 복사한 학습 코드가 parquet을 이 형태로 쓴다 |
| DEC-8 | 보조 비교의 Holm | 같은 비교·집합·부분집합의 타깃 4개를 한 묶음으로 보정하고 `holm_scope` 열에 적는다. 주 비교의 Holm(타깃 4개)과 따로 표시한다 |
| DEC-9 | 학습 job 묶음 | `submit_train.sh`가 19칸을 한 번에 제출한다(EA 격자 2, EA NN 4, EB 격자 2, EB SVR 4, EB NN 4, collect 1, compare 1, figures+report 1) |

## 실행 전 코드 검토 (2026-11-16, 검토자 4 + 검증자 4, 사전 등록 전)

확인된 결함을 사전 등록 전에 고쳤다(모두 새 코드).

| # | 파일 | 내용 | 수정 |
|---|---|---|---|
| R-1 | `ea_train.py`, `eb_train.py` | A-2·A-4·B-2·B-4 실패를 기록한 unit을 같은 명령으로 다시 내면 "cached"로 건너뛰어 gate가 통과로 바뀌었다(M-1만 남아 있었음) | 기록된 실패는 다시 돌 때도 STOP(`load_rec`, `cached`) |
| R-2 | `collect.py` | inner 점수가 NaN인 후보가 오류 없이 순위 꼴찌가 되어, 대표가 후보 9개에서 뽑힐 수 있었다. C-3는 test 열의 NaN만 봤다 | C-3에 unit json의 gate 기록(A-2·A-4·B-2·B-4·M-1)과 inner 점수·fold 점수의 유한성을 더함, inf도 잡음 |
| R-3 | `compare.py` | 주 비교의 한 줄 해석을 보정 전 95 % CI로 정했다 | 사전 등록한 Holm p < 0.05와 d의 부호로 정함 |
| R-4 | `eb_train.py` | seed 5개의 M-1 값을 Python `max`로 합쳐 NaN이 첫 값이 아니면 빠졌다(검증자는 실제로 일어나지 않는다고 봄) | `np.max`로 합침 |

검증자가 실제 결함이 아니라고 본 것(그대로 둠): `P_report.json`의 P-2 `ok`가 문자열 "True"로 저장됨(아무도 읽지 않음, `all_pass`는 올바름), `submit_train.sh`에 squeue 검사 없음(제출 전 사람이 확인하는 단계, 지시서 12절).
