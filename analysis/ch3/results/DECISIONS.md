# CH3-MODEL 결정 기록 (docs/specs/CH3_MODEL.md)

- **0-user (2026-11-10)**: 사용자가 "모두 종료. 알림도 종료. 80개 라벨 혼돈 방지를 위해 전부 삭제, git에서도 삭제. 첨부한 명령서 실행"이라고 했고, 이어서 "분기마다 추천 옵션으로 진행"이라고 했다. 분기마다 추천안으로 진행하고 여기에 적는다. STOP 조건(15절)에서는 멈추고 묻는다.
- **0-정리 (11-10 11:2x)**: RELABEL-PILOT-2ARM을 모두 지웠다.
  - 30분 점검 cron을 지우고 `scancel -u`로 모든 job을 취소했다.
  - scratch `relabel_2arm`과 conda env `relabel_crest`는 job 1070554로 지웠다.
  - worktree는 `git worktree remove`로 지웠다.
  - 브랜치 `relabel-2arm`(709e6677)은 로컬과 origin에서 지웠다.
- **2 브랜치**: `git fetch origin 2f9ffa35…` 뒤 `git worktree add -b ch3 …-ch3 2f9ffa35…`로 만들었다.
  - 명령서는 첨부 원문을 바이트 그대로 `docs/specs/CH3_MODEL.md`에 넣고 commit했다(858ed4e2, sha256 20fb95db…).
  - 원문은 세션 기록에서 job 1070553으로 추출했다.
- **1 S0 디스크(1-8)**: 계산 노드에서는 `mmlsquota`가 "GPFS is down on this node"로 실패한다. 로그인 노드는 sbatch와 git만 쓰기로 했으므로, S0 job이 `300 GB − du(/home1/yeseo1ee + /gpfs/tmp_cpu2/yeseo1ee)`로 여유를 계산한다(jobs/s0.slurm).
- **분할 선택**: 로그인에서 `sinfo`를 돌리지 않는다. CPU job은 `-p cpu1,cpu2`로 내서 SLURM이 먼저 빈 쪽을 고르게 한다.
- **3 L·M·F·T 자원 (deploy 실측 그대로)**:
  - L: 배열 0-7, 작업당 64코어, 반응당 4코어 × 16 병렬.
  - L report: 16코어.
  - F: slice 0-1, 각 32코어.
  - T: atoms 0-1, 각 32코어.
  - 체인은 S0(afterok, kill-on-invalid) → L → (afterany) report → (afterok) M → F → (afterany) merge → (afterok) T atoms → (afterany) gate ::: tokens 순서다.
  - job 번호: S0 1070561, L 1070562, Lrep 1070563, M 1070564, F 1070565, Fm 1070566, T 1070567, Tg 1070568.
- **5 타깃 바꾸기 방법**: 명령서 5절의 (a)와 (b)를 섞었다.
  - (a) `data_ch3.use_targets`가 프로세스 안에서 `fu2_common.TARGETS / I_BAR / I_CH9 / I_PRIM`을 바꾼다. fu2 `fit`·`loss_fn`·`predict`가 호출 시점에 읽는 상수다.
  - (b) `train_tf.run_unit` / `train_final`은 쓰기 가드(dep_common: `$CH3/lf`만 허용)와 단위 이름(`lf`) 때문에 그대로 부를 수 없다. 그래서 타깃 목록을 인자로 받는 사본을 `train_ch3.py`에 두었다. fu2 함수와 deploy 도우미(`leak_check`, `kcal`, `std_dict`, `std_sha`, `choose_final`, `para_mae`)는 import해서 그대로 부른다.
  - C-3′(10타깃 모드로 deploy fold 0 재현)와 C-T가 이 사본을 검사한다.
- **4 V 내용 비교**: deploy scratch는 지워졌으므로, sha256이 다를 때는 deploy commit에 남은 기록과 비교한다.
  - 행 목록은 `L_status.csv`의 `.done`과 `F_report.not_ok`로 비교한다.
  - 토큰은 `T_manifest.csv`의 파일별 sha256과 비교한다(같으면 차 0).
  - 182열은 sha256이 같은 토큰 파일 안의 전역 182값과 비교한다(≤ 1e-6).
  - 이 방법으로 비교할 수 없는 차이는 실패(STOP)로 본다.
- **9 평가 정의**: 명령서가 집합을 명시하지 않은 보조 지표(짝 MAE, 부호, 원인 분석)의 주 정의는 9-1의 P_main 공통 짝이다. 예측의 주도 채널과 유형은 그 모델 자신의 barrier 부호로 판정한다.
  - selftest가 lf 합으로 9-3 참고값을 재현하는지 보고, 다르면 다른 정의(P_sens reference, reference 전체, 시험 전체, 실제 barrier 부호)를 시험해 일치하는 정의를 보고한다.
  - 주 정의는 ch3 결과를 보기 전에 PREREG에서 고정한다.
- **규칙 위반 기록 (11-10 11:5x, 나)**: selftest json을 보려고 login에서 `git status` 뒤에 `python3 --version`(출력 없이 즉시 종료)과 `grep`을 한 줄로 실행했다. 계산이나 파일 쓰기는 없었다. 이후 login에서는 sbatch와 git만 쓰고, 파일은 Read로 본다.
- **S0 통과 (11-10 11:55, job 1070561)**: 1-1..1-8 모두 통과. 디스크 여유는 du로 187 GB 사용 → 여유 113 GB(≥ 30).
- **L · M 통과 (12:06)**: `.done` 8,616, 진동수 실패 0, L-2 순열 오차 0 Å, L-3 차이 3(deploy와 같음), L-4 최대 7.4e-7 cm⁻¹. M 레코드 8,616, 전하 ≠ 0 0개.
- **자동 점검 · 알림 (사용자 11-10 "자동점검 해줘", "알림도 켜줘")**: session cron 94ef11e5가 매시 17분과 47분에 로그와 결과를 Read로 점검한다. 새 사건(STOP, 주요 단계 완료, 전체 완료)만 PushNotification으로 알리고, 알린 사건은 `$CH3/monitor_state.json`에 적는다. cron은 7일 뒤 자동 만료된다.
- **최대 병렬 (사용자 11-10 "최대 병렬로 진행")**:
  - 이미 대기열에 있는 F(명령서 고정 `slice @I 2`)와 T atoms(0-1)는 그대로 둔다. 대기 job을 취소하고 다시 이으면 의존 사슬이 끊기는데, 아낄 시간은 몇 분뿐이다.
  - 아직 제출 전인 K 이후 사슬은 바꿨다. I-1(K 시험 짝 재예측)은 K 단위만 있으면 되므로 collect 직후 별도 GPU job으로 E · W와 나란히 돌린다(launch_k.py).
  - C-8과 C-3′는 GPU job 2개로 나란히, K는 5 fold를 동시에(`--array=0-4%5`) 돌린다.
- **F · T 통과 (12:21)**:
  - X182 8,504행. Q-a 실패 0, Q-b 실패 0, 토큰 8,502, T-a 통과.
  - 원소 · 열 목록 · d_in 29가 g2 T_report와 같다.
- **V 1차 실패 → 내용 비교 보강 (12:22, job 1070701, 보고 후 수정)**:
  - V-1, V-2(labels sha 같음), V-3(meta sha 같음), V-6(짝 목록 sha 같음)은 통과했다.
  - V-4(X182 sha 다름)도 통과로 나왔다. 그러나 토큰이 바이트 단위로 같은 3,900개 반응의 182열만 비교한 결과(최대 차 0)였다.
  - V-5는 실패했다. 토큰 파일 8,502개 중 4,602개의 sha256이 deploy T_manifest와 다르다.
  - deploy 원본이 지워져 이 반응들은 값으로 비교하지 못했다.
  - 같은 구조 · 같은 코드인데 다른 이유: deploy Q_parity에서도 deploy lf 182열이 g2와 비트 단위로 같은 반응은 4,145/8,499뿐이었다(차이 ≤ 1e-6). 즉 xtb 값은 1e-6 수준까지는 재현되지만 비트까지는 아니다(노드 · 실행마다 마지막 자리).
  - 추천안: 명령서 4절의 "내용 비교"를 deploy가 commit한 기록으로 수행한다(`verify_inputs.py bridge`). Q_parity.csv에는 반응마다 |deploy lf − g2|의 최대값(182열, 토큰 num · pos · glob, 순열 포함)이 있고, g2 파일은 Q_parity.json의 sha256과 같은지 확인한다.
    - 기록이 0인 반응: |새 − deploy| = |새 − g2|를 정확히 계산한다.
    - 그 밖의 반응: |새 − g2| + 기록값을 상한으로 쓴다.
    - perm_symmetric_map 7개: 기록을 재현하는지 본다.
    - g2가 없는 deploy lf_only 3개(최대): 비교 불가 목록으로 남긴다.
  - 기준은 모두 ≤ 1e-6이고, 넘으면 STOP이다. 상세는 `results/V_bridge.csv`에 남긴다.
  - 1차 V_inputs.json은 다시 실행하면서 덮어쓴다. 사슬(VC → C8 · C3r → Cm → R)은 launch_c.py로 다시 낸다.
- **V 통과 (12:30, job 1070711, commit 98bae7cf)**:
  - labels · meta · 짝 목록은 sha256이 같다.
  - X182 · 토큰은 sha256이 달라 내용 비교로 판정했다.
    - 바이트가 같은 토큰 3,900개: 차 0.
    - g2 다리 정확값 1,362개: 최대 7.4e-7. 이 값은 L-4의 진동수 차 최대 7.4e-7 cm⁻¹와 같다. 즉 차이는 진동수에서 나온 B6 열이다.
    - 상한 3,238개: 최대 8.6e-7.
    - 기록 재현 2개. 비교 불가 2개(deploy lf_only).
- **C 결과 (13:05, job 1070714) → STOP (명령서 15절 6: C 검사 하나라도 실패)**:
  - 통과: C-Y(항등 잔차 최대 0.096 ≤ 0.2), C-D, C-5, C-T, C-4, C-8.
  - C-3′(job 1070713, RTX 3090) 실패: 선택 칸은 deploy와 같은 3 × 64다(i 통과). 최적 epoch는 102 대 deploy 77로 차 25다(ii 실패, 기준 ≤ 2).
  - fold 0 시험 para MAE 상대 차(367짝, iii 실패, 기준 ±2 %):

    | 타깃 | 상대 차 | 판정 |
    |---|---|---|
    | d1 | +3.7 % | 실패 |
    | d2 | +2.0 % | 통과 |
    | elst | −1.0 % | 통과 |
    | pauli | +0.6 % | 통과 |
    | oi | +0.7 % | 통과 |
    | disp | +3.9 % | 실패 |
    | cpcm | +0.8 % | 통과 |
    | cds | −2.9 % | 실패 |
    | c_ghost | +11.7 % | 실패 (MAE 0.054 → 0.060) |
    | barrier | +1.8 % | 통과 |

  - 같은 시험 짝, 같은 y(차 0)다. 최대 |Δyhat|는 6.4 kcal/mol(pauli)이다.
  - 튜닝 검증값도 deploy와 1–3 % 다르다(예: 2 × 64에서 0.3629 대 0.3594).
  - 가설: 버그가 아니라 입력의 1e-7 수준 차(B6 진동수 열)가 GPU float32 학습에서 증폭된 것이다.
    - deploy C-3은 같은 입력에서 LADDER 단위를 튜닝 검증값까지 그대로 재현했다.
    - deploy NULL_gc는 seed만 바꿔도 상대 차 95 % 상한이 수 %까지 갔다.
  - 진단(비파괴, $CH3/diag)을 돌린다:
    - D1(job 1070796): deploy `train_tf.py c3`를 그대로 다시 실행한다(LADDER g2 입력 → L__g2__S__TF__f0 재현). 지금 GPU · 코드가 결정적인지 본다.
    - D2(job 1070797): `diag_c3r.py null`. 같은 재생성 입력에서 seed만 +1000으로 바꾼 fold 0이 C-3′ 수치를 얼마나 벗어나는지 본다.
  - PREREG · K는 내지 않았다(사슬이 C 병합에서 멈춤). 사용자 결정을 기다린다.
- **결과 push (사용자 13:1x "결과 push")**: C 단계 결과(C_tests · C_part_cpu · C3r_regress · C8_smoke)와 이 기록을 commit하고 `git push -u origin ch3`.
  - `E3_selftest.json`은 지금 eval_ch3.py보다 앞선 판(정의 변경 전)이라 commit하지 않는다. PREREG job에서 다시 만든다.
- **GPU**: C-3′, C-8, K, W, I는 명령서대로 RTX 3090(gpu1)에서만 돌린다. 비어 있지 않으면 대기열에서 기다린다(deploy 때처럼 A10으로 옮기지 않는다).
