2026-10-04T07:02:46+00:00 T00019 ts
  사유: autodE exception: ModuleNotFoundError: No module named 'matplotlib'
  처리: 환경 결함(autodE 의존 패키지 matplotlib·pillow·loky 누락)으로 판정, venv 에 설치 후 재실행. 실패 기록은 .envfix_ts 로 보존
2026-10-04T07:02:47+00:00 T00879 ts
  사유: autodE exception: ModuleNotFoundError: No module named 'matplotlib'
  처리: 환경 결함(autodE 의존 패키지 matplotlib·pillow·loky 누락)으로 판정, venv 에 설치 후 재실행. 실패 기록은 .envfix_ts 로 보존
2026-10-04T07:02:47+00:00 T01165 ts
  사유: autodE exception: ModuleNotFoundError: No module named 'matplotlib'
  처리: 환경 결함(autodE 의존 패키지 matplotlib·pillow·loky 누락)으로 판정, venv 에 설치 후 재실행. 실패 기록은 .envfix_ts 로 보존
2026-10-04T07:02:47+00:00 T01206 ts
  사유: autodE exception: ModuleNotFoundError: No module named 'matplotlib'
  처리: 환경 결함(autodE 의존 패키지 matplotlib·pillow·loky 누락)으로 판정, venv 에 설치 후 재실행. 실패 기록은 .envfix_ts 로 보존
2026-10-04T07:02:47+00:00 T10385 ts
  사유: autodE exception: ModuleNotFoundError: No module named 'matplotlib'
  처리: 환경 결함(autodE 의존 패키지 matplotlib·pillow·loky 누락)으로 판정, venv 에 설치 후 재실행. 실패 기록은 .envfix_ts 로 보존
2026-10-04T07:46:04+00:00 T00019 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  처리: 환경 결함 — OpenMPI 를 --disable-mpi-fortran 으로 빌드해 MPI_Type_match_size 가 MPI_ERR_ARG. Fortran 포함 재빌드 후 재실행. 기록은 .envfix2_ts 로 보존
2026-10-04T07:46:04+00:00 T00879 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  처리: 환경 결함 — OpenMPI 를 --disable-mpi-fortran 으로 빌드해 MPI_Type_match_size 가 MPI_ERR_ARG. Fortran 포함 재빌드 후 재실행. 기록은 .envfix2_ts 로 보존
2026-10-04T07:46:04+00:00 T01165 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  처리: 환경 결함 — OpenMPI 를 --disable-mpi-fortran 으로 빌드해 MPI_Type_match_size 가 MPI_ERR_ARG. Fortran 포함 재빌드 후 재실행. 기록은 .envfix2_ts 로 보존
2026-10-04T07:46:04+00:00 T01206 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  처리: 환경 결함 — OpenMPI 를 --disable-mpi-fortran 으로 빌드해 MPI_Type_match_size 가 MPI_ERR_ARG. Fortran 포함 재빌드 후 재실행. 기록은 .envfix2_ts 로 보존
2026-10-04T07:46:04+00:00 T10385 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  처리: 환경 결함 — OpenMPI 를 --disable-mpi-fortran 으로 빌드해 MPI_Type_match_size 가 MPI_ERR_ARG. Fortran 포함 재빌드 후 재실행. 기록은 .envfix2_ts 로 보존

## 지난 일 기록 (SPEC_D1_FIX1 §3 — 2026-10-05 에 PBS 로그 · ORCA 출력 머리부 · 표시 파일 시각 · 작업 기록에서 읽어 적음)

시각은 UTC (KST = UTC+9). 근거: nurion/pilot*.pbslog, 각 *_orca.out 의 "Host name" · "Working dir.: /var/tmp/pbs.<잡>.pbs",
timing.tsv, 표시 파일 mtime, /home01/e1977a01/.bash_history, 세션 기록. "추정"은 직접 읽지 못한 것.

### 1. 10/4 08:57 KST 의 1코어 실행 — 잡 24036745 `d1pilot`, node5778, PAR=12 · NC=1, 12행 전부
- 시작 2026-10-03T23:57:23Z (pilot.pbslog: "주의: ORCA 병렬이 안 된다(MPI_OK=0). 반응당 1코어, 동시 12 건", "par=12 cores/rxn=1").
- 원인: pilot.pbs 의 "MPI_OK=0 이면 반응당 1코어" 분기. SLURM_CPUS_PER_TASK 는 빠진 것이 아니라 그 분기가 1 로 넘겼다
  (pilot.pbs: SLURM_CPUS_PER_TASK="$NC"). 그때 nurion/mpi_status.txt 는 MPI_OK=0 이었다 (install2 · mpi_fix 가 병렬을 못 찾음).
  사용자에게 안내한 체인 `qsub nurion/mpi_solve.pbs → qsub -W depend=afterany:… nurion/pilot.pbs` 에서 mpi_solve 1차판이
  set -e 버그로 1분 만에 끝났고, afterany 라 파일럿이 그대로 시작했다.
- 고친 것 (FIX1 §2): pilot.pbs 의 1코어 분기를 없앴다 (MPI_OK=0 이면 멈춤). worker_pool.sh · run_task.sh · pilot.pbs · s0_host.sh 는
  반응당 코어가 config cpus_per_task(8)와 다르면 표시 없이 종료 코드 1 로 끝낸다.
- 행마다 끝난 시각과 이유:
  - T00879 01:27:22 · T01165 03:26:12 · T10385 05:02:20 · T01206 05:04:03 · T00019 05:33:45 — ts 단계가
    `ModuleNotFoundError: No module named 'matplotlib'` 로 .fail_ts (→ 아래 2 의 .envfix_ts)
  - T01914 · T01916 · T00836 · T02894 · T02935 — ts 단계 도중 qdel (09:30:11 ~ 09:34:26 사이, 추정)
  - C0020 · C0105 — 23:57:52 stereo 단계 rc=1 (stereo_check 가 pilot_common 을 import 하지 못함 → 00:49:04 /scratch 의
    d1_autode_pilot 링크로 고침). 그 뒤 C0105 는 잡 24036852 `d1d0` (node5780, 1코어, 00:57:27 ~ 11:34:22),
    C0020 은 잡 24036883 `d1d0b` (node0681, 1코어, 01:03:26 ~ 07:02:12) 에서 라벨까지 끝났다 (EDA 1코어).
- 1코어 결과를 8코어 재실행이 이어 쓴 것: r0 · r1 반응물 최적화는 D1 10행 모두, p0 생성물 최적화는 T00019 · T00879 · T01165 ·
  T01206 · T10385 · T01914 · T02935. T01916 · T00836 · T02894 는 p0 가 도는 중에 끊겨 d1B 가 8코어로 다시 했다.
  autodE 는 입력을 다시 쓰고(%pal nprocs 8) 정상 종료한 출력이 있으면 계산을 건너뛰므로, %pal 이 있는 .inp 옆의 .out 이
  1코어 실행분인 경우가 있다 (T01916 · T00836 · T02894 의 r0/r1).

### 2. .envfix_ts · .envfix2_ts 가 붙은 재실행 (T00019 T00879 T01165 T01206 T10385)
- .envfix_ts: 위 1 의 .fail_ts (node5778, matplotlib 없음). 사용자가 06:53:54Z 에 login05 에서
  `venv/bin/python -m pip install --only-binary=:all: matplotlib "pillow>=9.5.0" loky` 로 설치. 07:02:43Z 에 assistant 가
  컨테이너에서 .fail_ts → .envfix_ts 로 이름을 바꿨다 (원 spec 에 없는 표식). **바로잡음: pillow 는 빠져 있지 않았다**
  (rdkit 의 의존으로 이미 11.3.0 이 있었다 — 위 07:02 기록의 "matplotlib·pillow·loky 누락"은 matplotlib·loky 누락이 맞다).
- 07:00:00Z 잡 24037992 `dire5` (node1023): 남은 .fail_ts 때문에 아무것도 돌지 않고 18초 만에 끝남 (공백으로 나눈 -v 목록 때문에 T01165 빠짐).
- .envfix2_ts: 잡 24038032 `d1re5` (node0934, NC=8) 에서 07:15:57 ~ 07:16:08Z, 시작 뒤 73–84초 만에
  `CouldNotGetProperty: Could not get energy`. ORCA: SMD Hessian `%pal nprocs 8` 이 `MPI_Type_match_size … MPI_ERR_ARG` 로
  끝남 — OpenMPI 를 --disable-mpi-fortran 으로 빌드한 탓 (노드 문제 아님, ENV_REPORT.md §5). 07:46:03Z 에 assistant 가
  .envfix2_ts 로 이름을 바꾸고 MPI_OK=0 으로 돌렸다.
- 그 뒤: 잡 24038078 `d1re5` 2차 (node4487, NC=1, 07:53:37Z 시작) → mpi_fortran.pbs 가 Fortran 포함 OpenMPI 4.1.8f 를
  08:47:36Z 에 확인(MPI_OK=1) → 잡 24038698 `d1A` (node5778, NC=8, 09:34:26 ~ 17:49:47Z) 가 5행을 라벨까지.

### 3. T00019 의 Hessian "수동 실행" — 수동이 아니었다. 같은 폴더를 두 잡이 동시에 돌렸다
- transition_states/NiXL8Q_ll_ad_4-8_6-7_hess_orca.out: "Host name: node4487", "Working dir.: /var/tmp/pbs.24038078.pbs/…",
  직렬, 16:54:50 KST 시작, 1시간 16분, 09:11:27Z 끝. pilot.pbs → run_task.sh → run_ts.py → autodE 가 1코어(NC=1)로 돌린 것이다.
- 문제는 잡 24038078(1코어)이 09:34Z 부터 13:24 ~ 13:33Z(qdel)까지 d1A(8코어)와 **같은 5행을 같은 폴더에서 동시에** 돌렸다는 것이다
  (13:23:47Z 에 assistant 가 알리고 사용자가 qdel). 결과: T10385 의 1코어 Hessian 이 10:45:04Z 에 d1A 의 8코어 Hessian 파일을
  덮어썼다. d1A 의 OptTS 는 10:00:36Z 에 이미 Calc_Hess true 로 시작했으므로 TS 에는 영향이 없다 (그러나 디스크의 hess .inp 에는
  %pal 이 없고 autode.log 에 24038078 의 줄이 섞여 있다). T01165 · T01206 은 d1A 의 Hessian 이 09:51Z 에 덮어썼다 (추정).
- 고친 것: 작업 풀 밖에서 계산을 돌리지 않는다 (FIX1 §3). 본실험 · D0 대조 재계산은 claim 이 있는 작업 풀(worker_pool.sh)로만 돈다.

### 4. timing.tsv 에 없는 시도 · sp 행
- 저장소 run_stage 는 단계가 끝난 뒤에만 timing.tsv 에 한 줄을 쓴다 → 도중에 죽은 시도는 남지 않았다
  (T01914 · T01916 · T02894 · T00836 의 d1pilot 1코어 시도, d1A 5행의 node4487 1코어 시도, T02935 는 timing.tsv 자체가 없음).
  → FIX1 §3: run_env.sh 의 run_stage 가 단계 시작 때 attempts.tsv 에 한 줄을 쓴다.
- sp 행: 파일럿 때의 run_task.sh 가 run_sp 를 run_stage 밖에서 불렀다 (`if [ "$cmd" = "RUN_SP" ]; then run_sp "$JOB"`).
  → 이제 sp 도 run_stage 로 부른다 (끝난 행마다 ts · stereo · ref · inputs · sp · label 여섯 행).
2026-10-06T07:20:42+0900 T05174 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: cpu0088
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T08:29:54+0900 T08256 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node5242
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T08:29:54+0900 T01354 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node5242
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T08:38:50+0900 T11156 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node6244
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T09:27:12+0900 T01644 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node1214
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T09:34:16+0900 T10602 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node4942
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T10:01:11+0900 T11184 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node1172
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T10:03:28+0900 T11176 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node5244
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T10:06:44+0900 T10950 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node1317
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T10:42:06+0900 T08463 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node7301
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T10:52:05+0900 T07453 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: cpu0078
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T12:03:40+0900 T05223 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node7301
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T12:29:37+0900 T10267 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node0645
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T12:40:33+0900 T05758 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node8025
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T13:36:07+0900 T02680 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node3816
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T13:44:22+0900 T02672 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node3806
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T13:48:00+0900 T07975 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node0669
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T13:48:00+0900 T09235 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node0669
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T15:24:05+0900 T00064 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node5227
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T15:45:23+0900 T00835 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node7197
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T15:51:16+0900 T01370 sp
  사유: ORCA did not terminate normally: eda eda_frag2
  노드: node1221
  조치: §9.3 1회 재실행 (비정상 출력만 다시)
2026-10-06T18:41:12+0900 T01985 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node0093
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T18:49:00+0900 T02441 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node5924
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T18:53:12+0900 T09919 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node3916
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T21:26:34+0900 T09913 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node1303
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T22:19:33+0900 T04350 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: cpu0113
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-06T23:28:42+0900 T00283 sp
  사유: ORCA did not terminate normally: eda eda_frag2
  노드: node5363
  조치: §9.3 1회 재실행 (비정상 출력만 다시)
2026-10-06T23:54:21+0900 T06535 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node5928
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T00:55:41+0900 T04379 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node5497
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T01:02:42+0900 T10515 sp
  사유: ORCA did not terminate normally: eda eda_frag1
  노드: node6401
  조치: §9.3 1회 재실행 (비정상 출력만 다시)
2026-10-07T01:21:57+0900 T04843 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node1302
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T02:04:17+0900 T00716 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node4875
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T04:22:13+0900 T03723 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node6336
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T04:26:13+0900 T04584 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node3914
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T04:33:20+0900 T07623 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node0175
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T06:08:28+0900 T09804 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node4771
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T06:59:38+0900 T00239 sp
  사유: ORCA did not terminate normally: eda eda_frag1
  노드: node6242
  조치: §9.3 1회 재실행 (비정상 출력만 다시)
2026-10-07T07:06:38+0900 T00196 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node3803
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T07:09:27+0900 T05792 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node8039
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T07:16:13+0900 T11377 sp
  사유: ORCA did not terminate normally: eda eda_frag1
  노드: node0606
  조치: §9.3 1회 재실행 (비정상 출력만 다시)
2026-10-07T08:19:46+0900 T09445 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node0613
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T08:29:03+0900 T01829 sp
  사유: ORCA did not terminate normally: eda eda_frag1
  노드: node7272
  조치: §9.3 1회 재실행 (비정상 출력만 다시)
2026-10-07T09:30:03+0900 T02491 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node7196
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T09:47:39+0900 T08342 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node0423
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T10:36:13+0900 T00895 ts
  사유: autodE exception: CouldNotGetProperty: 
2026-10-07T10:36:13+0900 T02577 ts
  노드: node1223
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node1223
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T10:45:53+0900 T03785 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node5998
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T12:49:57+0900 T07231 sp
  사유: ORCA did not terminate normally: eda
  노드: cpu0107
  조치: §9.3 1회 재실행 (비정상 출력만 다시)
2026-10-07T13:36:48+0900 T09260 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node5006
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T15:27:42+0900 T07544 sp
  사유: ORCA did not terminate normally: eda
  노드: node4998
  조치: §9.3 1회 재실행 (비정상 출력만 다시)
2026-10-07T16:22:06+0900 T10219 sp
  사유: ORCA did not terminate normally: eda
  노드: cpu0088
  조치: §9.3 1회 재실행 (비정상 출력만 다시)
2026-10-07T16:37:09+0900 T08019 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: cpu0113
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T16:54:06+0900 T08373 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node5402
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T17:51:33+0900 T08906 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node3587
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-07T19:10:28+0900 T07389 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node7295
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-08T00:48:38+0900 T02662 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node0424
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-08T02:41:30+0900 T02758 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node7197
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-08T03:29:34+0900 T08321 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node7236
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-08T05:49:20+0900 T08019 sp
  사유: ORCA did not terminate normally: eda
  노드: cpu0113
  조치: §9.3 1회 재실행 (비정상 출력만 다시)
2026-10-08T08:34:54+0900 T10088 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node0733
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-08T09:05:27+0900 T05585 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node2427
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-08T16:13:14+0900 T09652 ts
  사유: autodE exception: CouldNotGetProperty: Could not find Hessian file
  노드: node0320
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-09T06:01:15+0900 T05430 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node1311
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-09T06:15:39+0900 T11033 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node0318
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-09T13:41:38+0900 T05721 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node1012
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-09T19:54:59+0900 T11496 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: node8041
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-10T07:28:40+0900 T02794 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node8027
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-10T08:29:14+0900 T10241 ts
  사유: autodE exception: CouldNotGetProperty: Could not get energy
  노드: node0595
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-13T09:33:14+09:00 S0 ts
  사유: autodE exception: CouldNotGetProperty: 
  노드: n040.hpc
  조치: §9.3 1회 재실행 (autodE checkpoint 에서 이어 감)
2026-10-13T09:33:42+09:00 C0105 sp
  사유: ORCA did not terminate normally: eda eda_frag1 eda_frag2
  노드: n116.hpc
  조치: §9.3 1회 재실행 (비정상 출력만 다시)
2026-10-13T09:34:53+09:00 C0020 sp
  사유: ORCA did not terminate normally: eda_frag2
  노드: n116.hpc
  조치: §9.3 1회 재실행 (비정상 출력만 다시)
2026-10-16T23:30:34+09:00 T08621 sp
  사유: ORCA did not terminate normally: eda
  노드: n116.hpc
  조치: §9.3 1회 재실행 (비정상 출력만 다시)
