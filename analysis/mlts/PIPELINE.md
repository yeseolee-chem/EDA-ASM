# MLTS 자동 단계 실행기 (`pipeline.py`) — 운영 메모 / operator notes

명령서 `docs/specs/DEPLOY_V2_MLTS.md`의 순서(0 → I → G → Z → FT → A → H → C → R → K → E → X → W → P → S)를
계산 노드의 launcher job이 10분마다 한 번씩 확인하고, 준비된 단계를 제출한다(CLAUDE.md 2번 방식).
로그인 노드에서는 제출 1회와 파일 읽기 외에 아무것도 하지 않는다.
The launcher is a 1-CPU sbatch job that re-queues itself; nothing stays resident on gate1.hpc.

## 시작 / start (lead, 1회)

1. `pipeline.py`, `jobs/pipeline.slurm`, `tests/test_pipeline.py`, `PIPELINE.md`를 commit한다. 실행기는 **git HEAD와 같은
   코드만** 돌린다: launcher 자신(`pipeline.py`, `mlt_common.py`)이 HEAD와 다르면 pass를 건너뛰고, 각 단계는 그 스크립트와
   그것이 import하는 `analysis/mlts` 모듈이 모두 HEAD와 같아야 제출된다. job이 시작할 때도 같은 파일을 다시 본다
   (`pipeline.py guard`, 다르면 exit 4 → 재시도 횟수를 쓰지 않고 commit 뒤 다시 낸다).
2. 디스크 값을 적는다. **값이 없거나 24 h보다 오래되면 아무것도 제출하지 않는다**(DECISIONS 0-8 "제출 전마다").
   login에서 `/usr/lpp/mmfs/bin/mmlsquota --block-size G`의 여유 GB를 Write 도구로 `$MLT/QUOTA_FREE_GB`에
   `<GB> <ISO 시각>` 한 줄로 쓴다(예: `102 2026-11-04T16:00:00`). 매시 점검 때 갱신한다.
   실행기는 그 뒤에 제출했거나 그 시각에 돌고 있던 job의 예상 쓰기량을 빼서 여유를 추정하고, 30 GB 밑이면 멈춘다.
3. `analysis/mlts`에서 `squeue -u $USER -h | wc -l` 확인 후 한 번만:

```
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-mlts/analysis/mlts && sbatch jobs/pipeline.slurm
```

Each run: stop if `$MLT/PIPELINE_PAUSE` is `exit` or `results/STOP_S.md` exists → queue the successor
(`--begin=now+10minutes`, `--dependency=singleton`) → self-check → `timeout 600 python pipeline.py once`.

## 무엇을 보나 / what to read (login: cat 만)

- `$MLT/PIPELINE_ATTENTION` — 한 줄 요약(HALT, QUOTA 값 없음, push 실패, 후속 launcher 없음, pass 실패 …). `ok`이면 할 일 없음.
- `results/PIPELINE_STATE.json` — 단계별 status (waiting_code · blocked · ready · queued · running · idle · done · skipped ·
  halted · cancelled), why, epoch(라운드), units(작업 · job id · 끝 상태 · 재시도 수), decision, commit.
- `results/PIPELINE_LOG.md` — pass마다 바뀐 것 한 줄 (submit / adopt / done / HALT / retry / commit).
- `results/DECISIONS.md` — 자동 결정마다 `- **auto <단계> (시각, pipeline)**: … <!-- mltp:<id> -->` 한 줄(id로 중복 방지).
  실행기가 이 파일 끝에 덧붙이므로, 손으로 고칠 때는 `PIPELINE_PAUSE`를 두고 고친다(동시에 쓰면 한 줄이 빠질 수 있음).
- sandbox에서: `bash jobs/sbx.sh reactot pipeline.py status`, `bash jobs/sbx.sh reactot pipeline.py dry`
  (읽기 전용 pass: 낼 sbatch 줄을 출력만, 아무것도 쓰지 않음).

## 멈춤 · 다시 시작 / pause · resume

- 새 제출만 멈춤: `$MLT/PIPELINE_PAUSE` 파일을 만든다(내용 비움). 대조 · 기록 · commit은 계속, launcher도 계속 돈다.
  다시 시작: 그 파일을 지운다(lead 판단 — 실행기는 아무것도 지우지 않음).
- launcher까지 끝냄: 내용을 `exit` 한 줄로. 다시 시작하려면 파일을 지우고 위의 `sbatch jobs/pipeline.slurm`.
- 실행기는 **어떤 job도 취소하지 않고 어떤 자료도 지우지 않는다.** lead가 job을 취소하면 그 작업은 다시 내지 않는다.
  단계의 산출물이 다 차면 그대로 끝나고, 모자라면 `cancelled` + `results/STOP_<단계>.md`(lead가 고름).

## HALT 풀기 / clear a HALT

추천안이 정해진 분기(G-D0 경로 수 재조정, Z 후보 1개, LODO는 능동학습 자료 포함, A-3 0.70–0.85, A″ arm 입력 실패 시
arm 제외, P-3 실행, A-3 목록 분할)는 자동으로 진행하고 DECISIONS에 적는다. 추천안이 없는 STOP(FT-2 2차 실패, A-3 < 0.70
또는 예산 안 엔진 없음, G 예산 초과, H-2 > 2 %, C-2 · C2-K · C-1 실패, E 등록 변경, A-4 디스크 부족, exit 3)과 운영 실패
(재시도 소진)는 **HALT**: `results/STOP_<단계>.md`(조건 · 수치 · 선택지 · 추천)를 쓰고 그 단계와 의존 단계를 멈춘다.
- 범위: 보통 단계의 HALT 동안에는 새 단계도 시작하지 않는다(이미 시작한 독립 단계는 계속). 보고용 filler 단계
  (FT2-r1, X-run, X-rep, H-early, FT-fin)의 HALT는 그 단계에 의존하는 단계만 멈춘다.
- 운영 HALT(재시도 소진)와 `cancelled`는 그 단계의 산출물이 나중에 다 차면(예: 손으로 낸 job) 저절로 풀린다.

푸는 법: 그 파일의 **첫 줄**에 다음 중 하나를 쓴다. 다음 pass가 상태에 적용 · 기록한 뒤 파일 이름을
`STOP_<단계>.resolved.md`로 바꾼다(지우지 않음; 같은 내용은 두 번 적용하지 않음).

| 첫 줄 | 효과 |
|---|---|
| `RESOLVE: retry` | 시도 횟수 0, 새 라운드로 다시 낸다(아직 도는 그 단계 job은 새 라운드로 넘어가고 다시 내지 않음) |
| `RESOLVE: accept` | 이 단계를 끝난 것으로 보고 다음으로 |
| `RESOLVE: skip` | 건너뛴다 (뒤 단계는 이 단계 산출물 없이 시작될 수 있음) |
| `RESOLVE: param <k>=<v>` | 이 단계 매개변수를 바꾸고 다시 낸다 |
| `RESOLVE: param <단계>.<k>=<v>` | 다른 단계(예: `G-D0f.n_g2=150`) 매개변수를 바꿔 그 단계를 다시 내고, 이 STOP은 통과 |

## 손으로 낸 job / manual jobs

실행기는 squeue와 (pass마다) sacct에서 등록부의 이름(`g_run1`, `g_run1rev`, `g_run3`, `g_run2`, `g_collect`, `z_omol`,
`z_mh1`, `z_aim`, `z_off24`, `z_xtb`, `z_speed`, `z_report`, `ft_data`, `ft_p_<후보>`, `g_plan_d0`, `g_g0_d0`,
`g_run_d0g1|3|2`, `g_collect_d0`; 그 밖의 `g_*d0*`는 G-D0을 막기만 함)을 자기 작업으로 받아들인다. 그 밖의 손 제출은
`--comment=mltp:<단계>:<작업>:manual`을 붙이면 같은 작업으로 받아들이고 중복 제출하지 않는다.
대기열 자리: 실행기는 모든 task(launcher · sandbox · 손 작업 포함) + 후속 launcher를 **19개 이하**로 두어 손 작업
1개 자리를 늘 남긴다. pass 사이에 손 작업을 2개 이상 내면 후속 launcher가 밀려날 수 있다(ATTENTION이 알림 →
`sbatch jobs/pipeline.slurm`).

## 자리 규칙 / slot policy (이 cluster는 선착순: `PriorityType=priority/basic`, `--nice` 안 씀)

- 대기열: 전체 task + 후속 launcher ≤ 20 − 1(손 작업 자리). 실행 10칸은 SLURM이 지킨다.
- 바로 시작할 수 있는 대기 task(Dependency · BeginTime · hold가 아닌 것, launcher 제외) ≤ 빈 실행 칸 + 2(핵심 경로).
  filler는 빈 실행 칸에만 넣어 핵심 작업 앞에서 기다리지 않게 한다. 앞 작업에 afterany로 걸린 다음 작업은 미리 낸다.
- CPU ≤ 640 · GPU ≤ 12 (대기 포함). 64코어 job은 cpu2, MACE 학습 gpu1/3/4/5, 같은 GPU 비교는 gpu3/4/5.
- sbatch 오류: MaxSubmit · 제어기 시간 초과 · Socket timed out은 그 pass의 제출만 멈춘다(HALT 아님; 시간 초과로
  생겼을 수 있는 job은 comment로 찾아 붙인다). 같은 고정 오류가 3 pass 이어지면 HALT.

## 단계 더하기 / add a stage

`pipeline.py`의 `registry()`에 `Stage(...)` 하나를 더한다(위상 순서 자리에):
`Stage(name, phase, crit, units_fn, after=[...], code=[(script, sub[, regex])], done=fn(P), decide=fn(P, S, st),
results=["<results 아래 glob>"], filler=False, active=fn(P) | None, partial=fn(P) | None)`. `units_fn(P, S, st)`는 지금 필요한
작업 목록을 돌려준다(`one(key, cmd, env=, part=, c=, mem=, gpu=, deps=, done=, sink=, disk_gb=)`,
`arr(akey, cmd + ["@I", n], n, ...)`). 바꾼 뒤 `bash jobs/sbx.sh reactot tests/test_pipeline.py`.

## 단계 코드가 지켜야 할 계약 / interface contract for phase code

- 명령: `python <script> <sub> [args] [<i|@I> <n>]`, `analysis/mlts`에서 `jobs/mlt.slurm`으로 실행(실행기가 앞에
  `pipeline.py guard <files> :::`를 붙임). `@I` = `$SLURM_ARRAY_TASK_ID`(`mlt_common.array_task`).
- exit 0 끝 · 2 미완 / 준비 안 됨(같은 명령 다시) · 3 STOP(HALT). exit 1 · 노드 오류는 1회만 다시, 메모리 부족은 1.5배로 1회.
- 단위마다 원자적 쓰기(`mlt_common.write_*`) + 결과가 있으면 건너뜀. 48 h 중 2 h 남으면 새 일을 잡지 않음(`MLT_JOB_T0`).
- 보고 JSON(`results/<X>_report.json`): `complete` (bool), `n_todo`, `stop` (null | 글), `est {units_total, unit_wall_h,
  workers}`, 그리고 아래 단계별 값. 등록 이름(`add_parser("<sub>")` 등)이 파일에 있어야 실행기가 그 단계를 연다.

| 단계 | 명령 (env, 자원) | 끝난 조건 · 읽는 값 |
|---|---|---|
| G4 | `gen_dft.py g4_plan --models <FT_0..4 1차>` (R, C 4c) → `g4_opt @I n --models ...` (M, GA 16c) → `run G4 @I n` (R, cpu2 64c) → `collect --g4` | `G4_plan.json` {complete, predicted_core_h (실행 전 예산 검사), est{opt,run}}; `G4_report.json` {complete, core_h_measured.total} + `$MLT/dft/gen_G4.extxyz`; G 전체 > 5,000 → HALT |
| X-rep | `ts_geom.py x_report` (M, C 4c) | `X_report.json` {complete} |
| A2-smoke | `mlts_autode.py smoke --model FT_0_al` (M, GA) | `A2_smoke.json` {pass} (C-3 문구) |
| A3 | `mlts_autode.py run a3 @I n --list $MLT/pipeline/lists/a3.csv` (M, GA 32c 96G) → `refine_dft a3 @I n` (R, cpu2 64c) → `a3_report` | `A3_report.json` {complete, arms{A, Ap, App}{same_ts_frac, core_h_median, gpu_min_median, n, disk_gb_per_rxn}, disk_gb_per_rxn, est}. 비용은 배포 비용 그대로: Ap는 G2 TS 생성 포함, App는 A 포함. 예산 A · Ap ≤ 0.6 core-h + 3 GPU-분, App ≤ 1 core-h + 3 GPU-분 |
| A4 | fold k: `mlts_autode.py run d1:<k> @I n --model FT_<k>_al --engine <A\|Ap\|App>` (M, GA 32c 96G); App이면 `refine_dft d1:<k> @I n`; 끝에 `status` | `A4_report.json` {complete, folds{k: {complete}}}; fold −1(anchor-only)은 `FT_0_al`; Ap의 G2 TS 없는 반응은 코드가 처리 · 기록 |
| A4-App | (A-3이 A″ arm을 정했을 때만) `mlts_autode.py refine_dft d1:<k> @I n --root $MLT/app` (R, cpu2 64c) → `status --root $MLT/app` | `$MLT/app/reports/A4_report.json` {complete}; A″ 폴더는 `$MLT/app/ade/jobs/<job_id>/`(A-2 형식) |
| H-App | 실행기가 `inputs_mlts.py ... --root $MLT/app` 사슬을 돌린다 | `$MLT/app/reports/H2_report.json`; K · E는 `$MLT/pipeline/arms.json` {engine, extra_arm_App, app_root, app_usable}을 읽는다 |
| C-agg | `tests.py all` (M, GA) | `C_tests.json` {complete, C2{pass}, C7{summary}} |
| R | `prereg_v2.py write` (R, C) | `PREREG_V2.md`를 실행기가 commit (HEAD와 같아야 K 시작) |
| K-C2C3 | `train_k.py plan` → `train_k.py <c2\|c3> <unit> @I n` → `train_k.py select` | `K_plan.json` {complete, units[{arm, unit, n, cpus, mem_gb, gpu}]}; `K_report.json` {complete, primary, leakage_C2{pass_}} |
| C2-K | `tests.py c2_k` (R, C 4c) | `C2K_report.json` {complete, pass, n_units, n_overlap} — K 채널 단위의 f{k} 학습 짝 ∩ fold k = ∅, 아니면 HALT |
| E | `eval_v2.py run` (R, C 16c) | `E_report.json` {complete, grade, prereg_change_needed} |
| W | `final_w.py channel`, `committee @I 3`, `report` | `W_report.json` {complete} |
| P | `predict_v2.py selfcheck`, `cert @I 1` (P-3, 추천안대로 실행), `report` | `P_report.json` {complete, selfcheck_pass} |
| S | `summary_v2.py write` | `SUMMARY_V2.md` → `STOP_S.md`("명령서 S: 멈춤"), launcher 끝 |

실행기가 직접 쓰는 결과: `results/FT_report.json` · `results/FT_geom.csv`(명령서 5절, FT2-r2 판정 때; FT-fin 뒤 갱신),
`$MLT/pipeline/lists/*.csv`, `$MLT/pipeline/arms.json`.

## 시험 / tests

`bash jobs/sbx.sh reactot tests/test_pipeline.py` — 가짜 squeue · sacct · sbatch · git · 시계, 모든 쓰기는
`$MLT/tmp/pipetest/<시나리오>/` 아래. 결과 요약 `$MLT/tmp/pipetest/results.json`.
