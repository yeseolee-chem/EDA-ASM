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

추천안이 정해진 분기(G-D0 사용자 선택 200 + 100의 예산 확인, Z 후보 1개, LODO는 능동학습 자료 포함, A-3 0.70–0.85,
A″ arm 입력 실패 시 arm 제외, P-3 실행, A-3 목록 분할)는 자동으로 진행하고 DECISIONS에 적는다. 보고만 하는 단계(N-scan,
D0-eval, X-rep와 filler)의 HALT는 그 단계에 의존하는 단계만 멈춘다. 추천안이 없는 STOP(FT-2 2차 실패, A-3 < 0.70
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
`d0p_plan`(G-D0f), `g_run_d0g1|3|2`(배열 원소마다; 그 배열 크기가 그 묶음의 `@I n`), `g_collect_d0`, `nur_<tar 약칭>`(N-scan;
comment `mltp:nur:<tar>:manual`도); 그 밖의 `g_*d0*`(`g_run_d0g1rev`, `g_run_d0g1shuf1-3` 등 보조 task)는 G-D0의 새 제출을
막기만 함)을 자기 작업으로 받아들인다. 그 밖의 손 제출은 `--comment=mltp:<단계>:<작업>:manual`을 붙이면 같은 작업으로
받아들이고 중복 제출하지 않는다.
- D1 G-2(1061710)는 사용자가 317/400에서 취소했다(결정 'Nurion'): G-D1은 G1 · G3가 빠짐없고 G-2가 317단위 이상이면 끝난 것으로
  보고(`G_report` complete=false여도) G-2를 다시 내지 않는다.
- Nurion: FT 자료 단계(FT-dp, FT-d1, FT-d2)는 `results/N_report.json` complete(결과 tar 23개 모두)가 되거나, lead가
  DECISIONS.md에 `- **N partial ok (...)**: ...`로 시작하는 줄을 쓰면(또는 `RESOLVE: param N-scan.partial_ok=1`) 시작한다.

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

env: R = reactot, M = mlts, D = d1 (스크립트가 필요하면 스스로 다른 env로 다시 실행). C = cpu1,cpu2, C2 = cpu2, GA = 모든 GPU.

| 단계 | 명령 (env, 자원) | 끝난 조건 · 읽는 값 |
|---|---|---|
| G-D1 | 손 job 넘겨받음(`g_run1/1rev/3/2`, `g_collect`); 다시 할 때 `gen_dft.py run G1\|G3 @I n` (R, C2 64c 240G) → `collect` | `G_report.json` complete, 또는 G1 · G3 빠짐 0 + G-2 ≥ 317/400(사용자 결정 'Nurion', G-2는 다시 내지 않음) |
| G-D0p / G-D0e | `gen_dft.py plan_d0` → `gen_dft.py g0_d0` (R, C 16c) | `G_plan_d0.json` (limit 없음) → `G0_estimate_d0.json`(계획 fingerprint − n_near 같음); 예산 = D1(실측 · 계획 기준 중 큰 값) + D0 G1/G3 + D0G2 200 + G-4 150 ≤ `gen_dft.BUDGET_CORE_H`(8,000) 아니면 HALT. g0_d0의 stop 표시(그때 상한)는 쓰지 않음 |
| G-D0f | `gen_d0.py plan_d0 --n-near 2 --n-g2 200 --n-holdout 100` (R, C 16c; 손 job `d0p_plan` 넘겨받음) | `G_plan_d0.json` budget.n_g2_selected = 200, budget.n_holdout = 100(평가 전용, 정제 안 함) + `d0_holdout.csv` |
| G-D0 | `gen_dft.py run D0G1\|D0G3 @I n` (R, C2 64c 240G), `gen_dft.py run_g2 @I n --d0` (D, C2; 학습용 200만), `collect --d0` | `G_report_d0.json` {complete}; n = 넘겨받은 배열 크기(D0G1 2, D0G3 1, D0G2 3) |
| N-scan | 올라왔고 아직 안 읽은 tar마다 `inv_nurion.py scan <tar> --workers 8 ::: inv_nurion.py verify <tar>` (R, C 8c 24G) → `inv_nurion.py collect` (R, C 4c 64G) | 단위 끝 = `$MLT/dft/nurion/<tar>/.done`; 단계 끝 = `N_report.json` complete(결과 tar 23개) 이고 모든 .done 뒤에 씀. 아직 안 올라온 tar는 기다림(re-arm). tar 사본은 사용자가 손으로 지운다(release 안 씀, 결정 'tar 삭제 방식') |
| FT2-p | `ts_geom.py run FTpilot_<c>_0 <f0_test.csv> label\|g2 <out> @I n` (M, GA 16c 64G; n = ⌈반응 수 × max(Z wall, 180 s) / 14 작업자 / 2 h⌉ ≤ 4, f0_test 1,874 → 4, 두 출발 GPU 8) → 1차가 끝나면(전부 기록, 또는 99.5 % 이상 + 1차 task 없음) 반응별 기준에 실패한 반응만 `ts_geom.py run ... <out>_recalc5 @I n --recalc 5 --only-failed <out>` | 필요: `$MLT/pipeline/lists/f0_test_ez.csv`(`ez_groups.py run`, M, C 4c; 없으면 기다림). 끝 = 두 출발 1차 기록 완료 + 실패 반응 모두 재측정. 판정(`decide_ft2p`): 반응마다 재측정 기록이 있으면 그것을 씀(recalc 5에서도 실패해야 실패); 실패 분류 (i) not_converged (ii) n_imag ≠ 1 (iii) other_saddle(수렴 · 허수 1개인데 다른 TS 또는 \|Δd\| > 0.030 Å)를 출발별 1차 / 재측정 뒤로 기록; E/Z 형제 있음 · 없음 무리의 (a) · (b) 수치; 라벨 출발 두 무리 차이가 같은 TS > 0.05 또는 \|Δd\| 중앙값 > 0.002 Å이면 관문은 '형제 없음'만(결정 줄 + `DEVIATIONS.md` 한 줄), 아니면 전체 |
| G4 | `g4_al.py g4_plan --models <FT_0..4 1차>` (R, C 4c 16G) → `g4_opt @I n --models ...` (M, GA 16c 64G) → `run G4 @I n` (R, C2 64c 240G) → `collect --g4` (R, C 4c 32G) | `G4_plan.json` {complete, n_d1 1,000, n_d0 500, predicted_core_h.total (실행 전 예산 검사), est{opt, run}}; `G4_report.json` {complete, core_h_measured.total, stop_file} + `$MLT/dft/gen_G4.extxyz`; G 실측 + G-4 > `gen_dft.BUDGET_CORE_H` → HALT |
| FT2-r2 | 목록이 하나라도 없으면 `ez_groups.py run --folds 0,1,2,3,4` (M, C 4c 16G; 있는 목록은 건너뜀) + 학습이 끝난 FT_<k>_al마다 `ts_geom.py run FT_<k>_al <f<k>_test.csv> label\|g2 <out> @I n` (FT2-p와 같은 n) → 1차가 끝난 (run, 출발)마다 실패 반응만 `... <out>_recalc5 @I n --recalc 5 --only-failed <out>` | 끝 = `$MLT/pipeline/lists/f0..f4_test_ez.csv` + 모든 1차 기록 + 실패 반응 모두 재측정(partial · progress도 재측정 포함). 판정(`decide_ft2r2`, FT2-p와 같은 규칙): 반응마다 재측정 기록이 있으면 그것; 실패 분류를 출발별 1차 / 재측정 뒤(5 fold 합침 + fold별); fold마다 자기 목록(학습 쪽 = T1 fold ≠ k)으로 나눈 E/Z 형제 있음 · 없음 (a) · (b)를 5 fold 합침; 라벨 출발 두 무리 차이 > 0.05 같은 TS 또는 > 0.002 Å이면 '형제 없음'만(결정 줄 + `DEVIATIONS.md`), 아니면 전체; 실패면 HALT. `results/FT_report.json`(gate_round2 = 판정 numbers) · `FT_geom.csv`(합친 기록 + fail_class · remeasured · first_pass_fail_class) |
| X-rep | `ts_geom.py x_report` (M, C 4c) | `X_report.json` {complete, per_dipole{pass_a}} |
| D0-eval | `ts_geom.py d0_eval run @I 3` (M, GA 16c 64G) → `ts_geom.py d0_eval report` (M, C 4c) — FT-fin 뒤, 보고만 | `D0_eval_report.json` {complete, per_model, pooled, floor_rmsd_A 0.009}; 기준 TS = Coley TS(D0-gauss), 출발 label = Coley TS, g2 = G2c xTB TS |
| A2-smoke | `mlts_autode.py smoke --model FT_0_al` (M, GA 8c 32G) | `A2_smoke.json` {complete, pass} (C-3 문구); exit 3 + complete → 판정 규칙(HALT) |
| A3 | `mlts_autode.py run a3 @I n --list $MLT/pipeline/lists/a3.csv` (M, GA 16c 64G: 작업자 6 × 2코어 + 서버) → `refine_dft a3 @I n` (R, C2 64c 240G) → `a3_report` (R, C 4c) | `A3_report.json` {complete, arms{A, Ap, App}{same_ts_frac, core_h_median, gpu_min_median, n, disk_gb_per_rxn, cost_basis}, est}. 비용 = 배당 비용(job 코어 · GPU를 작업자 수로 나눔): Ap는 G2 TS 생성 포함, App는 A 포함; 진척 = `$MLT/a3/<arm>/ade/jobs` |
| A4 | fold k: `mlts_autode.py run d1:<k> @I n --model FT_<k>_al --engine <A\|Ap\|App>` (M, GA 16c 64G); App이면 `refine_dft d1:<k> @I n`; 끝에 `status` | `A4_report.json` {complete, folds{k: {complete}}}; fold −1(anchor-only)은 `FT_0_al`; App이면 A는 `$MLT/ade_pre`, DFT 정제가 `$MLT/ade`(진척은 둘 다) |
| A4-App | (A-3이 A″ arm을 정했을 때만) `mlts_autode.py refine_dft d1:<k> @I n --root $MLT/app` (R, cpu2 64c) → `status --root $MLT/app` | `$MLT/app/reports/A4_report.json` {complete}; A″ 폴더는 `$MLT/app/ade/jobs/<job_id>/`(A-2 형식) |
| H-App | 실행기가 `inputs_mlts.py ... --root $MLT/app` 사슬을 돌린다 | `$MLT/app/reports/H2_report.json`; K · E는 `$MLT/pipeline/arms.json` {engine, extra_arm_App, app_root, app_usable}을 읽는다 |
| C-agg | `tests.py all` (M, GA 16c 64G) | `C_tests.json` {complete, stop, C1..C7{pass}}; C-1 … C-6 실패 또는 stop = HALT(exit 3도 판정 규칙으로), C-7은 보고만 |
| R | `prereg_v2.py write` (R, C 2c) | `PREREG_V2.md` + `prereg_v2.json`을 실행기가 commit; 끝 = 둘 다 tracked · HEAD와 같음 · json md_sha256 = sha256(md) |
| K-C1 | `train_k.py c1pred atoms\|xgb\|collect --refs <등록 refs>` 그리고 `--refs <다른 refs>` (R; atoms GA 8c 64G, xgb C 32c 64G, collect C 4c) | `C1_report.json` {refs = 등록 refs} + `C1_<다른>_report.json`, 둘 다 PREREG 뒤 |
| K-C2C3 | `train_k.py plan` (R, C 2c) → `train_k.py <c2\|c3\|c1v> <unit> @I 1` (K_plan 자원) → `train_k.py select` (R, C 4c) | `K_plan.json` {complete, units[{arm, unit, n, cpus, mem_gb, gpu}]}; `K_report.json` {complete, primary, configs{c: {predictions(경로), refs, table, units_dir, rung, final}}, leakage_C2{pass_}} |
| C2-K | `tests.py c2_k` (R, C 4c 32G) | `C2K_report.json` {complete, pass, n_units, n_overlap} — K 채널 단위의 f{k} 학습 짝 ∩ fold k = ∅, 아니면 HALT |
| E | `eval_v2.py run` (R, C 16c 64G) | `E_report.json` {complete, grade, prereg_change_needed, prereg_change_reason} + `E_metrics_pos.csv` |
| W | `final_w.py channel` (M, GA 8c 64G), `committee @I 3` (M, GA 16c; FT_all_s<i>_al), `report` (R, C 4c) | `W_report.json` {complete} |
| P | `predict_v2.py selfcheck` (M, GA 16c), `cert @I 1` (R, C2 64c; P-3, 추천안대로 실행), `report` (R, C 4c) | `P_report.json` {complete, selfcheck_pass}; SMILES 입력은 `predict_v2.py predict` |
| S | `summary_v2.py write` (R, C 2c) | `SUMMARY_V2.md` → `STOP_S.md`("명령서 S: 멈춤"), launcher 끝 |

실행기가 직접 쓰는 결과: `results/FT_report.json` · `results/FT_geom.csv`(명령서 5절, FT2-r2 판정 때; FT-fin 뒤 갱신),
`$MLT/pipeline/lists/*.csv`, `$MLT/pipeline/arms.json`.

## 시험 / tests

`bash jobs/sbx.sh reactot tests/test_pipeline.py` — 가짜 squeue · sacct · sbatch · git · 시계, 모든 쓰기는
`$MLT/tmp/pipetest/<시나리오>/` 아래. 결과 요약 `$MLT/tmp/pipetest/results.json`.
