# RUNBOOK — GEO2 orchestration (role ORCH): the launcher `analysis/geo2/g2_launch.py`

Spec: `docs/specs/GEO2.md` (the authority). Role runbooks the launcher follows step by step: `RUNBOOK_feat.md` (T, G, F,
H, P), `RUNBOOK_tests.md` (C tests, E3), `RUNBOOK_lad.md` (R, M3). Decisions of this role:
`results/DECISIONS_orch.md` (O-1 ..). **Nothing here has been run** (written on the login node; no python allowed there).

```bash
REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1
L=/gpfs/tmp_cpu2/yeseo1ee/geo2/logs          # every log (all jobs: --output=$L/%x_%j.out)
R=/gpfs/tmp_cpu2/yeseo1ee/geo2/run_g2        # launcher state / status / controls
```

## 0. Rules

- Login node (`gate1.hpc`): only `sbatch`, `squeue`, `sinfo`, `sacct`, `scancel`, `git`, `cat` / `ls` / `grep` / `touch` /
  `mv`. No python, no loops, no `sleep`, no resident process. All compute, the launcher included, runs under SLURM with
  `--time=48:00:00`.
- MaxSubmit 20 (array elements count), MaxJobs 10 running. The launcher keeps `squeue -u $USER -h -r | wc -l` <= 19
  after its submissions (one slot for its own next pass) and <= 2 of its tasks pending per class (cpu / gpu); SLURM
  enforces the 10 running. Before any submission by hand: `squeue -u $USER -h -r | wc -l` + (new tasks) <= 20.
- Partitions: CPU `-p cpu1,cpu2`, GPU `-p gpu1,gpu2,gpu3,gpu5,gpu6 --gres=gpu:1` (SLURM takes the one that starts first;
  check with `sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`, `sinfo -p gpu1,gpu2,gpu3,gpu5,gpu6 -o "%.9P %.6t %.10G %C"`).
- Never export `GEO2_SCRATCH`, `GEO_SCRATCH`, `D1LADDER_SCRATCH`, `D1PAIR_SCRATCH` with other values (the wrappers and
  the gates compare the default paths).
- **While the launcher chain is alive, never submit a GEO2 step by hand under another job name.** The runbook array
  lines (`-J geo2_g`, `geo2_hb`, `geo2_fs`, `geo2_m3` with `--array`) do not match the launcher's per-index task names
  (`geo2_g0`, `geo2_hb0`, `geo2_fs0`, `geo2_m3_0`, ...) and would run twice. A step submitted by hand under the
  launcher's task name (section 4) is adopted (counted as an attempt, judged when it ends).

## 1. Order and dependencies (the launcher's DAG; spec 8 order 0 → Y → Z → T → G·F·H·P → R → M3 → E → S)

```
GEO1 Y jobs geo2_y_s0..2 (running since 08:02, adopted) ─ geo2_y_budget (Y GPU-h <= 30) ─┐
geo2_gate_geo1  (spec 1: GEO1 results committed + pushed; 0-2..0-4, 0-6 files) ──────────┼─ geo2_z (build_geo) ─┐
   ├─ geo2_t2g ─ geo2_t2f ┐                                                             │                      │
   ├─ geo2_t3 ────────────┴───────────────────────────────────── (Z + T-2 + T-3) ──────────────────────────────┘
   │        ├─ geo2_hip (h_inputs 5 ::: h_pilot) ─ geo2_hb0..4 ─ geo2_hr (h_build ::: h_report) ─┐
   │        └─ geo2_g0..3 ─ geo2_gr ─ geo2_fs0..3 ─ geo2_fr (f_rest) ───────────────────────────────┴─ geo2_p
   ├─ geo2_creg_g, geo2_creg_c ─ geo2_creg_m (regressions R-1..R-3, early) ─┐                         │
   │                                     geo2_ctest_g, geo2_ctest_c (after geo2_p) ─ geo2_ctest_m ─┘
   │                                                                   └─ geo2_R (prereg write) ─ GATE geo2_gate_prereg
   │                                                                      (human: review, commit, push) ─ geo2_Rc (check)
   │        ┌─ geo2_x3 (XGB 10 units) ─ geo2_x3c (check) ───────────────────┐
   │        └─ geo2_m3_0..3 (AW / GNN / TF 30 units, 4 GPU workers) ─────────┴─ geo2_m3col (collect --stage 3 --with-xgb)
   │                                                  ─ geo2_Rc2 (prereg check again: LADDER results unchanged, spec 14)
   └─ geo2_e3t (E3 selftest) ───────────────────────────── geo2_e (eval_geo2 run) ─ geo2_e3 (evaluate e3)  = finished
```
`geo2_hip`, `geo2_g*` need `geo2_z`, `geo2_t2f`, `geo2_t3`; `geo2_x3`, `geo2_m3_*` need `geo2_Rc`; `geo2_e3` needs
`geo2_e`, `geo2_e3t`, `geo2_Rc2`. A STOP (or a task lost twice) halts only its dependants; the other branches go on.

## 2. Before starting the launcher (orchestrator, by hand)

1. **Code freeze.** Every role's code is reviewed and final: `feat_geo.py`, `g3c.py`, `jobs/geo2.slurm`,
   `build_geo.py`, `eval_geo2.py`, `prereg_geo2.py`, `g2_launch.py`, `analysis/d1_ladder/{train_atoms,train_xgb,inputs,
   evaluate}.py`, `tests/run_tests.py`. T-2 / T-3 record the code sha256 (an edit afterwards makes every G / F / H / P
   step exit 2 = lost twice = STOP), the C tests record theirs (an edit = prereg STOP D-39), the PREREG freezes them.
2. **Start conditions (spec 1)** — checked by hand or by the gate `geo2_gate_geo1`:

   | # | check (login node, read only) | launcher |
   |---|---|---|
   | 0-1 | `git -C $REPO log --oneline -1 -- analysis/geo/results/E_report.json` (and E_rxn.csv, R_select_f0..4.json) non-empty, `git -C $REPO status --short analysis/geo/results` clean for them, pushed; `ls /gpfs/tmp_cpu2/yeseo1ee/geo/preds/final_f?_s?.pkl \| wc -l` = 15 | gate waits (human step) |
   | 0-2 | `ls $REPO/analysis/d1_ladder/results/{L_predictions.parquet,L_report.json,L_xgb_report.json,E2_report.json,E2_metrics.csv}` | gate STOP if missing |
   | 0-3 | `ls $REPO/analysis/d1_g3/{feat.py,g3_common.py}` | gate STOP if missing |
   | 0-4 | `ls -d /gpfs/tmp_cpu2/yeseo1ee/d1_g2/g2_final/*/.done \| wc -l` = 8,616; `ls $REPO/analysis/d1_g2/results/labels_d1_hf_g2.parquet` | parquet: gate STOP if missing; the count: by hand only (Z also flags `no_g2`) |
   | 0-5 | `/usr/lpp/mmfs/bin/mmlsquota --block-size G` : >= 30 GB free (quota shared by /home1 and /gpfs/tmp_cpu2) | by hand only |
   | 0-6 | `grep -A3 '"selftest"' $REPO/analysis/geo2/results/selftest/E_report.json` shows `"all_pass": true` (job 1053810) | gate STOP if not true |
3. **Optional dry runs** (recommended once, after the code freeze; both read only, ~1 min):
   ```bash
   cd $REPO/analysis/geo2
   sbatch -p cpu1,cpu2 -J geo2_l_selftest --time=48:00:00 --cpus-per-task=1 --mem=2G --output=$L/%x_%j.out \
     --wrap "cd $REPO/analysis/geo2 && PYTHONDONTWRITEBYTECODE=1 exec /home1/yeseo1ee/miniconda3/envs/reactot/bin/python g2_launch.py selftest"
   sbatch -p cpu1,cpu2 -J geo2_l_status --time=48:00:00 --cpus-per-task=1 --mem=2G --output=$L/%x_%j.out \
     --wrap "cd $REPO/analysis/geo2 && PYTHONDONTWRITEBYTECODE=1 exec /home1/yeseo1ee/miniconda3/envs/reactot/bin/python g2_launch.py status"
   ```
   - selftest done: log ends with `"all_pass": true`, exit 0 (it also prints every task's sbatch line). STOP: exit 3,
     `problems` lists the missing job file / script / dependency / unit-order difference.
   - status: prints the task table (status per task) and `READY <task>: <sbatch line>` for what the first pass would
     submit; writes nothing.

## 3. Start the launcher (one line, login node)

```bash
cd $REPO/analysis/geo2
squeue -u $USER -h -r | wc -l        # must be <= 18 (launcher + its first submission)
sbatch -p cpu1,cpu2 -J geo2_launch --time=48:00:00 --cpus-per-task=1 --mem=2G --output=$L/%x_%j.out \
  --wrap "cd $REPO/analysis/geo2 && PYTHONDONTWRITEBYTECODE=1 exec /home1/yeseo1ee/miniconda3/envs/reactot/bin/python g2_launch.py"
```
- It may be started now: Y (running) is adopted by name, `geo2_gate_geo1` waits for the GEO1 commit + push, and
  nothing but a Y resubmission (if a Y job is lost) is submitted before that gate (T, the early C regressions and the
  E3 selftest wait for it too, O-13).
- Each pass (a few seconds to ~1 min): squeue / sacct, judge all 48 tasks, submit, write
  `$R/{state.json,status.json,STATUS.txt}` and `results/STOP_<phase>.md`, then submit the next pass
  (`geo2_launch`, `--begin=now+10minutes`) and exit. Restarting the same line at any time is safe (a lock and the
  "a pass is already pending" check keep one chain; state is idempotent).
- The interpreter is reactot's python by absolute path without `conda activate` (stdlib only), so every task inherits
  the environment of the shell that ran this `sbatch` line (as if the runbook lines were typed there).

## 4. Tasks (exact sbatch lines the launcher submits; `--parsable` added by the launcher)

Common part of every line: `--time=48:00:00 --output=/gpfs/tmp_cpu2/yeseo1ee/geo2/logs/%x_%j.out`. `CPU` =
`-p cpu1,cpu2`, `GPU` = `-p gpu1,gpu2,gpu3,gpu5,gpu6 --gres=gpu:1`. Submit dir = `cd` target (the wrappers use
`$SLURM_SUBMIT_DIR`). Every array of the role runbooks is submitted as separate jobs with the explicit index (O-14).

| task | phase | cd (submit dir) | resources | --export | job file + command |
|---|---|---|---|---|---|
| geo2_y_plan | Y | analysis/geo | CPU 4 cpu 32G | ALL | `jobs/gpu.slurm train_m1.py plan` |
| geo2_y_s{0,1,2} | Y | analysis/geo | GPU 4 cpu 32G | ALL,GEO_MAX_EP=300,GEO_PATIENCE=100000 | `jobs/gpu.slurm train_m1.py final {s}` |
| geo2_z | Z | analysis/geo2 | CPU 4 cpu 32G | ALL | `../geo/jobs/gpu.slurm build_geo.py build` |
| geo2_t2g | T | analysis/geo2 | CPU 3 cpu 8G | ALL,GEO2ENV=d1 | `jobs/geo2.slurm feat_geo.py t2_g` |
| geo2_t2f | T | analysis/d1_pair | CPU 8 cpu 32G | ALL,PYENV=venv5,D1PAIR_PHASE=F | `jobs/g.slurm ../geo2/feat_geo.py t2_f` |
| geo2_t3 | T | analysis/d1_hf | CPU 3 cpu 16G | ALL | `jobs/hf.slurm ../geo2/feat_geo.py t3` |
| geo2_hip | H | analysis/d1_hf | CPU 48 cpu 180G | ALL | `jobs/hf.slurm ../geo2/feat_geo.py h_inputs 5 ::: ../geo2/feat_geo.py h_pilot` |
| geo2_hb{k} (k = 0..4) | H | analysis/d1_hf | CPU 48 cpu 180G | ALL | `jobs/hf.slurm ../geo2/feat_geo.py h_batch {k} 5` |
| geo2_hr | H | analysis/d1_hf | CPU 4 cpu 32G | ALL | `jobs/hf.slurm ../geo2/feat_geo.py h_build ::: ../geo2/feat_geo.py h_report` |
| geo2_g{i} (i = 0..3) | G | analysis/geo2 | CPU 16 cpu 32G | ALL,GEO2ENV=d1 | `jobs/geo2.slurm feat_geo.py g {i} 4` |
| geo2_gr | G | analysis/d1_pair | CPU 16 cpu 32G | ALL,PYENV=venv5,D1PAIR_PHASE=F | `jobs/g.slurm ../geo2/feat_geo.py g_report` |
| geo2_fs{i} (i = 0..3) | F | analysis/d1_pair | CPU 32 cpu 64G | ALL,PYENV=venv5,D1PAIR_PHASE=F | `jobs/g.slurm ../geo2/feat_geo.py f_slice {i} 4` |
| geo2_fr | F | analysis/d1_pair | CPU 48 cpu 96G | ALL,PYENV=venv5,D1PAIR_PHASE=F | `jobs/g.slurm ../geo2/feat_geo.py f_rest` |
| geo2_p | P | analysis/geo2 | CPU 4 cpu 32G | ALL,GEO2ENV=reactot | `jobs/geo2.slurm feat_geo.py p ';;;' feat_geo.py p_report ';;;' feat_geo.py report` |
| geo2_creg_g | C | analysis/d1_ladder | GPU 4 cpu 32G | ALL | `jobs/gpu.slurm tests/run_tests.py geo gpu --regress-only` |
| geo2_creg_c | C | analysis/d1_pair | CPU 32 cpu 64G | ALL,PYENV=espley | `jobs/py.slurm ../d1_ladder/tests/run_tests.py geo cpu --regress-only` |
| geo2_creg_m | C | analysis/d1_ladder | CPU 1 cpu 4G | ALL,LADENV=reactot | `jobs/lad.slurm tests/run_tests.py geo merge --regress-only` |
| geo2_ctest_g | C | analysis/d1_ladder | GPU 4 cpu 32G | ALL | `jobs/gpu.slurm tests/run_tests.py geo gpu` |
| geo2_ctest_c | C | analysis/d1_pair | CPU 32 cpu 64G | ALL,PYENV=espley | `jobs/py.slurm ../d1_ladder/tests/run_tests.py geo cpu` |
| geo2_ctest_m | C | analysis/d1_ladder | CPU 1 cpu 4G | ALL,LADENV=reactot | `jobs/lad.slurm tests/run_tests.py geo merge` |
| geo2_e3t | E | analysis/d1_pair | CPU 4 cpu 32G | ALL,PYENV=espley | `jobs/py.slurm ../d1_ladder/evaluate.py e3-selftest` |
| geo2_R | R | analysis/d1_ladder | CPU 1 cpu 8G | ALL,LADENV=reactot | `jobs/lad.slurm ../geo2/prereg_geo2.py write` |
| geo2_Rc | R | analysis/d1_ladder | CPU 1 cpu 4G | ALL,LADENV=reactot | `jobs/lad.slurm ../geo2/prereg_geo2.py check` |
| geo2_x3 | M3 | analysis/d1_pair | CPU 32 cpu 64G | ALL,PYENV=espley | `jobs/py.slurm ../d1_ladder/train_xgb.py run 0 1 --stage 3` |
| geo2_x3c | M3 | analysis/d1_pair | CPU 2 cpu 16G | ALL,PYENV=espley | `jobs/py.slurm ../d1_ladder/train_xgb.py check --stage 3` |
| geo2_m3_{k} (k = 0..3) | M3 | analysis/d1_ladder | GPU 4 cpu 16G (retry: `--gres=gpu:<type>:1`, partitions of that type) | ALL | `jobs/gpu.slurm train_atoms.py run {k} 4 --stage 3` |
| geo2_m3col | M3 | analysis/d1_ladder | CPU 2 cpu 24G | ALL | `jobs/gpu.slurm train_atoms.py collect --stage 3 --with-xgb` |
| geo2_Rc2 | M3 | analysis/d1_ladder | CPU 1 cpu 4G | ALL,LADENV=reactot | `jobs/lad.slurm ../geo2/prereg_geo2.py check` |
| geo2_e | E | analysis/d1_pair | CPU 4 cpu 32G | ALL,PYENV=espley | `jobs/py.slurm ../geo2/eval_geo2.py run` |
| geo2_e3 | E | analysis/d1_pair | CPU 4 cpu 32G | ALL,PYENV=espley | `jobs/py.slurm ../d1_ladder/evaluate.py e3` |

Example of a full line (what `status` prints for a ready task):
```bash
cd $REPO/analysis/d1_hf && sbatch -p cpu1,cpu2 -J geo2_hb3 --time=48:00:00 --cpus-per-task=48 --mem=180G \
  --output=/gpfs/tmp_cpu2/yeseo1ee/geo2/logs/%x_%j.out --export=ALL jobs/hf.slurm ../geo2/feat_geo.py h_batch 3 5
```

### Done markers and STOP of each task (judged only while the task has no job in squeue)

| task | done (sticky once true) | STOP (→ `results/STOP_<phase>.md`) |
|---|---|---|
| geo2_y_plan | `analysis/geo/results/Y_plan.json` | exit 3 |
| geo2_y_s{s} | `/gpfs/tmp_cpu2/yeseo1ee/geo/preds/final_fm1_s{s}.pkl` | exit 3 |
| geo2_y_budget (check) | the 3 pkl + sacct GPU-h of all `geo2_y_s*` attempts since 2026-10-30 07:00 <= 30 | that sum > 30 GPU-h (spec 12 / 14 Y; measured, train.py writes no per-epoch history; jobs are never cancelled by the launcher) |
| geo2_gate_geo1 (gate) | 0-1 committed + pushed + 15 GEO1 pkl, 0-2..0-4 files, 0-6 selftest | 0-2..0-4 file missing or 0-6 not all_pass |
| geo2_z | `results/Z_report.json` `stop: false`, `n_folders > 0` | `stop: true` (`stop_reasons`), exit 3 |
| geo2_t2g | `results/F_steps/t2_g.json` rc 0 | rc 3 / exit 3 |
| geo2_t2f, geo2_t3 | `results/T_2.json` / `T_3.json` `verdict: pass` | `verdict: fail`, exit 3 |
| geo2_hip | `results/H_pilot.json` `pass: true, complete: true` + `results/hf_geo/H_batches/plan.json` | `projection_core_h` > 2,500 (spec 6-1), `stop: true`, `F_steps/h_inputs.json` rc 3, exit 3 |
| geo2_hb{k} | `results/F_steps/h_batch_w{k}of5.json` rc 0 | rc 3 (cumulative run failures > 1 %), exit 3 |
| geo2_hr | `results/H_report.json` `complete: true, stop: false` | `stop: true` (run failures, rel-energy gate > 0.5 %), `F_steps/h_build.json` rc 3, exit 3 |
| geo2_g{i} | `results/F_steps/g_0{i}of04.json` rc 0 | rc 3, exit 3 |
| geo2_gr | `results/F_steps/g_report.json` rc 0 + `results/G_report.json` `complete: true, stop: false` | `G_report.json` `stop: true` (order > 0.5 %, Freq > 1 %, B6 > 1 %), rc 3 |
| geo2_fs{i} | `results/F_steps/f_slice_0{i}of04.json` rc 0 | rc 3, exit 3 |
| geo2_fr | `results/F_steps/f_report.json` rc 0 | rc 3 of f_report / f_merge / f_atoms_00of01 / f_gate / f_tokens, exit 3 |
| geo2_p | `results/P_build_geo.json` `all_pass: true` + `P_geo_report.json` `stop: false` + `F_report.json` `complete: true, stop: false` | any of them `stop: true` (P-a, \|P_23\| < 0.98 \|P_g2\|), exit 3 |
| geo2_creg_{g,c}, geo2_ctest_{g,c} | part json `results/C_tests_geo{_regress,}_{gpu,cpu}.json` `running: false, status: pass, final_inputs: true` | `status: fail`, exit 3 |
| geo2_creg_m | `results/C_tests_geo_regress.json` `all_pass: true` | `failed` non-empty, exit 3 |
| geo2_ctest_m | `results/C_tests_geo.json` `all_pass: true` | `failed` non-empty; `gpu_estimate_geo.geo_gpu_hours` (or `analysis/d1_ladder/results/C_smoke_atoms_geo.json` `estimate_geo`) > 80 (spec 8-4); exit 3 |
| geo2_e3t | `results/e3_selftest/E3_selftest.json` `ok: true` | `ok: false`, exit 3 |
| geo2_R | `results/PREREG_GEO2.md` + `results/prereg_geo2.json` | exit 3 (`STOP: ...` in the log: C not current, smoke > 80, set inconsistency, LADDER file changed, ...) |
| geo2_gate_prereg (gate) | both PREREG files tracked, unchanged since HEAD, last commit on a remote-tracking branch | — (waits) |
| geo2_Rc | `results/R_check_geo2.json` `ok: true`, newer than `prereg_geo2.json` | `ok: false` (`bad`), exit 3 |
| geo2_x3 | `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/units/L__geo__{S,SH}__XGB__f{0..4}.json` (10) | exit 3 |
| geo2_x3c | `analysis/d1_ladder/results/L_xgb_report_geo.json` `ok: true`, newer than the 10 unit json | exit 3 |
| geo2_m3_{k} | its units `L__geo__<arm>__<model>__f<f>.json` (train_atoms unit_list(3)[k::4]) or its last job COMPLETED exit 0 | exit 3 (gate, fingerprint, gate M-1′, leak) |
| geo2_m3col | `analysis/d1_ladder/results/{L_predictions_geo.parquet, L_val_geo.csv, L_report_geo.json}` | exit 3; exit 2 twice (units missing: one of the 40 units absent = spec 8-4 STOP) |
| geo2_Rc2 | `results/R_check_geo2.json` `ok: true`, newer than `L_report_geo.json` | `ok: false` (LADDER result changed = spec 14), exit 3 |
| geo2_e | `results/E_report.json` `mode: run`, newer than `L_predictions_geo.parquet` | exit 3, or a non-zero exit with a log line `STOP: ...` (eval_geo2 SystemExit) |
| geo2_e3 | `results/E3_report.json`, newer than `L_predictions_geo.parquet` | exit 3 |

Every job task besides: a non-zero exit with a log line starting with `STOP` = STOP; an end without the done marker
otherwise (crash, exit 2, OOM, timeout, node failure, no sacct record after 30 min) = lost → resubmitted once, the second
loss = dead (STOP file with the log tail); a job cancelled with `scancel` = STOP (never resubmitted, O-7).

## 5. What the orchestrator does by hand

1. **Start** (sections 2–3), once the code is frozen.
2. **GATE `geo2_gate_geo1` (spec 1 0-1)** — when GEO1's last jobs ended (E_report.json at 09:20; the GEO1 launcher
   reported "all R tasks and the evaluation finished"): commit `analysis/geo/results/` (E_report.json, E_rxn.csv,
   R_select_f0..4.json and the other GEO1 result files) on branch `geo1`, push. The next pass (<= 10 min) sees it and
   submits T, the C regressions, the E3 selftest, then Z as soon as Y is done. Then create `geo2` from that HEAD
   (`git switch -c geo2`, spec header); switching does not change the working files, so running jobs are unaffected.
   Keep `docs/specs/GEO2.md` + the GEO2 code commits on `geo2`.
3. **GATE `geo2_gate_prereg` (spec 8-3 / 14; RUNBOOK_lad R-2)** — `STATUS.txt` shows
   `GATE geo2_gate_prereg: R-2 human step ...` after `geo2_R` wrote the PREREG. Read
   `analysis/geo2/results/PREREG_GEO2.md`; then one commit on `geo2` holding at least `results/PREREG_GEO2.md`,
   `results/prereg_geo2.json`, `prereg_geo2.py`, `analysis/d1_ladder/{train_atoms,train_xgb,inputs}.py`,
   `RUNBOOK_lad.md`, `results/DECISIONS_lad.md` (and the rest of the GEO2 code / runbooks / decisions not committed
   yet); `git push`. The launcher never commits; it continues with `geo2_Rc` and then M3 training (no training job is
   submitted before this gate). Do not edit any gated file after this commit.
   - To regenerate the draft before the commit (e.g. a review finding fixed upstream): `mv` both PREREG files aside,
     `touch $R/retry/geo2_R` (the launcher writes them again).
4. **A STOP** (`ls $REPO/analysis/geo2/results/STOP_*.md`): read the file (task, report key, job ids, log lines,
   the exact resubmission line), report to the user what went wrong (file and line, options) and wait (spec 14).
   After the fix / user decision:
   - report-json STOP: move the report aside (or rerun the step so that it is rewritten) and
     `touch $R/retry/<task>`; the next pass judges it again;
   - a user-accepted rate STOP (RUNBOOK_feat 5 / 6: `g_report --accept=B6`, `p --accept=F-1[,B6]`): submit that
     accept line by hand **with the launcher's task name** (`-J geo2_gr` / `-J geo2_p`): the launcher adopts it, and
     once its done marker holds the chain continues (the STOP file of the accepted gate stays as the record);
   - `STOP_LAUNCHER.md` = the launcher itself crashed: report, fix, resubmit the start line (section 3).
   - an XGB STOP "GroupKFold(4) fold 0 differs from the split file" (`geo2_creg_c`, `geo2_ctest_c`, `geo2_x3`; node-type
     tie order, RUNBOOK_tests T-R / RUNBOOK_lad S-3): resubmit that task's line from the STOP file by hand with the
     same `-J` plus `--nodelist=n112` (or `n115`), nothing else changed; the launcher adopts it.
   When the cause no longer holds, the launcher renames the file `STOP_<phase>.resolved_<time>.md`.
5. **Finished** (`STATUS.txt` outcome `finished`: `geo2_e` and `geo2_e3` done; `results/HPC_efficiency.csv` written):
   Phase S (spec 11) — `results/SUMMARY_GEO2.md` (Korean; first four sentences, sections, costs from
   `HPC_efficiency.csv` + `results/H_pilot.json` / `H_report.json` / `L_report_geo.json`), figures
   `results/figures/figG2_{ladder,class,models}` (a CPU sbatch job), commit + push, then wait for the user.

## 6. Monitoring and controls (login node, read only unless noted)

```bash
cat $R/STATUS.txt                                   # counts, gates waiting, STOPs, one line per task
squeue -u $USER -h -r -o "%i|%j|%T|%M|%R"           # geo2_launch PENDING (BeginTime) between passes
ls $REPO/analysis/geo2/results/STOP_*.md 2>/dev/null
tail -30 $(ls -t $L/geo2_launch_*.out | head -1)    # the last pass
```
- The chain is alive when `squeue` shows one `geo2_launch` job (PENDING with reason BeginTime, or RUNNING). If none is
  queued and `STATUS.txt` says `outcome running` (e.g. the last relaunch was refused at MaxSubmit and polling also
  failed), resubmit the start line (section 3).
- `outcome stalled` + exit 3 in the last launcher log = nothing can progress (only STOPs left): handle the STOP
  files, then `touch $R/retry/<task>` and resubmit the start line.
- **HALT:** `touch $R/HALT` — the next pass submits nothing and does not resubmit itself (running tasks keep
  running). Resume: `rm $R/HALT`, resubmit the start line.
- **Retry a task:** `touch $R/retry/<task>` — the next pass forgets its attempts (tries 0) and sticky done, keeps the
  M3 GPU-type pin, and judges it from its files again (a task whose done marker still holds is done again at once:
  move the marker aside first to really rerun it).
- **Cancel:** `scancel <id>` of a task is never undone by the launcher (STOP `cancelled`); `touch $R/retry/<task>` to
  let it run again. Ask the user before cancelling a running job.
- **After a code edit while running** (only with a user decision): `touch $R/HALT`, edit, move aside
  `results/T_2.json`, `T_3.json`, `results/F_steps/t2_g.json` (T reruns on the new code: feat_geo F-16), `touch
  $R/retry/{geo2_t2g,geo2_t2f,geo2_t3}`, `rm $R/HALT`, resubmit the start line. After R-2 no gated file may change.

## 7. Budget (queue slots, expected walls)

| stage | tasks queued at once | expected |
|---|---|---|
| Y (running) | 3 GPU | ~6.5–8.7 h per seed from 08:02 (DECISIONS Y-cost), sum <= 30 GPU-h |
| after gate geo1 | t2g, t3, creg_g (GPU), creg_c, e3t | T ~1 h, C regressions ~1–2 h (RUNBOOK_tests) |
| after Y + Z + T | hip, then hb0..4 (5 × 48 cpu) + g0..3 → gr → fs0..3 → fr | pilot ~1–2 h, H ~7 h per worker (~1,600 core-h), G/F ~1–2 h |
| after P | ctest_g (GPU), ctest_c → ctest_m → R | a few hours |
| after the PREREG gate | Rc, then x3 (CPU) + m3_0..3 (4 GPU) → x3c → m3col → Rc2 | 15–50 GPU-h (limit 80), XGB ~3–4 h |
| E | e, e3 | minutes |

GEO2 holds at most 10 running tasks (SLURM MaxJobs, shared with the launcher pass while it runs) + 2 pending per class
+ the launcher's next pass, i.e. <= 15 submit slots; the launcher never lets the whole queue (other projects included)
exceed 19 before its own next pass is submitted.
