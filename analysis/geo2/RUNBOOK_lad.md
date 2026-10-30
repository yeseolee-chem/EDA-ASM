# RUNBOOK — role LADA: LADDER stage 3 (rung `geo`) and the GEO2 pre-registration

Spec: `docs/specs/GEO2.md` 8-1 .. 8-4 (authority). Code of this role: `analysis/d1_ladder/train_atoms.py`,
`analysis/d1_ladder/train_xgb.py`, `analysis/d1_ladder/inputs.py` (stage-3 additions; decisions in
`analysis/geo2/results/DECISIONS_lad.md`), `analysis/geo2/prereg_geo2.py`. Nothing here has been run yet.

## 0. Rules for every step
- Login node: only `sbatch`, `squeue`, `sinfo`, `scancel`, `git`, reading files. No python, no loops, no sleep.
- Before every submission: `squeue -u $USER -h -r | wc -l` = N (array elements count). Need N + (tasks of the line) <= 20;
  at most 10 running. `sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`, `sinfo -p gpu1,gpu2,gpu3,gpu5,gpu6 -o "%.9P %.6t %.10G %C"`.
- Every job: `--time=48:00:00`, log under `/gpfs/tmp_cpu2/yeseo1ee/geo2/logs` (`GL` below). The wrappers' default log
  folders are LADDER / D1PAIR scratch, so `--output` is always given.
- Do not export `GEO2_SCRATCH`, `D1LADDER_SCRATCH`, `D1PAIR_SCRATCH` with other values: the gate compares absolute paths
  built from their defaults (`/gpfs/tmp_cpu2/yeseo1ee/{geo2,d1_ladder,d1_pair}`).
- Exit codes of every python command here: 0 = done, 2 = not ready (an input is missing; resubmit later, nothing
  written), 3 = STOP (spec 14: read the log line `STOP: ...` / the report key named below, report, wait).
- Idempotent: units are cached (`$LAD/units/<unit>.json` written last, fingerprint compared on rerun); reports are
  rewritten atomically. Resubmitting the same line is always safe.
- **Never run any LADDER stage-1 / 2 command from this worktree** (DECISIONS_lad D-42, D-38): `train_atoms.py
  smoke|run|collect` without `--stage 3`, `train_atoms.py estimate`, `train_xgb.py smoke|run|check` without `--stage 3`,
  `train_xgb.py l6` without `--stage 3`, `inputs.py build|pa|pairs` and the commands that check the LADDER builds per
  rung (`inputs.py p_report <rung>`, `evaluate.py e1|e2`), `tests/run_tests.py gpu|cpu|merge` (LADDER mode). The
  trainers now refuse their stage-1 / 2 write commands (exit 3, `GA-13` / `XG-G9` in the log) because the GEO2
  additions changed their sha256; inputs.py's sha256 changed too, so `inputs.py` reports the dft / g1 / g2 builds as
  stale here (exit 2) and the only repair (`inputs.py build`) would rewrite LADDER results (spec 14 STOP). Only
  `list --stage 1|2` (regression 1) and the GEO2 commands below are run. GEO2 consumers handle the stale builds by
  D1-G3 F-13 + the pristine inputs.py sha256 (run_tests G-12, evaluate E3-2).

```bash
REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1
GL=/gpfs/tmp_cpu2/yeseo1ee/geo2/logs
```

| wrapper | env | submit from | python |
|---|---|---|---|
| `analysis/d1_ladder/jobs/gpu.slurm` | env_d1fu2 (torch; GPU or CPU partitions) | `analysis/d1_ladder` | `train_atoms.py ...` |
| `analysis/d1_pair/jobs/py.slurm` + `--export=ALL,PYENV=espley` | env_espley (xgboost, sklearn 1.3.1) | `analysis/d1_pair` | `../d1_ladder/train_xgb.py ...` |
| `analysis/d1_ladder/jobs/lad.slurm` + `--export=ALL,LADENV=reactot` | conda reactot (pandas, git) | `analysis/d1_ladder` | `../geo2/prereg_geo2.py ...` |

## 1. Order and dependencies

```
GEO1 done (E_report.json, Y_plan) ─┐
Z (build_geo) ─ G ─ F ─ H ─ P (feat_geo p + p_report; role FEAT) ─┬─ S-1 / S-2 / S-3 smokes + l6 (here or inside C)
                                                                   └─ C tests geo (role LADB) -> C_tests_geo.json all_pass
                                                                         └─ R-1 prereg write -> R-2 commit + push (login)
                                                                               └─ R-3 check ─┬─ M-1 XGB run ─ M-2 XGB check ─┐
                                                                                              └─ M-3 atom run (GPU x4) ───────┴─ M-4 collect -> E
```
Gated code (`train_atoms.py`, `train_xgb.py`, `inputs.py`, `lad_common.py`, fu2 / rev 9 code) must not change after R-1:
the PREREG records their sha256 and every training job re-checks them.

## 2. Steps

### L-1 (optional) regression 1 alone: unit lists of stages 1 / 2 unchanged (spec 8-2 회귀 1; LADB's C tests also do it)
```bash
sbatch -p cpu1,cpu2 -J geo2_reg1 --time=48:00:00 --cpus-per-task=1 --mem=8G --output=$GL/%x_%j.out \
  --chdir=$REPO/analysis/d1_ladder --wrap 'set -uo pipefail; source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh; conda activate reactot; PY=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2/bin/python; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 D1LADDER_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair; R=/gpfs/tmp_cpu2/yeseo1ee/geo2/regress; $PY train_atoms.py list --stage 1 > $R/list_stage1.after.txt && $PY train_atoms.py list --stage 2 > $R/list_stage2.after.txt && $PY train_atoms.py list --stage 3 > $R/list_stage3.txt && cmp $R/list_stage1.before.txt $R/list_stage1.after.txt && cmp $R/list_stage2.before.txt $R/list_stage2.after.txt && wc -l $R/list_stage3.txt'
```
- Inputs: the code only. Outputs: `$GEO2/regress/list_stage{1,2}.after.txt`, `list_stage3.txt` (30 lines).
- Done: job exit 0 (both `cmp` silent). STOP: exit != 0 (cmp prints the first differing line) = spec 14 R·M3 STOP.

### S-1 atom smoke, rung geo (spec 8-2 smoke; skip if LADB's `run_tests.py geo` runs `train_atoms.cmd_smoke(3, ...)`)
```bash
cd $REPO/analysis/d1_ladder
sbatch -p gpu1,gpu2,gpu3,gpu5,gpu6 --gres=gpu:1 -J geo2_smA --time=48:00:00 --cpus-per-task=4 --mem=32G \
  --output=$GL/%x_%j.out jobs/gpu.slurm train_atoms.py smoke --stage 3
```
- Inputs: `$GEO2/fu2_geo/tokens`, `analysis/geo2/results/fu2_geo/*`, `$GEO2/inputs/geo.parquet`, labels, FU2 splits;
  for the estimate `analysis/d1_ladder/results/C_smoke_atoms.json` + `L_report.json` (read only).
- Output: `analysis/d1_ladder/results/C_smoke_atoms_geo.json` (scratch `$LAD/smoke_geo`, removed first).
- Done: exit 0 and `ok: true`. STOP: exit 3 (gate M-1′ failed) or `estimate_geo.geo_gpu_hours > 80` or
  `estimate_geo.ok` not true (spec 8-4; judged by R-1, which STOPs on both). Exit 2: geo tokens / table not ready
  (FEAT F / P not finished).
- Dependencies: FEAT F (tokens) and P (table). R-1 needs this file (exit 2 while missing): the C tests (G-9) write it;
  run S-1 only if they did not.

### S-2 XGB smoke, rung geo (skip if LADB runs `train_xgb.cmd_smoke(3, ...)`)
```bash
cd $REPO/analysis/d1_pair
sbatch -p cpu1,cpu2 -J geo2_smX --time=48:00:00 --cpus-per-task=32 --mem=64G --output=$GL/%x_%j.out \
  --export=ALL,PYENV=espley jobs/py.slurm ../d1_ladder/train_xgb.py smoke --stage 3
```
- Output: `analysis/d1_ladder/results/C_smoke_xgb_geo.json` (`ok`, `stage_cpu_hours_estimate`), scratch `$LAD/smoke_xgb_geo`.
- Done: exit 0, `ok: true`. STOP: exit 3. Dependencies: as S-1 (R-1 needs this file too, `ok: true`).

### S-3 L-6 helper with the modified code (spec 8-2 회귀 2, XGB part; skip if LADB calls `train_xgb.cmd_l6(3)`)
```bash
cd $REPO/analysis/d1_pair
sbatch -p cpu1,cpu2 -J geo2_l6 --time=48:00:00 --cpus-per-task=32 --mem=64G --output=$GL/%x_%j.out \
  --export=ALL,PYENV=espley jobs/py.slurm ../d1_ladder/train_xgb.py l6 --stage 3
```
- Output: `analysis/d1_ladder/results/C_l6_xgb_geo.json` (scratch `$LAD/smoke_xgb_l6_geo`); the LADDER
  `C_l6_xgb.json` is not touched (never run `l6` without `--stage 3` in GEO2).
- Done: exit 0, `ok: true`. STOP: exit 3 (yhat > 1e-6 from rev 9 / other best cell).
- GroupKFold note (train_xgb docstring): an inner-fold mismatch STOP depends on the node type; resubmit on the rev 9
  node type (n112 / n115) instead of changing anything.

### R-1 pre-registration write (spec 8-3; after `analysis/geo2/results/C_tests_geo.json` all_pass)
```bash
cd $REPO/analysis/d1_ladder
JR=$(sbatch --parsable -p cpu1,cpu2 -J geo2_R --time=48:00:00 --cpus-per-task=1 --mem=8G --output=$GL/%x_%j.out \
  --export=ALL,LADENV=reactot jobs/lad.slurm ../geo2/prereg_geo2.py write); echo "JR=$JR"
```
- Inputs (exit 2 while missing): `analysis/geo/results/E_report.json`, `Y_plan.json` (GEO1 / Y),
  `analysis/geo2/results/Z_report.json`, `P_build_geo.json`, `P_geo_report.json`, `P_geo_sets.json`,
  `C_tests_geo.json`, `results/selftest/E_report.json` + `E_metrics.csv`, `$GEO2/inputs/geo.parquet`,
  `$GEO2/X182_d1_geo.parquet`, `analysis/geo2/results/fu2_geo/{T_report.json,T_manifest.csv,Q_report.json,S_splits.json}`,
  `analysis/d1_ladder/results/C_smoke_atoms_geo.json` + `C_smoke_xgb_geo.json` (C tests G-9 or S-1 / S-2),
  `$LAD/F_pairs_cr2.csv`, the FU2 split files, `docs/specs/GEO2.md`, `docs/specs/D1LADDER.md`.
- Code freeze order (D-28, D-39): every code edit (train_atoms / train_xgb / inputs / lad_common, and LADB's
  run_tests.py / evaluate.py) must come BEFORE the final C tests; R-1 STOPs when `C_tests_geo.json` `code_sha256` differs
  from any of those files now, or its geo table / label sha256 differs from the files now. Fix -> rerun C (both parts +
  merge) -> R-1.
- Outputs: `analysis/geo2/results/PREREG_GEO2.md`, `analysis/geo2/results/prereg_geo2.json` (done marker: both files +
  log `=== exit 0`; stdout prints md / json sha256 and the set sizes).
- STOP (exit 3, log `STOP: ...`): either file already tracked by git (written once; never rerun after the commit); Z / P
  stop; C_tests_geo not all_pass, or not run on the code / geo table / labels now (D-39); geo table sha256 !=
  P_build_geo outputs; label sha256 != V_report outputs; pair-set inconsistency or |P_23| < 0.98 |P_g2|; list sha256
  not in P_geo_report; a geo unit / prediction / evaluation file exists; a stage-3 smoke not ok, its `estimate_geo` not
  computed, or the estimate > 80 GPU-h (D-40); a tracked `analysis/d1_ladder/results` file differs from git HEAD; a spec
  heading missing.
- The json also carries `code.fingerprint_keys` (D-41): the trainers' code_sha256 that every geo unit must record;
  `evaluate.py e3` (LADB) may check the 40 geo units against it (not against the ADD3 json).
- Rerun before the commit is allowed (replaces the untracked draft; `replaced_draft` records it).

### R-2 review, commit, push (login node; orchestrator)
- Read `PREREG_GEO2.md`. Then one commit holding both PREREG files and the final code of this role (branch `geo2`):
  `analysis/geo2/results/PREREG_GEO2.md`, `analysis/geo2/results/prereg_geo2.json`, `analysis/geo2/prereg_geo2.py`,
  `analysis/d1_ladder/{train_atoms.py,train_xgb.py,inputs.py}`, `analysis/geo2/RUNBOOK_lad.md`,
  `analysis/geo2/results/DECISIONS_lad.md`; push. Nothing may edit the gated files afterwards.

### R-3 gate check without training
```bash
cd $REPO/analysis/d1_ladder
JRC=$(sbatch --parsable -p cpu1,cpu2 -J geo2_Rc --time=48:00:00 --cpus-per-task=1 --mem=4G --output=$GL/%x_%j.out \
  --export=ALL,LADENV=reactot jobs/lad.slurm ../geo2/prereg_geo2.py check); echo "JRC=$JRC"
```
- Output: `analysis/geo2/results/R_check_geo2.json` (`ok: true`). Done: exit 0. STOP: exit 3 (`bad` lists the reason:
  not committed / modified / an input or LADDER result file changed).
- Dependency: after R-2 (submit only once the commit exists).

### M-1 XGB training, 10 units (spec 8-4)
```bash
cd $REPO/analysis/d1_pair
JX=$(sbatch --parsable -p cpu1,cpu2 -J geo2_x3 --time=48:00:00 --cpus-per-task=32 --mem=64G --output=$GL/%x_%j.out \
  --export=ALL,PYENV=espley --dependency=afterok:${JRC:?} --kill-on-invalid-dep=yes \
  jobs/py.slurm ../d1_ladder/train_xgb.py run 0 1 --stage 3); echo "JX=$JX"
```
- Gates inside (exit 3 if any fails): PREREG_GEO2 (committed, sha256, inputs, LADDER snapshot), GEO2 P
  (`P_build_geo.json`, `P_geo_report.json`, `P_geo_sets.json`, label record), `C_tests_geo.json` all_pass with
  `code_sha256` of train_atoms.py / train_xgb.py / inputs.py / lad_common.py = the files now (GA-14 / XG-G10).
- Outputs: `$LAD/units/L__geo__{S,SH}__XGB__f{0..4}.{parquet,json}` (json last = done marker per unit).
- Done: exit 0. STOP: exit 3 (gate, cache fingerprint mismatch, gate M-1′, leak check). Resubmit: the same line without
  `--dependency` (finished units are skipped). Expected ~100 CPU-h (~3–4 h on 32 CPUs).

### M-2 XGB check (needed by collect --with-xgb)
```bash
cd $REPO/analysis/d1_pair
JXC=$(sbatch --parsable -p cpu1,cpu2 -J geo2_x3c --time=48:00:00 --cpus-per-task=2 --mem=16G --output=$GL/%x_%j.out \
  --export=ALL,PYENV=espley --dependency=afterok:${JX:?} --kill-on-invalid-dep=yes \
  jobs/py.slurm ../d1_ladder/train_xgb.py check --stage 3); echo "JXC=$JXC"
```
- Output: `analysis/d1_ladder/results/L_xgb_report_geo.json` (`ok: true`, 10 units). Done: exit 0. Exit 2: units
  missing (M-1 not finished). STOP: exit 3 (`bad`: fingerprint / parquet / split lists).

### M-3 atom-model training, 30 units on 4 GPU tasks (spec 8-4)
```bash
cd $REPO/analysis/d1_ladder
JM=$(sbatch --parsable -p gpu1,gpu2,gpu3,gpu5,gpu6 --gres=gpu:1 -J geo2_m3 --time=48:00:00 --cpus-per-task=4 --mem=16G \
  --array=0-3 --output=$GL/%x_%A_%a.out --dependency=afterok:${JRC:?} --kill-on-invalid-dep=yes \
  jobs/gpu.slurm train_atoms.py run @I 4 --stage 3); echo "JM=$JM"
```
- Same gates as M-1 (train_atoms copy). Each task: 7–8 units (GNN, AW, TF interleaved by the run order).
- Outputs: `$LAD/units/L__geo__{S,SH}__{AW,GNN,TF}__f{0..4}.{json,parquet}` + `__s0..4.pt`.
- Done: every task exit 0 (30 json files). STOP: exit 3 (gate, cache fingerprint, gate M-1′, leak). Resubmit a dead
  task: the same line with `--array=<k>` and without `--dependency`, on the GPU type its finished units ran on (log line
  `[lad_gpu] ... gpu=<name>`; another type = exit 3 at the first cached unit; add e.g. `--constraint` / a partition of
  that type).
- Budget: spec 12 limit 80 GPU-h (expected 15–50; S-1 `estimate_geo`). After M-4, `L_report_geo.json`
  `gpu_hours_atoms` is the measured value.

### M-4 collect (geo rows only)
```bash
cd $REPO/analysis/d1_ladder
sbatch -p cpu1,cpu2 -J geo2_m3col --time=48:00:00 --cpus-per-task=2 --mem=24G --output=$GL/%x_%j.out \
  --dependency=afterok:${JM:?}:${JXC:?} --kill-on-invalid-dep=yes jobs/gpu.slurm train_atoms.py collect --stage 3 --with-xgb
```
- Inputs: the 40 geo units, `L_xgb_report_geo.json`, `prereg_geo2.json` (LADDER snapshot re-checked).
- Outputs: `analysis/d1_ladder/results/L_predictions_geo.parquet` (rung `geo` only, LADDER columns), `L_val_geo.csv`,
  `L_report_geo.json` (`units` with `val_kcal_mae`, XGB included = the format `eval_geo2.py load_units` reads).
- Done: exit 0 and the three files. Exit 2: units missing (log `GATE: n units missing`). STOP: exit 3 (fingerprint of a
  unit != current files / code, XGB report not current, LADDER result file changed, test / validation pair lists differ).
- Then: optional `prereg_geo2.py check` again (R-3 line without dependency) = spec 14 before/after sha256 comparison of
  the LADDER results; hand-off to Phase E (`eval_geo2.py run`, `evaluate.py e3`: other roles). 40 units are required
  for E (spec 8-4: one missing = STOP).

## 3. Task budget of the M stage
R-3 1 + M-1 1 + M-2 1 + M-3 4 + M-4 1 = 8 tasks in the queue at once (N + 8 <= 20); at most 5 running together
(1 CPU XGB + 4 GPU), then 1 collect.
