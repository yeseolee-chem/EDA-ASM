# RUNBOOK — GEO2b LADDER training slice (design steps 9–12: D0+D1 labels / splits, rung geo, prereg, M3)

Authority: `docs/specs/GEO2.md` 8-1 .. 8-4 read with `docs/specs/GEO1B.md` 4-3 and the user decision D0D1 (GEO1b
`analysis/geo1b/results/DECISIONS.md`, design `D0D1_design_investigation.json` steps 9–12). Decisions: `results/DECISIONS_lad.md`.
Tests: `RUNBOOK_tests.md`. The single end-to-end run order of GEO2b (Z, inputs, this slice, E; merge points; queue
arithmetic) is `RUNBOOK_geo2b.md`. Code: `analysis/d1_ladder/{train_atoms.py, train_xgb.py, tests/run_tests.py}` (stage 3 = rung geo,
split T1D0), `analysis/geo2/{d0d1_splits.py, prereg_geo2b.py, lad_selftest.py, m3_launch.sh, jobs/m3_gpu.slurm, jobs/m3_cpu.slurm}`.

## 0. Rules for every step
- Login node: only `sbatch` / `squeue` / `scancel` / `git` / reading files, and `bash analysis/geo2/m3_launch.sh <step>`
  (it calls only `squeue` and `sbatch`, no loop, nothing resident). No python on the login node.
- MaxJobs 10 running, MaxSubmit 20 queued **tasks** (array elements count). `m3_launch.sh` refuses (exit 2) when
  `squeue -u $USER -h -r | wc -l` + the step's tasks > 20, and when a job name of the step is still running / pending
  (GEO1B.md 6: never the same job twice; a dead task is resubmitted alone, M3 below). Every job: `--time=48:00:00`, partitions `cpu1,cpu2` /
  `gpu1,gpu2,gpu3,gpu5,gpu6` (SLURM takes the one with room), logs `/gpfs/tmp_cpu2/yeseo1ee/geo2b/logs` (`$GL`).
- Exit codes of every python command: 0 done, 2 not ready (an input missing; nothing written), 3 STOP (read the log line
  `STOP: ...`, report, wait). Every step is atomic + idempotent: resubmitting the same line is safe.
- Never run a LADDER stage-1 / 2 write command from this worktree (`train_atoms.py smoke|run|collect` without
  `--stage 3`, `train_atoms.py estimate`, `train_xgb.py smoke|run|check|l6` without `--stage 3`, `tests/run_tests.py
  gpu|cpu|merge`): the code refuses (exit 3, GA-13 / XG-G9 / G-17), LADDER results stay byte-identical.
- `--dry-run` on any `m3_launch.sh` step prints the exact sbatch lines without submitting.

```
REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b ; GL=/gpfs/tmp_cpu2/yeseo1ee/geo2b/logs
```

## 1. Order and dependencies
```
GEO1b: G2c build -> d0_pairs.py build (final, g2_source g2c) -> D/U/R/E (+ Y) -> commit on geo1b
  S-0  bring the final GEO1b results into geo2b (git; analysis/geo1b/results/D0_pairs_report.json, Q_d0_report.json, E_*)
  S-1  m3_launch.sh splits            -> $GEO2B/F_pairs_d0d1.csv, $GEO2B/splits/T1D0__f<k>.json, results/S_splits_d0d1.json
Z (build_geo, Z/E role) -> G / F / H / P (inputs role) -> results/P_build_geo.json + P_geo_report.json, $GEO2B/inputs/geo.parquet
  C    m3_launch.sh ctests            -> results/C_tests_geo.json all_pass (incl. the GPU smoke, XGB smoke, diag, R-1..R-3)
  R-1  m3_launch.sh prereg            -> results/PREREG_GEO2B.md + prereg_geo2b.json
  R-2  review + commit + push (login): both PREREG files + the final code of this slice (code freeze point)
  M3   m3_launch.sh m3 [--workers 9]  -> 40 units in $LAD/units/L__geo__*, results/L_predictions_geo.parquet,
                                         L_report_geo.json, L_xgb_report_geo.json, L_val_geo.csv; LADDER files re-checked
```
Code freeze: every edit of `train_atoms.py`, `train_xgb.py`, `inputs.py` (inputs role), `lad_common.py`,
`tests/run_tests.py`, `analysis/geo2/d0d1_splits.py` must come BEFORE the final C tests; prereg STOPs when C_tests_geo.json
`code_sha256` differs from them (evaluate.py is recorded only, LP-3), and every training job re-checks the five
training files (gate_c_geo) and the PREREG `inputs_sha256`.

## 2. Steps

### S-0 GEO1b final results into this worktree (git, login; orchestrator) = merge point M1 of RUNBOOK_geo2b.md (file list there)
- After GEO1b `d0_pairs.py build` ran with `g2_source` `g2c` (preliminary false, status ok) on branch geo1b, bring at least
  `analysis/geo1b/results/{D0_pairs_report.json, Q_d0_report.json}` (and the GEO1b E results the Z step needs) into
  geo2b. d0d1_splits.py reads THIS worktree's reports: exit 2 while they are preliminary (연결 S-4), STOP when they do
  not match `$GEO1B/d0/{D0_pairs.csv, d0_folds.csv, d0_twins.csv}` (LD-4 / LD-5).

### S-1 D0 + D1 labels and splits (design step 9) — 1 CPU task, < 1 min
```bash
cd $REPO && bash analysis/geo2/m3_launch.sh splits
# = sbatch --time=48:00:00 -p cpu1,cpu2 -J geo2b_splits --cpus-per-task=2 --mem=8G --output=$GL/%x_%j.out \
#     --export=ALL,M3ENV=reactot jobs/m3_cpu.slurm ../geo2/d0d1_splits.py build ::: ../geo2/d0d1_splits.py check
```
- Inputs: `$LAD/F_pairs_cr2.csv` (+ `analysis/d1_ladder/results/V_report.json`), `analysis/d1_pair/results/F_pairs.csv`,
  FU2 `S_splits.json` + `T1__f<k>.json`, `$GEO1B/d0/{D0_pairs.csv, d0_folds.csv, d0_twins.csv}`,
  `analysis/geo1b/results/{D0_pairs_report.json, Q_d0_report.json}`.
- Outputs: `$GEO2B/F_pairs_d0d1.csv` (8,389 D1 rows byte-identical to F_pairs_cr2.csv + every D0_pairs.csv row),
  `$GEO2B/splits/T1D0__f{0..4}.json`, `analysis/geo2/results/S_splits_d0d1.json` (written last; `all_pass` true,
  `preliminary` false).
- Done: log `=== exit 0` twice. Exit 2 (not ready, nothing written): an input missing or the GEO1b D0 build still
  preliminary (연결 S-4: merge M1 first). STOP (exit 3): inconsistent GEO1b D0 build, a D0 pair across folds, a D0 /
  D1 rxn_id overlap, a twin bound to a training fold, cr2 provenance. Rerun is byte-identical for the same inputs.
- Preliminary dry run (tests only, already done 2026-10-31): `d0d1_splits.py build --test-dir
  $GEO2B/tests_lad/splits_prelim --allow-preliminary` (3,804 D0 pairs, 3,564 ok; RUNBOOK_tests T-0).

### C  tests (design step 10): see RUNBOOK_tests.md (C-geo). Needs S-1, Z, and the inputs role's P.

### R-1 pre-registration write (design step 11) — 1 CPU task
```bash
cd $REPO && bash analysis/geo2/m3_launch.sh prereg
# = sbatch --time=48:00:00 -p cpu1,cpu2 -J geo2b_prereg --cpus-per-task=1 --mem=8G --output=$GL/%x_%j.out \
#     --export=ALL,M3ENV=reactot jobs/m3_cpu.slurm ../geo2/prereg_geo2b.py write
```
- Needs (exit 2 while missing): GEO1b `E_report.json`, `results/Z_report.json`, `P_build_geo.json`, `P_geo_report.json`,
  final `S_splits_d0d1.json`, `C_tests_geo.json` all_pass, `analysis/d1_ladder/results/{C_smoke_atoms_geo.json,
  C_smoke_xgb_geo.json, D0D1_diag_geo.json}`, the gated files. Also exit 2 while `Z_report.json` is running / not
  complete or `P_geo_report.json` is running / of another geo table (연결 S-2, S-3).
- Outputs: `analysis/geo2/results/PREREG_GEO2B.md`, `prereg_geo2b.json`.
- STOP: a PREREG file already tracked (written once), Z / P stop, D0D1 index not final, C tests not on the code / inputs
  now, smoke not ok, GPU projection missing or > 160 GPU-h, a geo unit / prediction / evaluation file exists, a tracked
  LADDER result differs from HEAD, a quoted spec heading missing.

### R-2 review, commit, push (login; orchestrator) — the code freeze point
- Read `PREREG_GEO2B.md`; one commit with `analysis/geo2/results/{PREREG_GEO2B.md, prereg_geo2b.json}` and the final code
  (this slice's files, inputs.py of the inputs role, d0d1_splits.py); push. Nothing may edit gated files afterwards.

### M3 training (design step 12) — 14 tasks queued (9 GPU + 5 CPU), at most 10 running
```bash
cd $REPO && bash analysis/geo2/m3_launch.sh m3 --workers 9
```
which submits, in order (from analysis/geo2):
```bash
JRC=$(sbatch --parsable --time=48:00:00 -p cpu1,cpu2 -J geo2b_Rc --cpus-per-task=1 --mem=4G --output=$GL/%x_%j.out \
      --export=ALL,M3ENV=reactot jobs/m3_cpu.slurm ../geo2/prereg_geo2b.py check)
JX=$(sbatch --parsable --time=48:00:00 -p cpu1,cpu2 -J geo2b_x3 --cpus-per-task=32 --mem=64G --output=$GL/%x_%j.out \
     --dependency=afterok:$JRC --kill-on-invalid-dep=yes --export=ALL,M3ENV=espley \
     jobs/m3_cpu.slurm train_xgb.py run 0 1 --stage 3)
JXC=$(sbatch --parsable --time=48:00:00 -p cpu1,cpu2 -J geo2b_x3c --cpus-per-task=2 --mem=16G --output=$GL/%x_%j.out \
      --dependency=afterok:$JX --kill-on-invalid-dep=yes --export=ALL,M3ENV=espley \
      jobs/m3_cpu.slurm train_xgb.py check --stage 3)
JM=$(sbatch --parsable --time=48:00:00 -p gpu1,gpu2,gpu3,gpu5,gpu6 --gres=gpu:1 -J geo2b_m3 --cpus-per-task=4 --mem=32G \
     --array=0-8 --output=$GL/%x_%A_%a.out --dependency=afterok:$JRC --kill-on-invalid-dep=yes \
     jobs/m3_gpu.slurm train_atoms.py run @I 9 --stage 3)
JCO=$(sbatch --parsable --time=48:00:00 -p cpu1,cpu2 -J geo2b_m3col --cpus-per-task=2 --mem=48G --output=$GL/%x_%j.out \
      --dependency=afterok:$JM:$JXC --kill-on-invalid-dep=yes --export=ALL,M3ENV=d1fu2 \
      jobs/m3_cpu.slurm train_atoms.py collect --stage 3 --with-xgb)
sbatch --time=48:00:00 -p cpu1,cpu2 -J geo2b_Rc2 --cpus-per-task=1 --mem=4G --output=$GL/%x_%j.out \
      --dependency=afterok:$JCO --kill-on-invalid-dep=yes --export=ALL,M3ENV=reactot jobs/m3_cpu.slurm ../geo2/prereg_geo2b.py check
```
- Gates inside every training job (exit 3 if any fails, before any unit): PREREG_GEO2B (committed, sha256, rungs geo,
  split T1D0, inputs, LADDER snapshot), GEO2b P (`P_build_geo.json`, `P_geo_report.json`), the D0D1 index (final,
  current), `C_tests_geo.json` all_pass with the training code's sha256.
- Units: `$LAD/units/L__geo__{S,SH}__{AW,GNN,TF}__f{0..4}.{json,parquet}` + `__s0..4.pt` (atom), `L__geo__{S,SH}__XGB__f{0..4}`
  (json written last = done marker). A cached unit with another fingerprint = exit 3.
- Run order `[i::9]` of the 30 atom units (GNN first): task 0 holds 2 GNN units (the longest slice; the smoke projection
  `busiest_slice_hours` gives the expected wall per worker count). `--workers 10` (one GNN per task) is possible only
  if nothing else runs (MaxJobs 10 incl. the XGB job).
- Resubmit a dead atom task k: the JM line with `--array=<k>` and no `--dependency`, on the partition of the GPU type its
  finished units ran on (log `[geo2b_gpu] ... gpu=<name>`); then the collect line with `--dependency=afterok:<new>:$JXC`
  and the final prereg check line (geo2b_Rc2) with `--dependency=afterok:<new collect>` (both were cancelled by
  `--kill-on-invalid-dep`). A dead XGB job: the JX line without `--dependency`, then JXC, then collect, then Rc2.
  Never rerun `m3_launch.sh m3` for this (it refuses while any geo2b_m3* / geo2b_x3* / geo2b_Rc* job is queued).
- Collect outputs: `analysis/d1_ladder/results/L_predictions_geo.parquet` (rung geo only, PRED_COLS; D1 pairs and D0
  test pairs with ref_kind / pos 'd0', ids D#####), `L_val_geo.csv`, `L_report_geo.json` (`units.*.val_kcal_mae`,
  `geo2.n_test_pairs` D1 / D0), `L_xgb_report_geo.json` (XGB check). LADDER `L_predictions.parquet`, `L_report.json`,
  `L_xgb_report.json` are never written; collect and the final prereg check compare the LADDER snapshot (exit 3 if any
  changed = GEO2.md 14 STOP).
- Budget: cap 160 GPU-h (GEO1b DECISIONS D0D1). Projection from the C-test smoke (`C_tests_geo.json gpu_estimate_geo`),
  measured afterwards in `L_report_geo.json gpu_hours_atoms`. XGB: `C_smoke_xgb_geo.json stage_cpu_hours_estimate`.
- Then: Phase E = `eval_d0d1.py run` (Z/E role; D1 and D0 separately; it imports the LADDER evaluate.py E2 functions
  unchanged — there is no `evaluate.py e3` and `eval_geo2.py run` exits 2 by design under D0D1; 연결 S-1). 1 CPU task:
```bash
cd $REPO/analysis/geo2 && sbatch --time=48:00:00 -p cpu1,cpu2 -J geo2b_e --cpus-per-task=4 --mem=32G \
  --output=$GL/%x_%j.out --export=ALL,M3ENV=espley jobs/m3_cpu.slurm ../geo2/eval_d0d1.py run
```
  Needs (exit 2 while missing): `L_report_geo.json`, `L_xgb_report_geo.json`, `L_predictions_geo.parquet`, GEO1b
  `E_rxn.csv`, the final D0 build (D0_pairs_report g2_source g2c); STOP if `D0_pairs.csv` != the `S_splits_d0d1.json`
  record (E-9). Done = `results/E_report.json` without `status` 'running' and the 8 tables; full order in RUNBOOK_geo2b.md.
