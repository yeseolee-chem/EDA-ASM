# RUNBOOK — GEO2b end to end (one run order: Z, inputs, LADDER training, E; GEO1b merge points; queue arithmetic)

This file is the single run order (연결 S-1). The slice RUNBOOKs keep the details and pass checks of each step:
`RUNBOOK_feat.md` (X, S-1, T, G, F, H, P), `RUNBOOK_lad.md` (splits, prereg, M3, E), `RUNBOOK_tests.md` (self-tests,
C tests). Authority: `docs/specs/GEO2.md` read with `docs/specs/GEO1B.md` 4-3 and the user decisions D0D1 / A-G2c
(`results/DECISIONS.md`). Judgement calls: `results/DECISIONS_{ZE,feat,lad}.md` (this file's fixes: '연결 S-n').

## 0. Rules (CLAUDE.md; every line below obeys them)
- Login node: only `sbatch`, `squeue`, `git`, reading files, and `bash analysis/geo2/m3_launch.sh <step>` (squeue +
  sbatch only). No python, no loops, no sleep, no mkdir. Every job `--time=48:00:00`; every job writes its own folders.
- Exit codes of every python step: 0 done; 2 not ready (an input missing, a step still running, a preliminary GEO1b
  build, a stale report: resubmit later, nothing written); 3 STOP (read the `STOP:` line, report, wait).
- **Queue arithmetic.** MaxSubmit 20 counts array tasks, MaxJobs 10 running, 640 CPUs per account (cpu2: <= 64 per job).
  Before every submission `N=$(squeue -u $USER -h -r | wc -l)`; submit only if `N + tasks <= 20`. On 2026-10-31 ~20:55
  N = 12 (G2c: 10 g2c_dyn running on 10 x 64 = 640 CPUs + g2c_build + g2c_report pending), so wave A (5) fits now but
  starts only as G2c jobs finish (MaxJobs 10 and the 640-CPU cap are both full).
- Every dependent line has `--kill-on-invalid-dep=yes` (a job whose dependency failed is cancelled; resubmit it).
- A job killed by the wall is resubmitted with the same line (all steps are atomic and idempotent).

```bash
REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b ; R=$REPO/analysis
L=/gpfs/tmp_cpu2/yeseo1ee/geo2b/logs ; GL=$L
```

| step | tasks | submit when N <= | needs |
|---|---|---|---|
| A: X, S-1, T | 5 | 15 | nothing (now) |
| M1 merge | git | – | GEO1b finished and committed on geo1b |
| Z | 1 | 19 | M1 |
| splits | 1 | 19 | M1 |
| B1: G + g_report + H pilot | 8 | 12 | Z, T, (pilot) S-1 |
| B2: F chain | 10 | 10 | g_report passed |
| C: H batch + h_build/report | 11 | 9 | pilot passed |
| D: p, p_report, report | 1 | 19 | F + H complete, M1 |
| ctests | 3 | 17 | splits, D |
| prereg | 1 | 19 | ctests all_pass, M1 |
| M3 (`--workers 9`) | 14 | 6 | prereg committed + pushed |
| E | 1 | 19 | M3 collect, M1 |

## 1. Wave A — now (no GEO1b dependency; 5 tasks)
```bash
cd $R/d1_hf
JX=$(sbatch --parsable -p cpu1,cpu2 -J g2b_x --time=48:00:00 --cpus-per-task=4 --mem=16G --output=$L/%x_%j.out \
  jobs/hf.slurm ../geo2/d0d1.py x 4)
JS1=$(sbatch --parsable -p cpu1,cpu2 -J g2b_s1 --time=48:00:00 --cpus-per-task=3 --mem=24G --dependency=afterok:$JX \
  --kill-on-invalid-dep=yes --output=$L/%x_%j.out jobs/hf.slurm ../geo2/d0d1.py s1 3)
JTH=$(sbatch --parsable -p cpu1,cpu2 -J g2b_th --time=48:00:00 --cpus-per-task=10 --mem=64G --dependency=afterok:$JX \
  --kill-on-invalid-dep=yes --output=$L/%x_%j.out jobs/hf.slurm ../geo2/feat_geo.py t3 ';;;' ../geo2/feat_geo.py td0_h)
cd $R/geo2
JTG=$(sbatch --parsable -p cpu1,cpu2 -J g2b_tg --time=48:00:00 --cpus-per-task=8 --mem=16G --dependency=afterok:$JX \
  --kill-on-invalid-dep=yes --output=$L/%x_%j.out --export=ALL,GEO2ENV=d1 jobs/geo2.slurm \
  feat_geo.py t2_g ';;;' feat_geo.py td0_g)
cd $R/d1_pair
JTF=$(sbatch --parsable -p cpu1,cpu2 -J g2b_tf --time=48:00:00 --cpus-per-task=8 --mem=32G --dependency=afterok:$JTG \
  --kill-on-invalid-dep=yes --output=$L/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F jobs/g.slurm \
  ../geo2/feat_geo.py t2_f ::: ../geo2/feat_geo.py td0_f)
```
- Pass: `results/X_d0d1_report.json` complete / stop false (n_d0 5,255, n_x_fail 0); `results/X_S1.json` pass;
  `results/T_2.json`, `T_3.json`, `T_d0f.json`, `T_d0h.json` pass with the current code (RUNBOOK_feat 2).
- X reads this tree's preliminary `analysis/geo1b/results/D0_pairs_report.json` (d0_folds.csv only; folds are frozen in
  GEO1b, so the result stays valid after M1). Any later edit of a feat_geo CODE_REL file makes the T records stale
  (downstream exit 2): resubmit the T lines.
- Optional now: the C regressions only (RUNBOOK_tests C-reg, 3 tasks, 1 GPU); the CPU self-tests T-0 already passed.

## 2. M1 — merge the final GEO1b results into geo2b (git, login; the orchestrator)
When GEO1b has finished on branch geo1b (worktree `eda-asm-prediction-geo1`): G2c build (`select_g2c.py build`),
final `d0_pairs.py build` (D0_pairs_report.json g2_source `g2c`, preliminary false, status ok), D (`data.py`), U, R,
E1 (`evaluate.py`), E2 (`rmsd_b.py`), Y (`train_m1.py`), committed on geo1b. geo2b has not changed `analysis/geo1b`
since the branch point f010af48, so the merge is clean:
```bash
cd $REPO && git show geo1b:analysis/geo1b/results/D0_pairs_report.json | grep -m1 '"g2_source"'   # must be "g2c"
cd $REPO && git diff --stat HEAD...geo1b -- analysis/geo1b                                        # what comes in
cd $REPO && git merge --no-ff --no-edit geo1b
```
Files geo2b reads after M1 (all must be present; missing = exit 2 at the reader):
- git, `analysis/geo1b/results/`: `D0_pairs_report.json` (final), `Q_d0_report.json`, `D_dataset_report.json`
  (`g2_input.c_select_sha256`, Z 연결 S-6), `E_rxn.csv`, `E_rxn_d0.csv` (Z gate; E bins), `E_report.json` (prereg needs
  it), `E_main.csv`, `E_rmsd.json`, `Y_plan.json`, `R_select_f0..4.json` (prereg evidence), `DECISIONS.md` (D0D1 bullet,
  first match; 연결 S-9), `DECISIONS_PAIR.md`.
- scratch, not in git: `$GEO1B/preds/final_f{0..4}_s{0,1,2}.pkl` and `final_fm1_s{0,1,2}.pkl` (18 files, Z),
  `$GEO1B/d0/{D0_pairs.csv, d0_folds.csv, d0_twins.csv, labels_d0_q.json, Q_keys_d0.parquet}`; G2c
  `/gpfs/tmp_cpu2/yeseo1ee/g2c/{d1,d0}/g2_final/` and the g2c worktree `analysis/g2c/results/C_select.csv`
  (sha256 = D_dataset_report g2_input.c_select_sha256).
- `$GEO1B` = `$GEO_SCRATCH` (default `/gpfs/tmp_cpu2/yeseo1ee/geo1b`) in every GEO2b reader (연결 S-5).

**After M1 the GEO1b outputs are frozen.** GEO1b must not rerun d0_pairs.py / d0_labels.py / data.py / evaluate.py /
training / Y, nor G2c `select_g2c.py build`. Z, splits, P, C, prereg, training and E all compare sha256 records and
STOP on a change; a needed GEO1b change means Z and everything after it run again.

## 3. After M1 (Z and the splits; 1 + 1 tasks)
```bash
cd $R/geo2
JZ=$(sbatch --parsable -p cpu1,cpu2 -J g2b_z --time=48:00:00 --cpus-per-task=4 --mem=32G --output=$L/%x_%j.out \
  ../geo1b/jobs/gpu.slurm build_geo.py build)
cd $REPO && bash analysis/geo2/m3_launch.sh splits
```
- Z (GEO2.md 4, GEO1B.md 4-3; env_d1fu2 through the GEO1b job file, GEO_SCRATCH = geo1b, GEO2_SCRATCH default geo2b):
  `results/Z_report.json` stop false, running false, complete true; `results/Z_status.csv`; `$GEO2B/geo_final/<rxn_id>/`.
  Exit 2 = an upstream file missing (nothing written: M1 / GEO1b not finished). While it runs (or if it was killed)
  Z_report says running true: every reader waits (exit 2). STOP (exit 3, Z_report stop true, running false, reason):
  E_rxn reproduction, fallback > 2 % per dataset, C_select.csv != the one GEO1b read (연결 S-6), etc.
- splits: `results/S_splits_d0d1.json` all_pass, preliminary false; exit 2 while the GEO1b report is preliminary (연결 S-4).

## 4. Inputs after Z and T (RUNBOOK_feat 3-5)
- **B1 (8 tasks, N <= 12):** G array 0-5 (`g A 6`), g_report (afterany G), H pilot (`h_inputs 10 ::: h_pilot`; needs
  X_S1.json pass) — exact lines RUNBOOK_feat 3 'B1'.
- **B2 (10 tasks, N <= 10):** f_slice 0-3 (afterok g_report while it is queued, else after G_report.json passed) ->
  f_merge -> f_atoms 0-3 -> f_rest — RUNBOOK_feat 3 'B2'.
- **C (11 tasks, N <= 9):** h_batch 0-9 (64 cores each = the whole CPU cap) + h_build ::: h_report — RUNBOOK_feat 4.
- **D (1 task, after F and H complete and M1):**
```bash
cd $R/geo2
JP=$(sbatch --parsable -p cpu1,cpu2 -J g2b_p --time=48:00:00 --cpus-per-task=4 --mem=32G --output=$L/%x_%j.out \
  --export=ALL,GEO2ENV=reactot jobs/geo2.slurm feat_geo.py p ';;;' feat_geo.py p_report ';;;' feat_geo.py report)
```
  Pass: `results/P_build_geo.json` all_pass; `results/P_geo_report.json` stop false, running false, `inputs[<geo
  table>]` = its sha256 (readers check it, 연결 S-3), `d0_reference_s3d` = report-only count (연결 S-8).
- **Expected F-1 STOP for D1 (연결 S-10).** f_rest / f_report is expected to STOP on F-1 for D1 (the same feature code
  gave 1.14 % D1PAIR G1, 1.42 % LADDER F0, 1.36 % D1-G2; the D1-G2 case was accepted by the user,
  `analysis/d1_g2/results/F_DECISION.md`). User policy: take the recommended option at a spec STOP and record it. So:
  read `results/F_steps/f_report.json` (summary: F-1 rate per dataset) and `$GEO2B/F_geo/mergeroot_report.json`
  (failure kinds). If D1 **and, checked separately, D0** have an F-1 rate no worse than the precedent band
  (<= 1.42 %) and the failures are mostly the precedent kinds (`ext:B1:engine_gate1`, `ext:B1:parse`,
  `g1feat:spe_fail`), add one bullet to `results/DECISIONS_feat_runtime.md` (key `F-1 accept`: rates n / N per dataset,
  failure kinds, the three precedents and F_DECISION.md, the policy, the failed reactions leave rung geo = deviation)
  and submit wave D with `feat_geo.py p --accept=F-1 ';;;' feat_geo.py p_report ';;;' feat_geo.py report --accept=F-1`.
  Outside the band (either dataset), other failure kinds, or any other STOP (B6, Q-a, T-a, G, H): stop and report.

## 5. LADDER training (RUNBOOK_lad / RUNBOOK_tests)
```bash
cd $REPO && bash analysis/geo2/m3_launch.sh ctests          # 3 tasks (GPU part, CPU part, merge); after splits + D
cd $REPO && bash analysis/geo2/m3_launch.sh prereg          # 1 task; after C_tests_geo.json all_pass
#   then (login, orchestrator): read results/PREREG_GEO2B.md; commit both PREREG files + the final code; push
cd $REPO && bash analysis/geo2/m3_launch.sh m3 --workers 9  # 14 tasks (needs N <= 6); refuses if a step job is queued
```
- ctests: `results/C_tests_geo.json` all_pass, training_gate, labels_geo_final. Any code edit afterwards: rerun all
  three parts (prereg compares the code sha256).
- prereg: exit 2 while a needed file is missing, Z_report running / not complete, or P_geo_report running / stale.
- M3: 40 units, `L_predictions_geo.parquet`, `L_report_geo.json`, `L_xgb_report_geo.json`; dead tasks: RUNBOOK_lad M3.

## 6. E (1 task, after the M3 collect)
```bash
cd $R/geo2 && sbatch --time=48:00:00 -p cpu1,cpu2 -J geo2b_e --cpus-per-task=4 --mem=32G --output=$GL/%x_%j.out \
  --export=ALL,M3ENV=espley jobs/m3_cpu.slurm ../geo2/eval_d0d1.py run
```
- `eval_d0d1.py run` (not `eval_geo2.py run`, which exits 2 by design; there is no `evaluate.py e3`): D1 tables
  `E_metrics.csv`, `E3_metrics.csv`, `E3_bends.csv`, `E3_bend_rates.json`, `E3_goal.csv`, `E_app.csv`, `E_bin.csv`,
  D0 table `E_d0_metrics.csv`, `E_report.json` (written last). Exit 2 while an input is missing or the D0 build is
  preliminary; STOP if `D0_pairs.csv` != `S_splits_d0d1.json` d0.d0_pairs_sha256 (E-9, 연결 S-4).
- Then `results/SUMMARY_GEO2B.md` (Korean, D1 and D0 side by side, never pooled), commit, push, stop.
