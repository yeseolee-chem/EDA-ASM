# RUNBOOK — GEO2b geo-rung inputs (Phase X, S-1, T, G, F, H, P for D1 + D0)

Authority: `docs/specs/GEO2.md` 5–7 read with `docs/specs/GEO1B.md` 4-3 and the user decision D0D1 (2026-10-31;
`analysis/geo1b/results/D0D1_design_investigation.json` design steps 6–8). Decisions: `results/DECISIONS_feat.md`
(judgment calls of this slice) and the DECISIONS dicts in `feat_geo.py` / `d0d1.py` (written at run time to
`results/DECISIONS_feat_runtime.md`). Code: `analysis/geo2/{d0d1.py, feat_geo.py, g3c.py, tests_feat.py,
jobs/geo2.slurm}`, `analysis/d1_ladder/inputs.py` (rung `geo`, PG-1..PG-4).

## 0. Rules for every step
- Login node: only `sbatch`, `squeue`, reading files. No python, no loops, no mkdir. All dirs are made by the job code.
- Every job has `--time=48:00:00`. MaxJobs 10 running, MaxSubmit 20 queued **tasks** (array elements count):
  `squeue -u $USER -h -r | wc -l` before every sbatch. cpu2 QOS: at most 64 CPUs per job; cpu1 nodes have 48 cores;
  the account cap is 640 CPUs.
- Exit codes of every command: 0 done; 2 not ready / incomplete (inputs missing, a step still open: resubmit later);
  3 STOP (read the `STOP:` line in the log and `results/F_steps/<step>.json` `fail_stage` / `fail`; report, wait).
- Every step is atomic and idempotent: resubmitting the same line skips finished work. A job hit by the 48 h wall is
  resubmitted with the same line (G stops taking new reactions after 44 h; H/S-1 re-run only unfinished ORCA calcs).
- Logs: `$L/<job-name>_<jobid>.out` (arrays `_<jobid>_<task>.out`). `hf.slurm` and `g.slurm` default to read-only log
  folders of other projects, so every line below passes `--output`.

```bash
R=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo2b/analysis ; L=/gpfs/tmp_cpu2/yeseo1ee/geo2b/logs
```

Job files and environments (`';;;'` must stay quoted in bash):

| job file (submit dir) | env | commands |
|---|---|---|
| `jobs/hf.slurm` (`$R/d1_hf`) | conda reactot + ORCA 6.1.1 serial (`jobs/hf_env.sh`) | `d0d1.py x / s1`, `feat_geo.py t3 / td0_h / h_*` |
| `jobs/geo2.slurm` (`$R/geo2`), `GEO2ENV=d1` | D1 run env (python 3.9 venv, xtb 6.7.1, ORCA 6.1.1) | `t2_g`, `td0_g`, `g` |
| `jobs/geo2.slurm` (`$R/geo2`), `GEO2ENV=reactot` | conda reactot | `p`, `p_report`, `report` |
| `jobs/g.slurm` (`$R/d1_pair`), `PYENV=venv5,D1PAIR_PHASE=F` | espley_rev5 venv (tblite 0.7.0) | `t2_f`, `td0_f`, `g_report`, `f_*` |

`hf.slurm` / `geo2.slurm` separators: `:::` = next command only after exit 0, `';;;'` = next regardless (job exit =
first non-zero). `g.slurm` knows only `:::` and replaces `@I` by the array task id; `feat_geo.py` also reads `A` / `@I`
as `$SLURM_ARRAY_TASK_ID` itself.

## 1. Order and dependencies
```
now (no GEO1b / Z dependency):   X ──> S-1 (D0 B3LYP reproduction, ~3–5 core-h)
                                   └─> T: t2_g+td0_g ──> t2_f+td0_f ;  t3+td0_h
after Z (build_geo.py: results/Z_report.json stop false, complete; Z/E role) and T passed:
                                   G (6 slices) ──> g_report ──> f_slice (4) ──> f_merge ──> f_atoms (4) ──> f_rest
after Z, T and S-1 passed:         h_inputs 10 + h_pilot ──> h_batch (10) ──> h_build + h_report
after F + H complete, the final GEO1b results merged into geo2b (analysis/geo1b/results/Q_d0_report.json,
D0_pairs_report.json with g2_source g2c; $GEO1B/d0/Q_keys_d0.parquet, D0_pairs.csv, d0_folds.csv) and the LADDER
P_report / P_build_{dft,g2} / P_{dft,g2} set files as they are:
                                   p ;;; p_report ;;; report
```
Everything in G/F/H/P re-checks the combined root (`results/X_d0d1_report.json` sha256 + link digests), the four T
records against the current code, `Z_report.json`, and (H) `results/X_S1.json`.

## 2. Wave A — Phase X, S-1, T (submit now; 5 tasks)
```bash
cd $R/d1_hf
JX=$(sbatch --parsable -p cpu1,cpu2 -J g2b_x --time=48:00:00 --cpus-per-task=4 --mem=16G --output=$L/%x_%j.out \
  jobs/hf.slurm ../geo2/d0d1.py x 4)
JS1=$(sbatch --parsable -p cpu1,cpu2 -J g2b_s1 --time=48:00:00 --cpus-per-task=3 --mem=24G --dependency=afterok:$JX \
  --output=$L/%x_%j.out jobs/hf.slurm ../geo2/d0d1.py s1 3)
JTH=$(sbatch --parsable -p cpu1,cpu2 -J g2b_th --time=48:00:00 --cpus-per-task=10 --mem=64G --dependency=afterok:$JX \
  --output=$L/%x_%j.out jobs/hf.slurm ../geo2/feat_geo.py t3 ';;;' ../geo2/feat_geo.py td0_h)
cd $R/geo2
JTG=$(sbatch --parsable -p cpu1,cpu2 -J g2b_tg --time=48:00:00 --cpus-per-task=8 --mem=16G --dependency=afterok:$JX \
  --output=$L/%x_%j.out --export=ALL,GEO2ENV=d1 jobs/geo2.slurm feat_geo.py t2_g ';;;' feat_geo.py td0_g)
cd $R/d1_pair
JTF=$(sbatch --parsable -p cpu1,cpu2 -J g2b_tf --time=48:00:00 --cpus-per-task=8 --mem=32G --dependency=afterok:$JTG \
  --output=$L/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F jobs/g.slurm \
  ../geo2/feat_geo.py t2_f ::: ../geo2/feat_geo.py td0_f)
```
Pass checks:
- X: `results/X_d0d1_report.json` `complete` true, `stop` false, `n_d0` 5,255, `n_x_fail` 0 (dry run on all 5,260
  status-ok D0: 0 failures), `link_digests` prof / jobs / hf_templates = 14,521 links each. Outputs:
  `$GEO2B/x_d0/inputs/D0G2_<rid>/{orig,hf}/*.inp + check.json`, `results/X_d0_manifest.csv`, `$GEO2B/comb/*`.
  Rerun = same line (unchanged reactions are kept). A changed combined file is a STOP; `x 4 --force` moves
  `$GEO2B/comb` aside to `comb.prev_<time>` (explicit only). Run X before GEO1b rewrites `$GEO1B/d0/d0_folds.csv`:
  X checks it against this tree's `analysis/geo1b/results/D0_pairs_report.json` (later reruns need the merged report).
- S-1: `results/X_S1.json` `pass` true (all 10 values of 3 D0 reactions within 0.01 kcal/mol of labels_all.json,
  status ok), `results/X_S1_compare.csv`. Runs at `%pal 1` with `D1HF_SCRATCH=$GEO2B/x_d0` (D1-HF runner). Exit 2 = an
  ORCA calc unfinished (resubmit), 3 = reproduction failed (STOP: no H).
- T: `results/T_2.json`, `T_3.json`, `T_d0f.json`, `T_d0h.json` all `pass` true with the current code (work dirs
  `$GEO2B/t2`, `t3`, `td0`, `td0h`). `--rerun` forces a fresh work dir. After any code edit the T records are stale
  (exit 2 downstream): resubmit wave A's T lines.

## 3. Wave B — G, F, H pilot (after Z and T; G + g_report + F 17 tasks, pilot 1 = 18 queued tasks: at most 2 other tasks may be queued)
```bash
cd $R/geo2
JG=$(sbatch --parsable -p cpu1,cpu2 -J g2b_g --time=48:00:00 --cpus-per-task=16 --mem=32G --array=0-5 \
  --output=$L/%x_%A_%a.out --export=ALL,GEO2ENV=d1 jobs/geo2.slurm feat_geo.py g A 6)
cd $R/d1_pair
JGR=$(sbatch --parsable -p cpu1,cpu2 -J g2b_gr --time=48:00:00 --cpus-per-task=16 --mem=32G --dependency=afterany:$JG \
  --output=$L/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F jobs/g.slurm ../geo2/feat_geo.py g_report)
JFS=$(sbatch --parsable -p cpu1,cpu2 -J g2b_fs --time=48:00:00 --cpus-per-task=32 --mem=64G --array=0-3 \
  --dependency=afterok:$JGR --output=$L/%x_%A_%a.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F jobs/g.slurm \
  ../geo2/feat_geo.py f_slice @I 4)
JFM=$(sbatch --parsable -p cpu1,cpu2 -J g2b_fm --time=48:00:00 --cpus-per-task=4 --mem=32G --dependency=afterok:$JFS \
  --output=$L/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F jobs/g.slurm ../geo2/feat_geo.py f_merge)
JFA=$(sbatch --parsable -p cpu1,cpu2 -J g2b_fa --time=48:00:00 --cpus-per-task=32 --mem=64G --array=0-3 \
  --dependency=afterok:$JFM --output=$L/%x_%A_%a.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F jobs/g.slurm \
  ../geo2/feat_geo.py f_atoms @I 4)
JFR=$(sbatch --parsable -p cpu1,cpu2 -J g2b_fr --time=48:00:00 --cpus-per-task=32 --mem=64G --dependency=afterany:$JFA \
  --output=$L/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F jobs/g.slurm ../geo2/feat_geo.py f_rest)
cd $R/d1_hf          # needs results/X_S1.json pass as well
JHP=$(sbatch --parsable -p cpu2 -J g2b_hp --time=48:00:00 --cpus-per-task=64 --mem=256G --output=$L/%x_%j.out \
  jobs/hf.slurm ../geo2/feat_geo.py h_inputs 10 ::: ../geo2/feat_geo.py h_pilot)
```
- G: slices write `$GEO2B/geo_feat/<rxn_id>/` (.done | .fail_<reason>), `results/F_steps/g_<ii>of06.json`. A slice
  that exits 2 (open reactions) is resubmitted alone (`--array=<i>`), then g_report and the F lines again (a pending
  `afterok` job whose dependency failed never starts; it must be resubmitted). `g A 6 retry` (explicit) moves that
  slice's .fail_freq_* / .fail_error_* / stale folders aside and recomputes them. Bytes per slice (review estimate from
  the LADDER dft_final / D0-G2 folders: 0.57 MB per D1, 0.86 MB per D0 folder) about 1.6 GB of the 2 GB
  `GEO2_G_BUDGET_GB` default.
- g_report: `results/G_report.json` complete, stop false; per dataset atom order <= 0.5 %, Freq/error <= 1 %, B6 not
  computable <= 1 %. A B6-only STOP may be accepted by the user only (`g_report --accept=B6`).
- F: `$GEO2B/X182_d1_geo.parquet` (D1 + D0 rows), `$GEO2B/fu2_geo/` atoms + tokens, `results/fu2_geo/{Q,T}_report.json`,
  `results/F_steps/f_report.json` summary per dataset (F-1, B6, Q-a, T-a <= 1 %; Q-a limit = ceil(0.5 % x n)).
  Expect the F-1 rule to STOP for D1: the same feature code gave 1.14 % (D1PAIR G1), 1.42 % (LADDER F0) and 1.36 %
  (D1-G2, `analysis/d1_g2/results/F_DECISION.md`, accepted by the user), mostly `ext:B1:engine_gate1`. Report it; only
  a user decision continues with `p --accept=F-1` (GEO-7).
- pilot: `results/H_pilot.json` `pass` true: 25 D1 + 25 D0 stratified by n_atoms quartile, stratified projection
  <= 4,000 core-h, rel-energy gate on the D1 orig rows. A projection over the cap is a STOP (report the numbers).
  On cpu1 use `-p cpu1 --cpus-per-task=48 --mem=200G` (D1-HF precedent: %maxcore 3500 x workers).

## 4. Wave C — H batch (after the pilot passed; 11 tasks; 10 x 64 cores = the whole 640-CPU account cap)
```bash
cd $R/d1_hf
JHB=$(sbatch --parsable -p cpu2 -J g2b_hb --time=48:00:00 --cpus-per-task=64 --mem=256G --array=0-9 \
  --dependency=afterok:$JHP --output=$L/%x_%A_%a.out jobs/hf.slurm ../geo2/feat_geo.py h_batch @I 10)
JHR=$(sbatch --parsable -p cpu1,cpu2 -J g2b_hr --time=48:00:00 --cpus-per-task=4 --mem=32G --dependency=afterany:$JHB \
  --output=$L/%x_%j.out jobs/hf.slurm ../geo2/feat_geo.py h_build ::: ../geo2/feat_geo.py h_report)
```
- The array size must equal the `h_inputs` W (10). Runs under `$GEO2B/hf_geo/runs/g2/<job_id>/`; per batch
  `results/F_steps/h_batch_w<k>of10.json`. h_build: `results/hf_geo/labels_d1_hf_g2.parquet` (D1 + D0 rows; the name is
  the inputs.py interface), `results/hf_geo/H_status.csv`. h_report: `results/H_report.json` complete, stop false;
  per dataset run failures <= 1 %, input failures <= 1 %; rel gate (D1 rows with G2c choice `orig`) <= 0.5 %.
- Submit after G/F are done if other CPU work must run: while these 10 jobs run nothing else fits under the cap.
  A slice hit by the wall: resubmit `--array=<k>`, then the h_build line.

## 5. Wave D — P (after F and H complete and GEO1b final merged; 1 task)
```bash
cd $R/geo2
JP=$(sbatch --parsable -p cpu1,cpu2 -J g2b_p --time=48:00:00 --cpus-per-task=4 --mem=32G --output=$L/%x_%j.out \
  --export=ALL,GEO2ENV=reactot jobs/geo2.slurm feat_geo.py p ';;;' feat_geo.py p_report ';;;' feat_geo.py report)
```
- p: `$GEO2B/inputs/geo.parquet`, `results/P_build_geo.json` `all_pass` true, `stop` false, `outputs` = sha256 of the
  table (the LADDER gate_p_geo reads exactly these). Needs `inputs.RUNGS['geo'] == rung_cfg('geo')`, the LADDER
  Phase Q report (repo-relative rule F-13) and `Q_d0_report.json` all_pass with the current `Q_keys_d0.parquet`.
  `p --accept=F-1[,B6]` only on a user decision (GEO-7).
- p_report: `results/P_geo_report.json` (+ `P_geo_sets.json`, `$GEO2B/inputs/sets/P_geo_d0.csv`): D1 P_geo / P_23 / P_023
  over S_folds.csv (STOP |P_23| < 0.98 |P_g2|, kept as a pipeline check), D0 usable pairs per fold and edge kind,
  D0 pairs lost because an end is unusable, usable reactions per dataset and fold. Exit 2 while
  `D0_pairs_report.json` `g2_source` is not `g2c` (the preliminary D0 build), also when the final GEO1b d0_pairs.py run
  has already rewritten `$GEO1B/d0/D0_pairs.csv` but its report is not merged into this tree yet (review R-1); a
  sha256 difference against a final (`g2c`) report is exit 3.
- report: `results/F_report.json` + `results/F_G_status.csv`.

## 6. Tests (no ORCA / xtb; foreground `srun --jobid=<running job> --overlap -n1 -c1 timeout 500 ...`)
`tests_feat.py lint` (undefined names), `unit` (21 tests -> `results/U_feat_tests.json`), `wrap` (Phase X into
`$GEO2B/_selftest` + the GEO-15 HF wrappers on 3 D1 + 6 D0 reactions -> `results/U_feat_wrap.json`), `gorder` (D1 env;
G atom-order gate on D1 + D0 folders -> `results/U_feat_gorder.json`). `lint` / `unit` / `wrap` in conda reactot,
`gorder` in the D1 env.

## 7. Cost (estimates; core-h)
| step | estimate | basis |
|---|---|---|
| X | 0.1 | dry run on all D0: 45 s on 1 core |
| S-1 | 3–5 | 19-atom D0 label run 70 min serial; 2 neutral + 1 charged smallest |
| T | ~10 | t2/t3 3 rxn each; td0_h 20 D0 HF EDAs ~5–7 |
| G + g_report | ~30 | LADDER G0: 9,266 rxn 14 core-h -> 14,521 rxn |
| F | ~35–40 | LADDER F0 scale x 1.6 |
| H | ~2,800–3,500 | D1-G2 H 0.187 core-h/rxn for D1; D0 by the pilot; cap 4,000 (STOP above) |
| P | ~5 | |
| total | ~3,000–3,600 | H dominates; H wall ~5–7 h on 640 cores |
