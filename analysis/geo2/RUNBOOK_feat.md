# RUNBOOK — GEO2 role FEAT (spec 5 T-2 / T-3, 6 G / F / H / P, 7 P report)

Code: `analysis/geo2/feat_geo.py` (copy of `analysis/d1_g3/feat.py` with the spec 6-1 changes),
`analysis/geo2/g3c.py` (copy of `analysis/d1_g3/g3_common.py`), `analysis/geo2/jobs/geo2.slurm` (copy of
`analysis/d1_g3/jobs/g3.slurm`). Decisions: `analysis/geo2/results/DECISIONS_feat.md` (+ the runtime keys
`feat_geo/<key>` that every command appends to `results/DECISIONS.md`).

Nothing here was run when it was written (login node). Every command is idempotent: resubmitting the same line
skips finished work (G folders with a marker, F slice files / mergeroot / atoms npz, HF reactions with `.done`, the
plan) and recomputes the reports. Exit codes of every command: `0` done, `2` not ready / incomplete (resubmit later,
nothing is wrong), `3` STOP (spec 6-2 / 7) or hard error: read `analysis/geo2/results/F_steps/<step>.json`
(`fail_stage`, `fail`) and the report named below, report to the user and wait.

## 0. Before every submission (login node, one-shot commands only)

```bash
G=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1/analysis/geo2; L=/gpfs/tmp_cpu2/yeseo1ee/geo2/logs
squeue -u $USER -h -r | wc -l            # MaxSubmit 20 counts array elements: N + (tasks of the wave) <= 20
sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"     # cpu1 / cpu2 are interchangeable; '-p cpu1,cpu2' lets SLURM take the free one
ls -d $L                                  # the logs folder exists (spec 1 'folder' job); every job gives --output under it
```

- All jobs: `--time=48:00:00`. `g.slurm` and `hf.slurm` default their log to read-only scratch folders, so every line
  passes `--output=$L/...`.
- Every dependency uses `${X:?}` so that a refused `sbatch` stops the following lines instead of passing an empty id.
  A dependency on a job that already ended OK and was purged from SLURM makes `sbatch` fail: drop that
  `--dependency` term (check with `sacct -j <id> -X -o JobID,State,ExitCode`).
- **T currency (F-10 / F-16):** `results/T_2.json`, `T_3.json` record the sha256 of every file in `CODE_REL`
  (`feat_geo.py`, `g3c.py`, `jobs/geo2.slurm`, the D1LADDER / D1-G2 / D1PAIR / D1-HF code they call). Any later edit
  of those files makes every G / F / H / P command exit 2 until T-2 and T-3 are rerun. Run T last after code review.

## 1. Environments (job file, how to submit, what the command checks)

| step | job file (submit from) | env | checked by |
|---|---|---|---|
| t2_g, g | `jobs/geo2.slurm` (`analysis/geo2`) | `--export=ALL,GEO2ENV=d1` (D1 run env: venv python 3.9, ORCA 6.1.1, xtb 6.7.1) | `env_d1`: `$D1_PY`, `ORCA_BIN`, `OMP_NUM_THREADS=1`, `GEO2_SCRATCH`, `GEO2ENV` |
| t2_f, g_report, f_slice, f_atoms, f_rest (f_merge, f_gate, f_tokens, f_report) | `jobs/g.slurm` (`analysis/d1_pair`) | `--export=ALL,PYENV=venv5,D1PAIR_PHASE=F` | `test_t.env_f` |
| t3, h_inputs, h_pilot, h_batch, h_build, h_report | `jobs/hf.slurm` (`analysis/d1_hf`) | default (conda reactot + ORCA, `D1HF_SCRATCH` = d1_hf scratch, no `D1HF_*` overrides) | `test_t.env_hf` (cwd must be `analysis/d1_hf`) |
| p, p_report, report | `jobs/geo2.slurm` (`analysis/geo2`) | `--export=ALL,GEO2ENV=reactot` | `env_reactot` |

`g.slurm` replaces the argument `@I` by `$SLURM_ARRAY_TASK_ID`; with `geo2.slurm` / `hf.slurm` the script itself reads
`A` / `@I` from `$SLURM_ARRAY_TASK_ID`.

## 2. Preconditions owned by other roles

| needed by | file | made by |
|---|---|---|
| g, f_*, h_*, p, p_report | `analysis/geo2/results/Z_report.json` with `stop: false`, `Z_status.csv` (Phase Z, `build_geo.py build`) | Z job (spec 4). Missing = exit 2, Z STOP = exit 3 |
| p | D1LADDER Phase Q ready (`analysis/d1_ladder/results/Q_report.json`), FU2 S_splits (unchanged LADDER inputs) | existing |
| p (only if LADA fixed it) | `inputs.RUNGS['geo']` in `analysis/d1_ladder/inputs.py` must equal `feat_geo.rung_cfg('geo')` by string values | LADA (spec 8-1) |
| p_report | `$LAD/inputs/{dft,g2}.parquet`, `analysis/d1_ladder/results/P_report.json`, `P_build_{dft,g2}.json`, `$LAD/inputs/sets/P_{dft,g2}.csv` (checked: sha256 consistent today) | existing |

## 3. Wave A — Phase T (3 tasks; independent of Y / Z, may run while Y trains)

```bash
cd $G
JT2G=$(sbatch --parsable -p cpu1,cpu2 -J geo2_t2g --time=48:00:00 --cpus-per-task=3 --mem=8G \
     --output=$L/%x_%j.out --export=ALL,GEO2ENV=d1 jobs/geo2.slurm feat_geo.py t2_g); echo "JT2G=$JT2G"
JT2F=$(cd ../d1_pair && sbatch --parsable -p cpu1,cpu2 -J geo2_t2f --time=48:00:00 --cpus-per-task=8 --mem=32G \
     --output=$L/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F --dependency=afterok:${JT2G:?} \
     --kill-on-invalid-dep=yes jobs/g.slurm ../geo2/feat_geo.py t2_f); echo "JT2F=$JT2F"
JT3=$(cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J geo2_t3 --time=48:00:00 --cpus-per-task=3 --mem=16G \
     --output=$L/%x_%j.out jobs/hf.slurm ../geo2/feat_geo.py t3); echo "JT3=$JT3"
```

| step | inputs | output / done marker | STOP looks like |
|---|---|---|---|
| t2_g | `$LAD/dft_final/{1000003,1000019,1000614}` (read only) | `$GEO2/t2/root/<rxn>/.done` (3), `F_steps/t2_g.json` rc 0; `T_2.json` says `running` until t2_f | exit 3, `T_2.json` `verdict: fail` (`fail_stage t2_g:...`) |
| t2_f | `$GEO2/t2` made by t2_g with the current code | `results/T_2.json` `verdict: pass` (182 columns <= 1e-8 vs `X182_d1_dft`, 3/3 token sha256 = `F0_tokens_manifest.csv`) | exit 3, `verdict: fail` (see `x182_compare`, `tokens`) — spec 14 T STOP |
| t3 | `$LAD/dft_final`, `d1_hf/inputs/<T>/hf` (read only) | `results/T_3.json` `verdict: pass` (<= 1e-6 kcal/mol vs `labels_d1_hf.parquet` geom=dft) | exit 3, `verdict: fail` (see `compare`) |

Rerun from scratch (only after a code change, done automatically by F-16): `feat_geo.py t2_g --rerun` / `t3 --rerun`.

## 4. Wave B1 — H chain (7 tasks, W = 5; the long pole, ~1,600 core-h) — after Z and T passed

Start H as soon as Z (`build_geo.py`) exited 0 and T passed; H does not wait for G / F. `JZ` = the Z job id (drop
that term if Z already finished).

```bash
W=5
JHI=$(cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J geo2_hip --time=48:00:00 --cpus-per-task=48 --mem=180G \
     --output=$L/%x_%j.out --dependency=afterok:${JZ:?}:${JT2F:?}:${JT3:?} --kill-on-invalid-dep=yes \
     jobs/hf.slurm ../geo2/feat_geo.py h_inputs ${W:?} ::: ../geo2/feat_geo.py h_pilot); echo "JHI=$JHI"
JHB=$(cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J geo2_hb --time=48:00:00 --cpus-per-task=48 --mem=180G \
     --array=0-$((W-1)) --output=$L/%x_%A_%a.out --dependency=afterok:${JHI:?} --kill-on-invalid-dep=yes \
     jobs/hf.slurm ../geo2/feat_geo.py h_batch @I ${W:?}); echo "JHB=$JHB"
JHR=$(cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J geo2_hr --time=48:00:00 --cpus-per-task=4 --mem=32G \
     --output=$L/%x_%j.out --dependency=afterok:${JHI:?},afterany:${JHB:?} --kill-on-invalid-dep=yes \
     jobs/hf.slurm ../geo2/feat_geo.py h_build ::: ../geo2/feat_geo.py h_report); echo "JHR=$JHR"
```

| step | inputs | output / done marker | STOP looks like |
|---|---|---|---|
| h_inputs W | geo base (`$GEO2/geo_final/<rxn>/.done`), `d1_hf/inputs/<T>/hf/*.inp` templates | `$GEO2/hf_geo/inputs/<T>/g2/*.inp`, `results/hf_geo/g2_inputs.csv`, `results/hf_geo/H_batches/{plan.json, H_bNN.csv, H_pilot.csv}` (plan.json written last) | exit 3: geominputs status != ok for > 1 % of the base, or a plan for another W / changed inputs exists (move `results/hf_geo/H_batches` aside by hand) |
| h_pilot | plan.json, `H_pilot.csv` (first 50 inputs-ok reactions by job_id) | `results/H_pilot.json` `pass: true`, `projection_core_h` (the 50 runs stay in `$GEO2/hf_geo/runs/g2` and are skipped later) | exit 3: `stop_reasons` = projection (pilot mean core-h x inputs-ok reactions, spec 6-1 rule only) > 2,500 core-h; the D1-G2-calibrated value is information (`projection.d1g2_calibrated`; above 2,500 alone = `warnings` + a `GEO-4 pilot warning` line in `results/DECISIONS.md`, no STOP) or rel-energy mismatch vs D1-G2 > 0.5 % of the pilot (= any one); exit 2: a pilot reaction not finished (resubmit the JHI line without h_inputs work: it is skipped) |
| h_batch k W | plan.json, a passing `H_pilot.json` of this plan | `$GEO2/hf_geo/runs/g2/<T>/.done` (or `.fail`), `results/hf_geo/M_checks/H_bNN.json` for its batches, `F_steps/h_batch_w<k>of5.json` rc 0 | exit 3: cumulative run-failure rate (fail + gate statuses) over this plan's batch checks > 1 % (`h_gate`); exit 2: run.py left reactions (killed / 45 h) — resubmit that worker |
| h_build | runs | `results/hf_geo/labels_d1_hf_g2.parquet`, `results/hf_geo/H_status.csv`, `$GEO2/hf_geo/labels_d1_hf_g2.json` | exit 3 only on a child error |
| h_report | h_build record unchanged | `results/H_report.json` `complete: true, stop: false` | exit 3: run failures > 1 %, input check > 1 %, rel-energy gate (`summary.rel_gate`) > 0.5 % vs `analysis/d1_g2/results/labels_d1_hf_g2.parquet`, stale runs; exit 2: open / missing checks |

Resubmit one dead / exit-2 worker k (the same plan, W = 5), then JHR again:
```bash
JHBk=$(cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J geo2_hb$k --time=48:00:00 --cpus-per-task=48 --mem=180G \
     --output=$L/%x_%j.out jobs/hf.slurm ../geo2/feat_geo.py h_batch $k 5); echo "JHBk=$JHBk"
JHR=$(cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J geo2_hr --time=48:00:00 --cpus-per-task=4 --mem=32G \
     --output=$L/%x_%j.out --dependency=afterany:${JHBk:?} jobs/hf.slurm ../geo2/feat_geo.py h_build ::: \
     ../geo2/feat_geo.py h_report); echo "JHR=$JHR"
```
`.fail` reactions are not retried (D1-HF rule: one SlowConv retry inside run.py only).

## 5. Wave B2 — G and F chain (10 tasks) — after Z and T passed

```bash
cd $G
JG=$(sbatch --parsable -p cpu1,cpu2 -J geo2_g --time=48:00:00 --cpus-per-task=16 --mem=32G --array=0-3 \
     --output=$L/%x_%A_%a.out --export=ALL,GEO2ENV=d1 --dependency=afterok:${JZ:?}:${JT2F:?}:${JT3:?} \
     --kill-on-invalid-dep=yes jobs/geo2.slurm feat_geo.py g A 4); echo "JG=$JG"
JGR=$(cd ../d1_pair && sbatch --parsable -p cpu1,cpu2 -J geo2_gr --time=48:00:00 --cpus-per-task=16 --mem=32G \
     --output=$L/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F --dependency=afterok:${JG:?} \
     --kill-on-invalid-dep=yes jobs/g.slurm ../geo2/feat_geo.py g_report); echo "JGR=$JGR"
JFS=$(cd ../d1_pair && sbatch --parsable -p cpu1,cpu2 -J geo2_fs --time=48:00:00 --cpus-per-task=32 --mem=64G \
     --array=0-3 --output=$L/%x_%A_%a.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F --dependency=afterok:${JGR:?} \
     --kill-on-invalid-dep=yes jobs/g.slurm ../geo2/feat_geo.py f_slice @I 4); echo "JFS=$JFS"
JFR=$(cd ../d1_pair && sbatch --parsable -p cpu1,cpu2 -J geo2_fr --time=48:00:00 --cpus-per-task=48 --mem=96G \
     --output=$L/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F --dependency=afterok:${JFS:?} \
     --kill-on-invalid-dep=yes jobs/g.slurm ../geo2/feat_geo.py f_rest); echo "JFR=$JFR"
```

| step | inputs | output / done marker | STOP looks like |
|---|---|---|---|
| g i 4 | geo base, D1 job `ts.xyz` / `ref_*.xyz`, G2 inputs `input_meta_d1_g2.csv` | `$GEO2/geo_feat/<rxn>/{.done \| .fail_<reason>}` with `ts.xyz, rel1.xyz, rel2.xyz` (byte copies of geo_final), `work/ts.hess`, `result.json`; `F_steps/g_0<i>of04.json` rc 0 | exit 3: both canary reactions failed (environment), >= 20 and > 25 % Freq / error failures in the slice (abort), byte budget `GEO2_G_BUDGET_GB` (2 GB per slice; ~1.2 GB expected), stale copies; exit 2: reactions left (44 h) — resubmit that slice (`--array=<i>`) |
| g_report | all G slices complete (no open, no stale folder) | `results/G_report.json` `complete: true, stop: false` (counts, rates, `b6_not_computable`), `$GEO2/G_geo/{gate_ids.csv, gate.json}` (ts_fragments + `ext_features.block_b6` on every `.done` folder, child `_gcheck`); `F_steps/g_report.json` rc 0 | exit 3 (`G_report.json` `stop_reasons`): atom-order failures (order gate + ts_fragments) > 0.5 % of the base, Freq / error > 1 % of the base, B6 not computable > 1 % of the `.done` folders (spec 6-2 G), stale copies, `_gcheck` not on the geo root env; exit 2: G slices not complete |
| f_slice i 4 | all G slices complete; G gate over the whole base: atom-order failures <= 0.5 %, Freq / error <= 1 % (spec 6-2 G); a current passing `results/G_report.json` (GEO-11) | `$GEO2/F_geo/{root_fp.json, ids_all.csv}` (under `$GEO2/locks/F_geo.lock`), `$GEO2/F_geo/out/xtb/f_slices/slice<i>of04_{g1.parquet, ext.parquet, root.json, versions.json}` | exit 3 `fail_stage G`: the G gate or the G report STOP (spec 6-2 G); exit 2: G not complete, or `G_report.json` missing / not for the current G folders (run g_report) |
| f_rest | the 4 slices | `$GEO2/X182_d1_geo.parquet`, `$GEO2/F_geo/mergeroot_report.json`, `$GEO2/fu2_geo/{atoms,tokens}`, `results/fu2_geo/{Q_report.json, Q_manifest.csv, T_report.json, T_manifest.csv}`; `F_steps/f_report.json` rc 0 | exit 3 (`F_steps/f_report.json` `summary.stop_reasons`): F-1 > 1 % (every G `.done` reaction without a finite 182-feature row, B6-only rows included = g0f0 F0; `gate.F1_without_B6_only` is information only), B6 > 1 %, B6 status of G (`G_report.json`) != F `status_B6`, Q-a > 1 %, T-a > 1 %, FU2 Q / T gate STOP, mergeroot stop, token layout != D1PAIR g1, inconsistent counts (spec 6-2 F) |

If f_rest died: the same JFR line without `--dependency` (merge / atoms npz cached, gate / tokens rerun). If a G retry
is needed (explicit, after reading the `.fail_freq_*` / `.fail_error_*` result.json files):
`sbatch ... --array=<i> jobs/geo2.slurm feat_geo.py g A 4 retry` — then the JGR line again (the old `G_report.json` is
no longer current), and the next f_slice moves the old F outputs aside (F-15): all four f_slice tasks and f_rest must
run again.

**After a user decision on the G B6 rate STOP only** (GEO-7 / GEO-11): the JGR line with `feat_geo.py g_report
--accept=B6` (no `--dependency`), then JFS / JFR as above. F counts the same B6 rate again (G and F must agree), so
f_rest will end with exit 3 on that rule (f_report has no `--accept`); after checking that B6 is its only STOP, run the
§6 accept line with `p --accept=B6` / `report --accept=B6` (no `--dependency`). Order / Freq / stale STOPs of
g_report cannot be accepted.

## 6. Wave C — P, pair report, F report (1 task) — after f_rest and h_report exited 0

```bash
cd $G
JP=$(sbatch --parsable -p cpu1,cpu2 -J geo2_p --time=48:00:00 --cpus-per-task=4 --mem=32G --output=$L/%x_%j.out \
     --export=ALL,GEO2ENV=reactot --dependency=afterok:${JFR:?}:${JHR:?} --kill-on-invalid-dep=yes \
     jobs/geo2.slurm feat_geo.py p ';;;' feat_geo.py p_report ';;;' feat_geo.py report); echo "JP=$JP"
```

| step | inputs | output / done marker | STOP looks like |
|---|---|---|---|
| p | G / G report / F / H gates complete without STOP, D1LADDER Phase Q ready (repo-relative fingerprint, F-13) | `$GEO2/inputs/geo.parquet` (9,266 rows, `usable` flags), `results/P_build_geo.json` `all_pass: true` (`rung_cfg` = the `inputs.RUNGS['geo']` entry) | exit 3: G / F / H STOP, P-a gate or re-assembly checks fail (spec 6-2 P), `inputs.RUNGS['geo']` fixed by LADA differs from `rung_cfg('geo')` (GEO-8); `P_build_geo.json` `stop_reason` |
| p_report | a passing current `P_build_geo.json`; LADDER P_report / tables / set files consistent | `results/P_geo_sets.json` (`P_geo, P_23, P_023, P_g2, P_dft` sorted pair_id lists), `results/P_geo_report.json` `stop: false`, `$GEO2/inputs/sets/P_{geo,23,023}.csv` | exit 3: `|P_23| < 0.98 |P_g2|` (all pairs or P_main reference pairs; spec 7 STOP), LADDER inputs changed (recomputed P_dft / P_g2 != LADDER set files) |
| report | everything above | `results/F_report.json` `complete: true, stop: false`, `results/F_G_status.csv` | exit 3 with `stop_reasons`; exit 2 with `incomplete` |

`';;;'` (quoted) = run the next command regardless, so `report` always rewrites `F_report.json`; the job exit code is
the first non-zero one.

**After a user decision on an F rate STOP only** (LADDER `--accept-f0-stop` / D1-G2 `F_DECISION.md` precedent, GEO-7):
```bash
JP=$(sbatch --parsable -p cpu1,cpu2 -J geo2_p --time=48:00:00 --cpus-per-task=4 --mem=32G --output=$L/%x_%j.out \
     --export=ALL,GEO2ENV=reactot jobs/geo2.slurm feat_geo.py p --accept=F-1 ';;;' feat_geo.py p_report ';;;' \
     feat_geo.py report --accept=F-1); echo "JP=$JP"
```
(`--accept=F-1,B6` if B6 was accepted too). Only the F-1 / B6 rate rules can be accepted; every other STOP stays a
STOP. The acceptance is written to `P_build_geo.json` / `F_report.json` (`accepted_stops`).

## 7. Task budget and order

| wave | tasks | can start |
|---|---|---|
| A (T) | 3 | now (code final) |
| B1 (H) | 1 + 5 + 1 = 7 | Z exit 0 and T pass |
| B2 (G, G report, F) | 4 + 1 + 4 + 1 = 10 | Z exit 0 and T pass |
| C (P) | 1 | f_rest and h_report exit 0 |

All of B1 + B2 + C at once = 18 tasks: submit B1 first (longest), then B2 and C when `squeue -u $USER -h -r | wc -l`
leaves room (MaxSubmit 20; MaxJobs 10 running: B1 workers 5 + G 4 fit). Expected walls: G ~15-30 min per slice
(LADDER G0: 9,266 reactions in 2 x 16 cores x 27 min); g_report ~1-5 min (LADDER g0_report, the same per-folder check on
9,266 folders with 16 cpus: 43 s, `analysis/d1_ladder/results/G0_steps/g0_report.json`); f_slice ~10 min; f_rest ~10-20 min; h_inputs ~15 min +
pilot ~1-2 h; each H worker ~7 h (48 serial ORCA, ~1,700 reactions x ~0.19 core-h); h_build / h_report ~30 min;
p ~1 h (P-a re-parses every HF run); p_report / report a few minutes.
