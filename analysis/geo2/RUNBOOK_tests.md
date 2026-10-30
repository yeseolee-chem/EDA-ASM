# RUNBOOK — role LADB: GEO2 8-2 tests (`C_tests_geo.json`) and the Phase E model part (`evaluate.py e3`)

Spec: `docs/specs/GEO2.md` 8-2 (tests), 9-2 / 10 (E3 part) — the authority. Code of this role:
`analysis/d1_ladder/tests/run_tests.py` (mode `geo`, decisions G-1 .. G-17 in its docstring) and
`analysis/d1_ladder/evaluate.py` (commands `e3`, `e3-selftest`, decisions E3-1 .. E3-8 + E3-2b = `DECISIONS3`). All judgment
calls: `analysis/geo2/results/DECISIONS_tests.md`. **Nothing here has been run yet** (written on the login node).

## 0. Rules for every step
- Login node: only `sbatch`, `squeue`, `sinfo`, `scancel`, `git`, reading files. No python, no loops, no sleep.
- Before every submission: `squeue -u $USER -h -r | wc -l` = N (array elements count): N + (tasks of the step) <= 20,
  at most 10 running. Pick partitions with idle capacity:
  `sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`, `sinfo -p gpu1,gpu2,gpu3,gpu5,gpu6 -o "%.9P %.6t %.10G %C"`.
- Every job: `--time=48:00:00`, log under `/gpfs/tmp_cpu2/yeseo1ee/geo2/logs` (`$GL`): the wrappers' default log
  folders are LADDER / D1PAIR scratch, so `--output` is always given.
- Do not export `GEO2_SCRATCH`, `D1LADDER_SCRATCH`, `D1PAIR_SCRATCH` with other values (paths are compared as strings).
- Exit codes: 0 = done (all pass on the final inputs), 2 = not ready / waiting / a part not run (resubmit later; no
  failure), 3 = STOP (spec 14: read the json key named below, report, wait).
- Idempotent: each part rewrites its part file atomically, the merge recomputes from the part files, the scratch
  `$LAD/tests_geo/<sub-run>` is removed before each sub-run. Resubmitting the same line is always safe.
- **Never run in GEO2** (they rewrite LADDER result files = the PREREG_GEO2 snapshot STOP of spec 14):
  `tests/run_tests.py gpu | cpu | merge` (without `geo`; they write `results/C_tests*.json`) and
  `evaluate.py e1 | e2` (they write `results/E1_*`, `E2_*`, `results/figures`). The run_tests.py LADDER modes now
  refuse in code (G-17: `STOP: run_tests.py <cmd> refused ...`, exit 3, nothing written) while the code differs from
  `results/C_tests.json` `code_sha256`. In this worktree `evaluate.py e1 | e2` stop with exit 2 at their first
  currency check (`inputs.p_report(rung)`: build fingerprints stale, DECISIONS_tests T-35), before any write. Do not
  rely on either: still never submit them.

```bash
REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-geo1
GL=/gpfs/tmp_cpu2/yeseo1ee/geo2/logs
```

| wrapper | env | submit from | python |
|---|---|---|---|
| `analysis/d1_ladder/jobs/gpu.slurm` | env_d1fu2 (torch; GPU node) | `analysis/d1_ladder` | `tests/run_tests.py geo gpu ...` |
| `analysis/d1_pair/jobs/py.slurm` + `--export=ALL,PYENV=espley` | env_espley (xgboost, sklearn 1.3.1, scipy) | `analysis/d1_pair` | `../d1_ladder/tests/run_tests.py geo cpu ...`, `../d1_ladder/evaluate.py e3 ...` |
| `analysis/d1_ladder/jobs/lad.slurm` + `--export=ALL,LADENV=reactot` | conda reactot | `analysis/d1_ladder` | `tests/run_tests.py geo merge ...` |

## 1. Order and dependencies

```
now (needs only the LADDER results, $GEO2/regress/list_stage{1,2}.before.txt, $GEO2/pristine_head):
  T-R  run_tests.py geo {gpu,cpu} --regress-only + merge   -> results/C_tests_geo_regress.json   (early R-1 .. R-3)
  E-T  evaluate.py e3-selftest                               -> results/e3_selftest/E3_selftest.json
after FEAT P + p_report (P_build_geo.json all_pass, P_geo_report.json stop false; RUNBOOK_feat):
  C    run_tests.py geo {gpu,cpu} + merge                    -> results/C_tests_geo.json all_pass
         └─ (fresh `geo merge` right before) LADA R-1 prereg write -> R-2 commit -> R-3 check -> M-1 .. M-4 (RUNBOOK_lad)
after LADA M-4 collect (results/L_predictions_geo.parquet, L_report_geo.json; M-2 L_xgb_report_geo.json):
  E3   evaluate.py e3                                        -> analysis/geo2/results/E3_*.csv / .json
```
(`results/` above = `analysis/geo2/results/`.) T-R and E-T are independent of each other and of FEAT; run them now.

## 2. Steps

### T-R  regressions only (spec 8-2 회귀 1–3), before the geo inputs exist
```bash
cd $REPO/analysis/d1_ladder
JG=$(sbatch --parsable -p gpu1,gpu2,gpu3,gpu5,gpu6 --gres=gpu:1 -J geo2_creg_g --time=48:00:00 --cpus-per-task=4 \
  --mem=32G --output=$GL/%x_%j.out jobs/gpu.slurm tests/run_tests.py geo gpu --regress-only); echo "JG=$JG"
cd $REPO/analysis/d1_pair
JC=$(sbatch --parsable -p cpu1,cpu2 -J geo2_creg_c --time=48:00:00 --cpus-per-task=32 --mem=64G \
  --output=$GL/%x_%j.out --export=ALL,PYENV=espley \
  jobs/py.slurm ../d1_ladder/tests/run_tests.py geo cpu --regress-only); echo "JC=$JC"
cd $REPO/analysis/d1_ladder
sbatch -p cpu1,cpu2 -J geo2_creg_m --time=48:00:00 --cpus-per-task=1 --mem=4G --output=$GL/%x_%j.out \
  --dependency=afterany:${JG:?}:${JC:?} --export=ALL,LADENV=reactot \
  jobs/lad.slurm tests/run_tests.py geo merge --regress-only
```
- Tasks: 3 (N + 3 <= 20). Expected (not measured): gpu part < 1 h (two `train_atoms.py list` processes, FU2 Data
  load for L-6, L-10); cpu part ~1–2 h on 32 CPUs (two full-grid XGB units of 490 fits each: L-6 g1·S f0 and R-3
  g2·S f0).
- Inputs (read only): train_atoms.py / train_xgb.py / inputs.py / lad_common.py / evaluate.py (this worktree),
  `$GEO2/regress/list_stage{1,2}.before.txt`, `$GEO2/pristine_head/analysis/d1_ladder/inputs.py`, LADDER
  `$LAD/inputs/{dft,g1,g2}.parquet` + `results/P_build_*.json`, `results/P_report.json`, `results/V_report.json`,
  `$LAD/F_pairs_cr2.csv`, D1PAIR `F_pairs.csv` / `S_folds.csv`, FU2 units `T1__*__AG__f0`, rev 9
  `T1__C__XGB__<t>__f0`, `analysis/d1_hf/results/A_summary.csv`, `results/L_predictions.parquet` + `L_report.json`,
  `$LAD/units/L__g2__S__XGB__f0.json`.
- Outputs: `analysis/geo2/results/C_tests_geo_regress_{gpu,cpu}.json`, merged `C_tests_geo_regress.json`;
  `analysis/d1_ladder/results/C_l6_xgb_geo.json` (R-2 XGB L-6, `train_xgb.py l6 --stage 3`); scratch
  `$LAD/tests_geo/r3_xgb`, `$LAD/smoke_xgb_l6_geo`.
- Done: the merge job exits 0 and `C_tests_geo_regress.json` has `all_pass: true`
  (`regressions: {R-1: pass, R-2: pass, R-3: pass}`).
- STOP (exit 3, `failed` non-empty; spec 14 R·M3 "8-2 검사 실패"): read `checks.<R-k>.parts.<gpu|cpu>`:
  - R-1: `stages.stage1|stage2.first_diff` (unit list changed) or `stage3.first_diff`.
  - R-2: `L6` / `L10` = the LADDER check bodies (criteria unchanged); `build_currency.rungs.<rung>.disallowed` = a
    LADDER input other than inputs.py changed since its Phase P build (G-12); `build_currency.p_report_inputs.
    disallowed` = the `results/P_report.json` `inputs` entry of `S_folds.csv` (found by repo-relative path) does not
    carry the current file's sha256 (G-12b; the L-10 `inputs_pairs_frame` sub-check then fails with "(G-12b)").
  - R-3: `targets.<t>` (`same_pairs`, `max_abs_yhat` > 1e-6).
  - An XGB check that stops with "GroupKFold(4) fold 0 differs from the split file" is a node-type tie-order effect
    (train_xgb docstring): resubmit the cpu line with `--nodelist=n112` (or `n115`, the rev 9 node types), change
    nothing else.
- Exit 2 (`not_final` / `waiting_reasons`): a part has not run, an env / GPU error (`errors`), or a LADDER input is
  missing — resubmit that part, then the merge line alone (no `--dependency`).
- This file is **not** the training gate (`training_gate: false`); step C repeats R-1 .. R-3.

### E-T  E3 selftest (geo := a copy of g2), now
```bash
cd $REPO/analysis/d1_pair
sbatch -p cpu1,cpu2 -J geo2_e3t --time=48:00:00 --cpus-per-task=4 --mem=32G --output=$GL/%x_%j.out \
  --export=ALL,PYENV=espley jobs/py.slurm ../d1_ladder/evaluate.py e3-selftest
```
- Tasks: 1. Expected (not measured): about the E2 run time of two rungs (bootstrap CIs + Hammett bend calls).
- Inputs: the LADDER E2 inputs (`results/L_predictions.parquet`, `L_report.json`, `L_val.csv`, `$LAD/units/*.json`,
  `P_report.json` + sets / fold copies, labels, ADD3), `results/E2_report.json`, `E2_metrics.csv`, `E2_goal.csv`,
  `E2_bend_rates.json`, FU2 Phase E code + `E_tests.json`, `analysis/geo2/results/selftest/E_report.json`
  (eval_geo2 selftest, job 1053810).
- Outputs: `analysis/geo2/results/e3_selftest/{E3_metrics.csv, E3_bends.csv, E3_goal.csv, E3_compare.csv,
  E3_bend_rates.json, E3_selftest.json}` (`E3_selftest.json` written last = done marker).
- Done: exit 0 and `E3_selftest.json` `ok: true`.
- A STOP before `E3_selftest.json` is written (log line `STOP: results/P_report.json inputs: no entry for ...
  (E3-2b ...)`) = the LADDER `S_folds.csv` changed since Phase P (its sha256 differs from the repo-relative
  P_report entry): report, do not rebuild.
- STOP (exit 3 = spec 14 E "selftest 실패"): `checks` lists the false ones; `check_detail` and `e2_crosscheck.problems`
  say which rows differ. `e2_crosscheck` false = E3 does not reproduce `E2_metrics.csv` / `E2_goal.csv` /
  `E2_bend_rates.json` for g2 (a code or input change since E2). `eval_geo2_selftest_equal` false = REP8 choice or
  REP8 oi / cpcm para MAE differs from `eval_geo2.py selftest`.
- Exit 2: an input missing / not committed (`fu2_eval`, ADD3) — fix the input, resubmit.

### C  full tests, rung geo (spec 8-2; the stage-3 training gate)
Start only after FEAT `p` and `p_report` passed (`analysis/geo2/results/P_build_geo.json` all_pass and
`P_geo_report.json` stop false; RUNBOOK_feat), so the geo table, tokens, X182 and HF parquet are final.
```bash
cd $REPO/analysis/d1_ladder
JG=$(sbatch --parsable -p gpu1,gpu2,gpu3,gpu5,gpu6 --gres=gpu:1 -J geo2_ctest_g --time=48:00:00 --cpus-per-task=4 \
  --mem=32G --output=$GL/%x_%j.out jobs/gpu.slurm tests/run_tests.py geo gpu); echo "JG=$JG"
cd $REPO/analysis/d1_pair
JC=$(sbatch --parsable -p cpu1,cpu2 -J geo2_ctest_c --time=48:00:00 --cpus-per-task=32 --mem=64G \
  --output=$GL/%x_%j.out --export=ALL,PYENV=espley jobs/py.slurm ../d1_ladder/tests/run_tests.py geo cpu); echo "JC=$JC"
cd $REPO/analysis/d1_ladder
JCM=$(sbatch --parsable -p cpu1,cpu2 -J geo2_ctest_m --time=48:00:00 --cpus-per-task=1 --mem=4G \
  --output=$GL/%x_%j.out --dependency=afterany:${JG:?}:${JC:?} --export=ALL,LADENV=reactot \
  jobs/lad.slurm tests/run_tests.py geo merge); echo "JCM=$JCM"
```
- Tasks: 3. Expected (not measured): gpu part a few hours (L-1 1-epoch fits, spy run_unit for 3 models x 2 arms,
  `train_atoms.py smoke --stage 3` = 6 configs x 2 cells x 2 epochs + 1 final seed, R-2); cpu part ~2–4 h (L-4 /
  spies / smoke XGB on geo + the two full-grid units of R-2 / R-3).
- Inputs (exit 2 / `waiting` while missing): `$GEO2/inputs/geo.parquet` + `analysis/geo2/results/P_build_geo.json`
  (all_pass, outputs sha256 of the table), `analysis/geo2/results/fu2_geo/{T_report.json, T_manifest.csv,
  Q_report.json, S_splits.json}` + `$GEO2/fu2_geo/tokens`, `$GEO2/X182_d1_geo.parquet`,
  `analysis/geo2/results/hf_geo/labels_d1_hf_g2.parquet`; plus every T-R input.
- Outputs: `analysis/geo2/results/C_tests_geo_{gpu,cpu}.json`, merged `C_tests_geo.json`;
  `analysis/d1_ladder/results/C_smoke_atoms_geo.json`, `C_smoke_xgb_geo.json`, `C_l6_xgb_geo.json`; scratch
  `$LAD/tests_geo/*`, `$LAD/smoke_geo`, `$LAD/smoke_xgb_geo`, `$LAD/smoke_xgb_l6_geo`. Nothing under `$LAD/units`
  (C-25 guard on the smokes).
- Done: the merge job exits 0 and `C_tests_geo.json` has `all_pass: true` (`training_gate: true`).
- Also read `C_tests_geo.json` `gpu_estimate_geo.geo_gpu_hours`: > 80 = spec 8-4 / 14 STOP (judged again by
  `prereg_geo2.py`, G-9).
- STOP (exit 3, `failed`): per check `checks.<name>.parts.<gpu|cpu>`; L-4: `rungs.geo.foreign` / `unknown` /
  `n_dft_or_g1_paths` / `n_g2_paths` / `g1_folder_paths`; L-5: `rungs.geo.*_vs_table` or `table_provenance`; smoke:
  `configs.<unit>.ok`, `units_written_by_smoke`; R-*: as T-R. A geo rung `status: fail` with "inputs.RUNGS['geo']
  differs" = the FEAT / LADA geo paths disagree with this file (G-7): report.
- Exit 2: geo inputs not ready (`waiting_reasons` starts with `geo:`), a part not run / env error; resubmit the part,
  then the merge line alone.
- **Before LADA R-1 (prereg write):** resubmit only the merge line (no dependency). The merge re-checks that the
  code / labels / tables recorded by both parts are still current (`current`); if any of the six files in
  `code_sha256` changed after the parts ran, `all_pass` turns false and the parts must be rerun. The PREREG must
  freeze the code that these tests passed.

### E3  model-level evaluation of rung geo (spec 9-2), after LADA M-4
```bash
cd $REPO/analysis/d1_pair
sbatch -p cpu1,cpu2 -J geo2_e3 --time=48:00:00 --cpus-per-task=4 --mem=32G --output=$GL/%x_%j.out \
  --export=ALL,PYENV=espley --dependency=afterok:<M-4 collect job id> --kill-on-invalid-dep=yes \
  jobs/py.slurm ../geo2/eval_geo2.py run ::: ../d1_ladder/evaluate.py e3
```
- (`eval_geo2.py run` first is optional — another role's command — but lets E3 record the eval_geo2 cross-check;
  `evaluate.py e3` alone is the same line without `../geo2/eval_geo2.py run ::: `. Without a pending collect job,
  drop `--dependency`.)
- Tasks: 1.
- Inputs (exit 2 while missing / stale): `analysis/d1_ladder/results/L_predictions_geo.parquet`, `L_report_geo.json`,
  `L_val_geo.csv`, `L_xgb_report_geo.json`, `$LAD/units/L__geo__*.json` (40), `analysis/geo2/results/PREREG_GEO2.md`
  + `prereg_geo2.json` (committed), `P_build_geo.json`, `P_geo_report.json`, `P_geo_sets.json`,
  `$GEO2/inputs/geo.parquet`; the LADDER E2 inputs and outputs as in E-T.
- Outputs: `analysis/geo2/results/E3_metrics.csv`, `E3_bends.csv`, `E3_goal.csv`, `E3_compare.csv`,
  `E3_bend_rates.json`, `E3_report.json` (written last = done marker). Nothing in `analysis/d1_ladder/results`.
- Done: exit 0 and `E3_report.json`; the spec 10-1 '목표 도달' bend part = `E3_report.json goal_reached`
  (`bend_part`, with `mae_part` and `reached`), per-model geo vs g2 numbers = `E3_metrics.csv` (pair_set `P_23`) and
  `E3_report.json model_table`.
- STOP (exit 3, log line `STOP: ...`): the P_report `S_folds.csv` entry (E3-2b, as in E-T), spec 16 E pair lists
  differ (FU2 pair_sets_gate / common_lists), GEO2 sets
  inconsistent (E3-3), PREREG_GEO2 not committed / LADDER snapshot changed / a geo unit without the PREREG sha256
  (E3-4), L_report_geo / L_xgb_report_geo disagree with the units, LINE != SH baseline, or the E2 cross-check
  (E3-7: `E3 is not the E2 computation ...`). Report and wait.
- `eval_geo2_crosscheck.equal` false is recorded, not a STOP (REP8 choice / goal_mae REP8 differs between
  eval_geo2.py and E3): report it with the SUMMARY.

## 3. Task budget
T-R 3 + E-T 1 now (4 tasks; at most 3 running at once). C: 3 tasks. E3: 1 task. Nothing here holds a GPU except the
two `geo gpu` parts (1 GPU each, different steps).
