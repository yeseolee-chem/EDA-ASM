# RUNBOOK — GEO2b tests of the LADDER training slice (`C_tests_geo.json`, design step 10) and the CPU self-tests

Code: `analysis/d1_ladder/tests/run_tests.py` mode `geo` (decisions G-* / GB-T* in its docstring), `analysis/geo2/lad_selftest.py`.
Decisions: `results/DECISIONS_lad.md` (C절). Rules of RUNBOOK_lad.md 0 apply (login node: sbatch / squeue / git only).

## T-0 CPU self-tests (no geo input needed; already run 2026-10-31 via `srun --overlap`, all PASS)
```bash
cd $REPO/analysis/geo2
sbatch --time=48:00:00 -p cpu1,cpu2 -J geo2b_selftest --cpus-per-task=1 --mem=8G --output=$GL/%x_%j.out \
  --export=ALL,M3ENV=reactot jobs/m3_cpu.slurm ../geo2/lad_selftest.py splits ::: ../geo2/lad_selftest.py lists
sbatch --time=48:00:00 -p cpu1,cpu2 -J geo2b_selftest_x --cpus-per-task=1 --mem=8G --output=$GL/%x_%j.out \
  --export=ALL,M3ENV=espley jobs/m3_cpu.slurm ../geo2/lad_selftest.py xgb
sbatch --time=48:00:00 -p cpu1,cpu2 -J geo2b_selftest_a --cpus-per-task=1 --mem=16G --output=$GL/%x_%j.out \
  --export=ALL,M3ENV=d1fu2 jobs/m3_cpu.slurm ../geo2/lad_selftest.py atoms
```
- `splits`: d0d1 build in `$GEO2B/tests_lad/splits_prelim` (preliminary allowed), idempotent bytes, reader check, 9
  negative cases caught, production refusal of a preliminary build.
- `lists`: `train_atoms.py list --stage 1|2` / `train_xgb.py list --stage 1|2` = the LADDER unit names of
  `L_report.json` / `L_xgb_report.json` (regression 1), `--stage 3` = 30 / 10 geo units.
- `xgb` / `atoms`: L-11′ load of the test label file, `load_folds('T1D0')`, rung g2 forced to split T1D0 on the test labels
  = the LADDER T1 lists fold by fold (D0 pairs have no g2 token), stage-3 gates refuse now, LADDER-write guard,
  (xgb) synthetic T1D0 rung through the production XData.split + run_unit (zero model), (atoms) the GPU-projection rule.
- Results: `$GEO2B/tests_lad/selftest_<part>.json` (`all_pass`).

## C-reg regressions only (R-1 .. R-3; no geo input needed) — 3 tasks, needs a GPU
```bash
cd $REPO/analysis/geo2
JG=$(sbatch --parsable --time=48:00:00 -p gpu1,gpu2,gpu3,gpu5,gpu6 --gres=gpu:1 -J geo2b_creg_g --cpus-per-task=4 \
     --mem=48G --output=$GL/%x_%j.out jobs/m3_gpu.slurm tests/run_tests.py geo gpu --regress-only)
JC=$(sbatch --parsable --time=48:00:00 -p cpu1,cpu2 -J geo2b_creg_c --cpus-per-task=32 --mem=64G --output=$GL/%x_%j.out \
     --export=ALL,M3ENV=espley jobs/m3_cpu.slurm tests/run_tests.py geo cpu --regress-only)
sbatch --time=48:00:00 -p cpu1,cpu2 -J geo2b_creg_m --cpus-per-task=1 --mem=4G --output=$GL/%x_%j.out \
     --dependency=afterany:$JG:$JC --export=ALL,M3ENV=reactot jobs/m3_cpu.slurm tests/run_tests.py geo merge --regress-only
```
- Output `analysis/geo2/results/C_tests_geo_regress.json` (`regressions: {R-1, R-2, R-3: pass}`); NOT the training gate.
  Optional early check; the full C run repeats R-1 .. R-3. The cpu part also writes `results/C_l6_xgb_geo.json` (R-2).

## C-geo full tests (the stage-3 training gate) — 3 tasks; after S-1 (final D0D1 index), Z and the inputs role's P
```bash
cd $REPO && bash analysis/geo2/m3_launch.sh ctests
# = (from analysis/geo2)
#   sbatch --time=48:00:00 -p gpu1,gpu2,gpu3,gpu5,gpu6 --gres=gpu:1 -J geo2b_ctest_g --cpus-per-task=4 --mem=48G \
#          --output=$GL/%x_%j.out jobs/m3_gpu.slurm tests/run_tests.py geo gpu
#   sbatch --time=48:00:00 -p cpu1,cpu2 -J geo2b_ctest_c --cpus-per-task=32 --mem=64G --output=$GL/%x_%j.out \
#          --export=ALL,M3ENV=espley jobs/m3_cpu.slurm tests/run_tests.py geo cpu
#   sbatch --time=48:00:00 -p cpu1,cpu2 -J geo2b_ctest_m --cpus-per-task=1 --mem=4G --output=$GL/%x_%j.out \
#          --dependency=afterany:<gpu>:<cpu> --export=ALL,M3ENV=reactot jobs/m3_cpu.slurm tests/run_tests.py geo merge
```
- gpu part (env_d1fu2, 1 GPU): R-1 atoms, L-4 (child process), L-3, L-11′, L-D0, L-5, L-7, L-1, spy run_unit -> L-2 / L-8 /
  L-9, smoke (`train_atoms.py smoke --stage 3`: 6 configs, 2 epochs, smallest + largest cell, gate M-1′ on test pairs and
  on 200 random D0 training pairs; writes `analysis/d1_ladder/results/C_smoke_atoms_geo.json` with `estimate_geo`), R-2
  atoms (L-6 vs the FU2 units, L-10). GPU time: the smoke ~ 6 configs x (2 cells x 2 epochs + 1 final x 2 epochs); total
  part expected 1–3 GPU-h (not measured).
- cpu part (env_espley, 32 CPUs): R-1 XGB, L-4, L-3, L-11′, L-D0, L-5 (+ table provenance), L-1, spies -> L-2 / L-8 /
  L-9, smoke (`C_smoke_xgb_geo.json`), diag (`D0D1_diag_geo.json`), R-2 XGB (L-6 full g1 x S fold 0 = rev 9, L-10 + the
  evaluate.py LINE), R-3 (full g2 x S XGB fold 0 = `L_predictions.parquet` <= 1e-6). Expected 1–3 h on 32 CPUs.
- merge (reactot): `analysis/geo2/results/C_tests_geo.json`; done = `all_pass: true`, `training_gate: true`,
  `labels_geo_final: true`; `gpu_estimate_geo` = the smoke projection, `gpu_cap_hours` = 160 (judged by prereg).
- Exit 3 (`failed`): read `checks.<check>.parts.<gpu|cpu>`; L-D0 `folds.<arm>/f<k>` names the broken rule; L-11
  `d1_max_abs_vs_F_pairs_cr2` / `d0_max_abs_vs_D0_pairs`; smoke `configs.*.d0_train_gates_ok`. Exit 2: a part has not
  run, an env / GPU error, or an input not final (`waiting_reasons`): fix, resubmit that part, then merge alone.
- An XGB check that stops with "GroupKFold(4) fold 0 differs from the split file" is a node tie-order effect (train_xgb
  docstring): resubmit the cpu line with `--nodelist=n112` (or n115), change nothing else.
- Code freeze: any code edit after this run -> rerun all three (prereg compares the code sha256, LP-3).

## What needs a GPU
- C-reg gpu part (R-2 L-6 FU2 tensors run on CPU tensors but `require_gpu` gates the part), C-geo gpu part (L-1, L-2,
  L-4 child, L-7 path, L-8, L-9 spy runs and the atom smoke), and the M3 atom array. Everything else is CPU.
