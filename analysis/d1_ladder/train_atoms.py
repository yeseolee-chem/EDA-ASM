#!/usr/bin/env python3
r"""train_atoms.py — D1-REF·LADDER Phase M1 / M2 atom-token pair models AW / GNN / TF (docs/specs/D1LADDER.md 4-2 .. 4-4,
10, 12): FU2 M-2 'as is', variant A+G, on the rung x arm units  L__<rung>__<arm>__<model>__f<k>  (T1 folds 0..4).
sbatch only (GPU wrapper jobs/gpu.slurm = env_d1fu2). Nothing in this file was run when it was written (login node):
every path is untested until it runs inside a job.

CLI (submit from analysis/d1_ladder; exit 0 = done, 2 = not ready / units missing / usage, 3 = STOP)
  python train_atoms.py list [--stage 1|2]                 unit names in run order (stage 1: 60, stage 2: 30)
  python train_atoms.py smoke [--stage 1] [--labels P]     = run --smoke
  python train_atoms.py run --smoke [--stage 1] [--labels P]
        spec 8 smoke: fold 0, every rung x arm x model of the stage (stage 1 = dft, g1), tuning cells smallest + largest,
        2 epochs, 1 final seed, gate M-1' on every fit -> results/C_smoke_atoms.json. Bypasses the ADD3 gate; writes
        only $LAD/smoke (removed first, never read by run / collect); refuses $LAD/units.
  python train_atoms.py run <i|@I|A> <n> [--stage 1|2] [--labels P]
        units [i::n] of the stage (A = $SLURM_ARRAY_TASK_ID; gpu.slurm replaces @I). Exit 3 unless the ADD3 gate passes
        (results/PREREG_D1PAIR_ADD3.md + prereg_d1pair_add3.json committed, unchanged, sha256 as recorded). Idempotent:
        a unit whose $LAD/units/<unit>.json exists is skipped after its fingerprint is compared (mismatch -> exit 3).
  python train_atoms.py collect [--stage 1|2] [--with-xgb] [--labels P]
        every atom unit of the rungs of stages <= stage (+ the XGB units of train_xgb.py with --with-xgb) ->
        results/L_predictions.parquet, results/L_val.csv, results/L_report.json; exit 2 while units are missing.
  python train_atoms.py estimate
        GPU hours from the 30 cached FU2 Phase M units ($D1PAIR_SCRATCH/fu2/units/*.json, read only) ->
        results/M1_gpu_estimate.json.

GEO2b stage 3 (docs/specs/GEO2.md 8-1 read with GEO1B.md 4-3; user decision D0D1 2026-10-31; decisions GA-* / GB-* at
  the end of this docstring and analysis/geo2/results/DECISIONS_lad.md): rung 'geo' = STAGE_RUNGS[3] only (30 atom units
  L__geo__<arm>__<model>__f<k>) on D0 + D1 pairs: label file $GEO2/F_pairs_d0d1.csv and split 'T1D0' ($GEO2/splits/
  T1D0__f<k>.json, index analysis/geo2/results/S_splits_d0d1.json; both built by analysis/geo2/d0d1_splits.py, whose
  reader rules this file imports). Stage 1 / 2 unit lists, gates, split T1 and outputs are unchanged; but this file's
  sha256 is no longer the one that wrote the LADDER results, so the stage-1 / 2 commands smoke, run, collect and
  estimate STOP before writing anything (GA-13); only `list --stage 1|2` is usable here.
  python train_atoms.py list --stage 3                   the 30 geo units
  python train_atoms.py smoke --stage 3 [--labels P]     fold 0 geo smoke -> results/C_smoke_atoms_geo.json, scratch
        $LAD/smoke_geo (+ a GPU-hour projection from the LADDER g1 / dft units, cap 160 GPU-h, GA-9 / GB-9)
  python train_atoms.py run <i|@I|A> <n> --stage 3 [--labels P]
        gates (exit 3 unless all pass): PREREG_GEO2B (analysis/geo2/results/PREREG_GEO2B.md + prereg_geo2b.json, the
        gate_add3 rule + the LADDER result-file snapshot, GA-4 / GA-5) instead of ADD3; GEO2b Phase P
        (analysis/geo2/results/P_build_geo.json + P_geo_report.json, GB-6) instead of results/P_report.json; the D0D1
        split index (GB-2); analysis/geo2/results/C_tests_geo.json all_pass on this code (GA-7 / GA-14) instead of
        C_tests.json. Units -> $LAD/units/L__geo__* (new files only).
  python train_atoms.py collect --stage 3 [--with-xgb] [--labels P]
        the geo units only -> results/L_predictions_geo.parquet, L_val_geo.csv, L_report_geo.json (same columns /
        keys as the LADDER files; D0 test pairs carry ref_kind / pos 'd0' and ids D#####; --with-xgb needs
        results/L_xgb_report_geo.json of `train_xgb.py check --stage 3`). The existing results/L_predictions.parquet,
        L_val.csv, L_report.json are never written by stage 3.
  geo inputs: table $GEO2/inputs/geo.parquet ($GEO2 = $GEO2_SCRATCH, default /gpfs/tmp_cpu2/yeseo1ee/geo2b = $GEO2B),
  tokens $GEO2/fu2_geo/tokens, reports analysis/geo2/results/fu2_geo (fu2-tokens-3-geo / fu2-atoms-3-geo); D1 and D0
  reactions in one table / one token root (D0 rxn_id = labels_all.json rxn_id < 10^6, D1 >= 1000001; disjoint,
  checked by d0d1_splits.py build).
  The default label file is stage dependent: --stage 3 -> $GEO2/F_pairs_d0d1.csv, else $LAD/F_pairs_cr2.csv (GB-3).

Rules (spec 4-2 .. 4-4, 10; FU2 M-1 / M-2 unchanged unless named)
  pairs / folds  the FU2 T1 split files $D1PAIR_SCRATCH/fu2/splits/T1__f<k>.json (= d1_train.masks arm C + the first
                 inner GroupKFold(4) fold; sha256 checked against analysis/d1_pair/results/fu2/S_splits.json): train =
                 outer-train T_main, test = held-out P_sens u A_main, inner_train / inner_val = tuning. Pairs whose a or b
                 is not usable in the rung (no token, no input-table row, a non-finite H or 182 value) are dropped from
                 all four lists (FU2 dropped pairs without atoms the same way) -> both arms of a rung use P_<rung>.
  arm S          = FU2 exactly: fu2/train.py fit / Std / gate_m1 / predict / batch / epoch_batches imported and called
                 unchanged on LadData (an fu2 train.Data subclass that loads the rung's tokens, the new label file and
                 the usable-pair filter); y = dd_<t> of the label file (default $LAD/F_pairs_cr2.csv).
  arm SH         global token = [token glob (182, X182 row of the rung) | H 9 of $LAD/inputs/<rung>.parquet], standardised
                 per fit on the training reactions (FU2 T-3). Per target beta_t = sum(dH*dy) / sum(dH^2), no intercept,
                 c_ghost beta = 0, fitted ONLY on the fit's own pairs (tuning: inner_train; final: train = T_main);
                 residual r = y - beta*dH, standardised by the fit's residual SD (ddof 0, no mean removed); loss =
                 mean |rhat - r| (standardised, all 10 targets as FU2) + 0.1 * mean |yhat_bar - sum_9 yhat| / s_bar with
                 yhat = beta*dH + rhat * sd_r (the FINAL prediction) and s_bar = the fit's SD of the original dd_barrier;
                 tuning / early-stopping metric = mean over the 9 primary targets of MAE(final yhat, y) / SD(original y)
                 of the fit's training pairs (= the FU2 unit). Training order flips (a, b) <-> (b, a) per pair and epoch
                 with FU2 epoch_batches on the stacked [r_std | baseline] (both change sign together).
  both arms      AdamW 1e-3 / 1e-4, cosine 300, dropout 0.1, batch 64, patience 30, tuning seed 20261019, grid cell =
                 lowest validation metric (ties: grid order), final = all train pairs, the tuned best epoch, seeds
                 20261019 + 0..4, mean; gate M-1' (|g(a,b) + g(b,a)|, |g(a,a)| <= 1e-4 kcal/mol, float64 recomputation,
                 200 random test pairs, inputs only) on every tuning and final fit, on the FINAL prediction (baseline
                 included; dH is exactly antisymmetric in float64).
  validation MAE (spec 13-1) per target, kcal/mol: the final prediction of each tuning fit (inner_train, its best
                 epoch) on inner_val; the chosen cell's values are the unit's 'val_kcal_mae'.

Inputs (Phase P contract; the files are read directly, inputs.py's loader API is not used; inputs.py is imported only
  for check_paths in the ADD3 gate, MA-15)
  $LAD/inputs/<rung>.parquet   one row per reaction: rxn_id (unique), the 9 H columns H_<t> (or hf_<t>; exactly one
                 complete set must exist) for t in d1 d2 elst pauli oi disp cpcm cds barrier (kcal/mol, d1 / d2 /
                 barrier re-assembled by the spec 4-1 rule). Optional and cross-checked when present (STOP on a
                 difference): the 182 global columns under the T_report glob_cols names (= token glob, 1e-12 relative),
                 token_sha256 (= T_manifest sha256), usable (= the rule above).
  rung tokens    dft: analysis/d1_ladder/results/fu2/{T_report.json, T_manifest.csv, Q_report.json, S_splits.json} +
                 $LAD/fu2/tokens (fu2-tokens-3-dft / fu2-atoms-3-dft); g1: analysis/d1_pair/results/fu2 +
                 $D1PAIR_SCRATCH/fu2/tokens (fu2-tokens-3 / fu2-atoms-3); g2: analysis/d1_g2/results/fu2 + $G2/fu2/tokens
                 (-g2). T_report token / atom versions, Q_report / T_manifest / S_splits sha256 and every token sha256
                 are checked as fu2 train.Data does; the token layout must equal fu2/tokens.py BIN_COLS / NUM_BASE (+EXT).
  labels         --labels (default $LAD/F_pairs_cr2.csv): F_pairs_cr2.csv needs results/V_report.json all_pass and its
                 outputs sha256, F_pairs_cr.csv results/Q_report.json likewise, the original D1PAIR F_pairs.csv the
                 sha256 recorded in FU2 S_splits.json; every non-original file must equal F_pairs.csv in every column
                 but dd_d1 / dd_d2 / dd_barrier (non-dd columns identical, dd of the 7 other targets <= 1e-12; L-11).
  L-4            every input path is listed per role in the unit json ('inputs'); the files the process actually opened
                 while loading (sys.addaudithook 'open' events; pyarrow parquet reads are not Python opens and appear
                 only in 'inputs') are summarised in 'opened_inputs'. Rung-independent roles: labels*, split*.

Cache ($LAD/units/<unit>.json written last, .parquet = test predictions, __s<j>.pt = the 5 final models)
  fingerprint = code version d1-ladder-2 + sha256 of this file, lad_common.py and the FU2 code files (fu2 train.py
  CODE_SHA), rung, arm, variant, rung input table sha256, label file sha256, pair-list sha256 (P_<rung>: usable pairs of
  the 5 fold lists), this unit's split lists sha256, S_splits / T_report / T_manifest / Q_report sha256, ADD3 md sha256,
  grid, training rules, torch / CUDA / cuDNN / GPU name / numpy / pandas. A cached unit with another fingerprint ->
  exit 3 (spec 10). collect compares every key but 'gpu' (it may run on CPU, FU2 collect rule).

ADD3 gate (spec 9; = d1_split.gate_add1 / fu2 train.gate_add2): results/PREREG_D1PAIR_ADD3.md and prereg_d1pair_add3.json
  tracked by git and unchanged since HEAD, sha256(md) = json md_sha256, json inputs_sha256 {absolute path: sha256}
  contains at least the label file and $LAD/inputs/<rung>.parquet of every rung trained, and every listed file still
  has its sha256. Paths are absolute, unresolved (os.path.abspath), e.g. /gpfs/tmp_cpu2/yeseo1ee/d1_ladder/F_pairs_cr2.csv.

Outputs
  $LAD/units/L__<rung>__<arm>__<model>__f<k>.{json,parquet}, __s0..4.pt; $LAD/smoke/ (smoke only)
  results/C_smoke_atoms.json, results/L_predictions.parquet (pair_id, ref_kind, pos, fold, rung, arm, model, target, y,
  yhat, base, beta, yhat_seed_sd; S: base = beta = 0), results/L_val.csv (rung, arm, model, fold, target, val_kcal_mae,
  n_val, val_pairs_sha256), results/L_report.json, results/M1_gpu_estimate.json.
  XGB units (train_xgb.py) read by collect --with-xgb: $LAD/units/L__<rung>__<arm>__XGB__f<k>.{json,parquet}, json keys
  parquet_sha256, val_kcal_mae {target: kcal}, val_pairs_sha256 (L-9: must equal the atom units'), parquet columns
  PRED_COLS; collect --with-xgb needs a current results/L_xgb_report.json (train_xgb.py check: full XGB fingerprints,
  per-unit json / parquet sha256) and compares the XGB units' shared fingerprint keys with the atom side (MA-14).

sbatch (login node: one-shot commands only). Stage 1 = 60 units = 4 rung-arms x 3 models x 5 folds (540 trainings).
GPU estimate from the FU2 A+G units (sum of tuning + final fit seconds; 21 on RTX 3090, 9 on A10): AW 0.62 h, GNN
2.30 h (max 3.9 h, A10), TF 0.35 h per unit -> 16.4 GPU-h per rung-arm, ~65.5 GPU-h for stage 1 (SH adds 9 global
columns: negligible). 4 array tasks x 15 units (5 GNN + 5 AW + 5 TF each by the run order) ~ 16.4 h per task.
ONE combined Phase M1 plan with train_xgb.py (review M-2 / M-3; spec 15: at most 4 LADDER jobs at a time): the XGB
array first (%4, ~1-2 h), its check, then the atom array (%4) after the check, then ONE collect --with-xgb. Peak = 4
running LADDER tasks if nothing else of LADDER runs; MaxSubmit: N + 10 <= 20 (array elements count).
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1ladder/analysis/d1_pair; L=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/logs
  squeue -u $USER -h -r | wc -l                          # N (all your tasks, D1-G2 included)
  squeue -u $USER -h -r -o %j | grep -c '^lad'           # C = LADDER tasks running or pending: start M1 only when C = 0
  sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"; sinfo -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 -o "%.9P %.6t %.10G %C"
  JX=$(sbatch --parsable -p cpu1,cpu2 -J lad_xgb --time=48:00:00 --cpus-per-task=32 --mem=64G --array=0-3%4 \
       --output=$L/%x_%A_%a.out --export=ALL,PYENV=espley jobs/py.slurm ../d1_ladder/train_xgb.py run @I 4); echo "JX=$JX"
  JXC=$(sbatch --parsable -p cpu1,cpu2 -J lad_xchk --time=48:00:00 --cpus-per-task=2 --mem=16G --output=$L/%x_%j.out \
       --export=ALL,PYENV=espley --dependency=afterok:${JX:?} --kill-on-invalid-dep=yes \
       jobs/py.slurm ../d1_ladder/train_xgb.py check --stage 1); echo "JXC=$JXC"
  cd ../d1_ladder
  JM=$(sbatch --parsable -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 --gres=gpu:1 -J lad_m1 --time=48:00:00 --cpus-per-task=4 \
       --mem=16G --array=0-3%4 --output=$L/%x_%A_%a.out --dependency=afterany:${JXC:?} \
       jobs/gpu.slurm train_atoms.py run @I 4); echo "JM=$JM"
  sbatch -p cpu1,cpu2 -J lad_m1c --time=48:00:00 --cpus-per-task=2 --mem=24G --output=$L/%x_%j.out \
       --dependency=afterok:${JM:?}:${JXC:?} --kill-on-invalid-dep=yes jobs/gpu.slurm train_atoms.py collect --stage 1 --with-xgb
  (no atom-only collect: two collects would race on the same outputs, lad_common.write_atomic uses one fixed .tmp)
  Resubmission: a task that died -> the JM line with --array=<k> (finished units are skipped; no --dependency) on the
  GPU type its finished units ran on (log line '[lad_gpu] ... gpu=<name>'; another type -> exit 3 at the first cached
  unit), then the collect line with --dependency=afterok:<new JM>.
  Phase C only (no ADD3): `train_atoms.py smoke` runs inside tests/run_tests.py gpu (smoke check); estimate:
  sbatch -p cpu1,cpu2 -J lad_est --time=48:00:00 --cpus-per-task=1 --mem=4G jobs/gpu.slurm train_atoms.py estimate

Decisions (user 2026-10-22: every judgment = the recommended option; recorded here and in the unit json)
  MA-1  inputs.py does not exist yet: the Phase P files are read directly with the contract above (one loader,
        rung_base); a column set other than the contract is a STOP naming the columns found, not a guess.
  MA-2  arm S calls the FU2 functions themselves (no copy), so 'FU2 exactly' holds by construction; only the data object
        differs (rung token root, label file, usable-pair filter). On g1 with the original labels and P_g1 = the FU2
        pairs the tensors, y and s_y equal the FU2 unit (L-6: same T_manifest row order, same split lists).
  MA-3  arm SH = fit_sh, a line-for-line copy of fu2 train.fit with only the spec 4-2 changes (targets, loss, metric);
        it reuses fu2 epoch_batches / batch / predict / Std and models.py unchanged (same RNG stream as FU2).
  MA-4  splits = the FU2 T1 split files (pair-id lists, rung independent); P_<rung> filter as FU2's atom filter. The
        validation pairs (spec 13-1, L-9) are inner_val of those files for every model.
  MA-5  usable reaction = token + input-table row + finite H 9 (+ finite 182 when the table carries them); the same
        P_<rung> for S and SH, so SH vs S (13-4) compares identical pairs.
  MA-6  validation kcal MAE = final prediction of the chosen cell's tuning fit on inner_val (every cell is stored too).
  MA-7  cache compare includes the GPU name (spec 10 literal, FU2 precedent); collect ignores 'gpu' only.
  MA-8  ADD3 json must list the label file and the trained rungs' input tables in inputs_sha256 (as gate_add2 required
        its inputs); no other ADD3 key is interpreted here.
  MA-9  label provenance is checked (V_report / Q_report outputs sha256; original = FU2 S_splits F_pairs sha256) and the
        L-11 invariant (only dd_d1 / dd_d2 / dd_barrier may differ from F_pairs.csv) is enforced on every load.
  MA-10 S predictions carry base = 0 and beta = 0; SH base = beta_final * dH (beta of the T_main fit, identical for the 5
        seeds, asserted), yhat = mean over seeds of the final prediction, yhat_seed_sd = its SD (ddof 0, FU2).
  MA-11 run order = model (GNN, AW, TF) x rung x arm x fold, sliced [i::n]: with n = 4 every task gets 5 GNN + 5 AW + 5 TF.
  MA-12 the 5 final models are saved (.pt) as FU2 did (~1 GB for stage 1 at most); smoke = FU2 smoke rules per config.
  MA-13 L-2 / L-8 asserted in every fit: the beta / scale pairs are exactly the fit's training pairs, none of them is a
        validation or test pair, and no test reaction is a scaling reaction (exit 3 otherwise; spec 16 M).
Review fixes 2026-10-23 (FIXER; recommended option each time)
  MA-6n note for ADD3 (review M-6, no code change: FU2 early stopping on inner_val is fixed by spec 4-3): the atom
        models' val_kcal_mae is the best-epoch value on the same inner_val pairs that early stopping and the cell choice
        used (optimistic); XGB's is the fold-0 score of a cell chosen by the 4-fold mean. Phase A must record this
        selection asymmetry of the spec 13-1 REP rule before training.
  MA-14 collect --with-xgb: results/L_xgb_report.json ok on the same labels / train_xgb.py with every XGB unit's json and
        parquet sha256, and the XGB units' shared fingerprint keys (XGB_SHARED_KEYS) = the atom side's current values.
  MA-15 the ADD3 gate does not open listed files that belong only to other rungs' columns (inputs.check_paths), so a g2
        process opens no dft / g1 path (L-4); every file of the process's own rungs and every shared file is verified.
  MA-16 run: results/P_report.json all_pass and the rung tables + label file sha256 = what inputs.py pairs recorded
        (review P-8: labels and H from one Phase P state).
  MA-17 run --stage 2: results/C_tests.json all_pass with g2_verified (review C-1).
  MA-18 token sha256 cross-check accepts the column token_sha256 or tok_sha256 (the name inputs.py writes; review P-5).
GEO2 additions 2026-10-30 (docs/specs/GEO2.md 8-1; role LADA; reviewed and ported to GEO2b 2026-10-31, GB-*)
  GA-1  rung list and paths only: STAGE_RUNGS[3] = ("geo",), RUNG_ORDER + "geo", RUNGS["geo"], inputs_path("geo") =
        $GEO2/inputs/geo.parquet; models / grids / training rules (fu2 models.py, train.py, the fits here) unchanged.
  GA-2  stage 1 / 2 code paths are unchanged (same unit lists, gates, outputs, messages); every stage-3 branch is
        selected by int(stage) == 3, rung "geo" or split T1D0 only. Exception GA-13: the stage-1 / 2 write commands
        refuse to run with code other than the code recorded in results/L_report.json (= always, with these additions).
  GA-3  stage-3 outputs carry "_geo" before the extension (res_out): L_predictions_geo.parquet, L_val_geo.csv,
        L_report_geo.json, C_smoke_atoms_geo.json; smoke scratch $LAD/smoke_geo (the stage-1 $LAD/smoke is not removed).
  GA-4  PREREG_GEO2B gate = gate_add3 line for line on the GEO2b files (+ json rungs == ["geo"], split == "T1D0"); geo
        units can be written to $LAD/units only after it (_check_root rung check); ADD3 does not open stage 3.
  GA-5  the gate also verifies prereg_geo2b.json ladder_results_sha256 (every analysis/d1_ladder/results file at
        pre-registration, repo-relative keys as D1-G3 F-13) before training and again in collect --stage 3 (spec 14:
        an existing LADDER result file changed = STOP).
  GA-7  C gate for stage 3 = analysis/geo2/results/C_tests_geo.json all_pass (tests/run_tests.py geo).
  GA-8  fingerprint: geo units add prereg_geo2b (sha256 of PREREG_GEO2B.md, None in smoke) and d0d1_code (sha256 of
        analysis/geo2/d0d1_splits.py); add3 stays the ADD3 md sha256 for every rung; the stage 1 / 2 fingerprint dicts
        are unchanged. XGB_SHARED_KEYS + prereg_geo2b, d0d1_code.
  GA-9  smoke --stage 3 adds 'estimate_geo' (read only): per arm x model the LADDER g1 (and dft) atom-unit wall hours
        (results/L_report.json) x (geo smoke tuning epoch seconds / stage-1 smoke g1 (dft) epoch seconds,
        results/C_smoke_atoms.json); the projection = the larger of the two totals, against the cap (GB-9).
  GA-10 no absolute repo path enters a unit fingerprint (code sha256 keys are file names), so the cross-worktree rule of
        D1-G3 F-13 is needed only for the LADDER result snapshot (GA-5). Every existing LADDER unit records the pre-GEO2
        code_sha256 (train_atoms.py fb00f604...), so no stage-1 / 2 unit cache matches this file (GA-13).
  GA-11 lad_common.py unchanged: no stage-3 write goes outside $LAD / analysis/d1_ladder/results.
  GA-12 the geo rung needs inputs.py RUNGS["geo"] (inputs role) and rung_roots entries for $GEO2B and analysis/geo2/
        results (column "geo") so that check_paths classifies the geo files (the ADD3 / PREREG skip rule, MA-15).
  GA-13 guard_ladder_write: smoke / run / collect with --stage 1|2 and estimate STOP (exit 3) before any write or folder
        removal when CODE_SHA != results/L_report.json code_sha256 (the code of the LADDER results; no such file = a
        fresh LADDER tree, old behaviour). With these additions they differ always, so this file can never rewrite
        results/C_smoke_atoms.json, L_predictions.parquet, L_val.csv, L_report.json, M1_gpu_estimate.json, $LAD/smoke or
        a stage-1 / 2 unit. `list` is not guarded (regression 1).
  GA-14 gate_c_geo also requires C_tests_geo.json code_sha256 of train_atoms.py, train_xgb.py, inputs.py, lad_common.py,
        geo2/d0d1_splits.py (= run_tests.py GEO_CODE_FILES keys) = the files now: the tests ran on the code that trains.
GEO2b D0 + D1 (user decision D0D1 2026-10-31; analysis/geo1b/results/D0D1_design_investigation.json design step 9)
  GB-1  split 'T1D0' = the split of rung geo (RUNG_SPLIT); every other rung keeps T1 (load_folds('T1') unchanged).
        load_folds('T1D0') = d0d1_splits.load_s_splits(): S_splits_d0d1.json all_pass / not preliminary, the label file
        and the 5 split files with their recorded sha256, the D1 part of every list = the FU2 T1__f<k> list exactly
        (same ids, same order), D0 pairs (ok True) after it: train / inner_train += D0 fold != k, test += D0 fold k,
        inner_val D1 only; each ok D0 pair tested once; D0 inputs = a final GEO1b build (d0_provenance).
  GB-2  run --stage 3 also needs that index (load_folds raises exit 3 / 2 through it before any unit).
  GB-3  labels: the default label file of --stage 3 is $GEO2/F_pairs_d0d1.csv. label_provenance accepts it only when
        S_splits_d0d1.json records its sha256; load_labels applies L-11' instead of L-11 to it: its first
        len(F_pairs_cr2.csv) bytes are F_pairs_cr2.csv (D1 rows byte-identical; F_pairs_cr2.csv itself then passes the
        LADDER provenance + L-11 rules), the remaining rows = the F_pairs columns of D0_pairs.csv as text
        (d0d1_splits.l11_d0d1); pair ids D##### <-> ref_kind 'd0'.
  GB-4  LadData.split for T1D0: the same filter / leak checks as T1 (test pair or test reaction in train = STOP) plus a
        D0 pair in inner_val = STOP; counts of D0 pairs per list recorded (dropped / n_d0).
  GB-5  run_unit: when the fit's training pairs hold D0 pairs, gate M-1' runs a second time on 200 random D0 training
        pairs of that fit (gate_M1_d0_train; same function, tolerance, float64 recomputation; its own rng seed), on every
        tuning and final fit; failure = STOP; collect requires it ok. T1 units: no such pairs, no key (unchanged json).
  GB-6  GEO2b Phase P gate (gate_p_geo): P_build_geo.json all_pass, not stop, outputs sha256 of $GEO2/inputs/geo.parquet
        = the file; P_geo_report.json stop false. The GEO2 P_geo_sets / |P_23| / LADDER P_report label checks are
        dropped (D0D1: no geo-vs-g2 comparison; the label file is recorded by S_splits_d0d1.json, GB-3).
  GB-9  GPU cap 160 GPU-h (DECISIONS D0D1 '상한 조정'); estimate_geo = max(g1-, dft-scaled totals) + per-slice hours of
        the [i::n] run order for n = 1 .. 10 (launcher sizing); a prereg STOP when over the cap, not a smoke failure.
  GB-10 diagnostics of s_y / SH beta for D1 alone and D1 + D0 come from train_xgb.py diag --stage 3 (CPU, the same pair
        lists after the usable filter); nothing in training reads them.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # no __pycache__ in the read-only analysis/d1_pair/fu2 code folder (spec 15)

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

# ---------------------------------------------------------------- L-4: files this process opens for reading
_OPEN_LOG: list = []
_SKIP_PREFIX = tuple(sorted({p for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, "/proc", "/sys", "/dev", "/etc",
                                         "/usr", "/lib", "/lib64", "/opt") if p}))
_SKIP_EXT = (".py", ".pyc", ".so", ".pth", ".typed")


def _audit(event, args):
    """'open' audit events of reads outside the interpreter / system trees (never raises)."""
    if event != "open":
        return
    try:
        a = tuple(args) + (None, None, None)
        p, mode, flags = a[0], a[1], a[2]
        if p is None or isinstance(p, int):
            return
        if mode is not None:
            if any(c in str(mode) for c in "wax+"):
                return
        elif isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR):
            return
        s = os.path.abspath(os.fsdecode(p))
        if s.endswith(_SKIP_EXT) or s.startswith(_SKIP_PREFIX) or "/site-packages/" in s:
            return
        _OPEN_LOG.append(s)
    except Exception:          # noqa: BLE001  (an audit hook must never raise into the caller)
        return


sys.addaudithook(_audit)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lad_common as L  # noqa: E402

FU2_DIR = L.P_DIR / "fu2"
sys.path.insert(0, str(FU2_DIR))
import torch  # noqa: E402
import fu2_common as F  # noqa: E402          (constants, FU2 paths)
import models as Mo  # noqa: E402             (AW / GNN / TF, GRID)
import train as FT  # noqa: E402              (fu2/train.py: fit, Std, gate_m1, predict, batch, epoch_batches, ...)
import tokens as FTok  # noqa: E402           (TOKEN_VERSION, BIN_COLS, NUM_BASE, NUM_EXT)
import atoms as FA  # noqa: E402              (ATOM_VERSION_D1PAIR)

C = F.C
CODE_VERSION = "d1-ladder-2"                                  # spec 10
VARIANT = "A+G"                                               # spec 4-2 / 4-3
ARMS = ("S", "SH")
MODELS = tuple(Mo.MODELS)                                     # AW, GNN, TF
MODEL_RANK = {"GNN": 0, "AW": 1, "TF": 2}                     # long units first (MA-11)
SPLIT, NF = "T1", 5
FOLD_KEYS = ("train", "test", "inner_train", "inner_val")
STAGE_RUNGS = {1: ("dft", "g1"), 2: ("g2",), 3: ("geo",)}     # stage 3 = GEO2b (docs/specs/GEO2.md 8-1, GA-1)
RUNG_ORDER = ("dft", "g1", "g2", "geo")
GEO_STAGE = 3
sys.path.insert(0, str(L.REPO / "analysis" / "geo2"))
import d0d1_splits as DD  # noqa: E402        (GEO2b D0 + D1 label / split rules, numpy / pandas only; GB-1, GB-3)
SPLIT_D0D1 = DD.SPLIT                                         # "T1D0"
RUNG_SPLIT = {"geo": SPLIT_D0D1}                              # GB-1: rung -> split; other rungs T1
TARGETS = list(F.TARGETS)                                     # d1 d2 elst pauli oi disp cpcm cds c_ghost barrier
I_GHOST = TARGETS.index("c_ghost")
H_TARGETS = [t for t in TARGETS if t != "c_ghost"]            # spec 4-1: the 9 HF values
H_IDX = [TARGETS.index(t) for t in H_TARGETS]
H_PREFIXES = ("H_", "hf_")
CHANGED_DD = ("d1", "d2", "barrier")                          # Q / V change only these (spec 0-1, L-11)
UNITS = L.LAD / "units"
SMOKE = L.LAD / "smoke"
INPUTS = L.LAD / "inputs"
DEFAULT_LABELS = L.LAD / "F_pairs_cr2.csv"
ORIG_LABELS = C.RES / "F_pairs.csv"
FU2_S_SPLITS = F.FU2_RES / "S_splits.json"
ADD3_MD = L.RES / "PREREG_D1PAIR_ADD3.md"
ADD3_JSON = L.RES / "prereg_d1pair_add3.json"
# GEO2b stage 3 (docs/specs/GEO2.md 6-1, 8-1 with GEO1B.md 4-3; read only here; = analysis/geo2/geo2_common GEO2 / RES)
GEO2 = Path(os.environ.get("GEO2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo2b"))      # $GEO2B (= DD.GEO2)
GEO2_RES = L.REPO / "analysis" / "geo2" / "results"
GEO_INPUTS = {"geo": GEO2 / "inputs" / "geo.parquet"}         # the other rungs keep $LAD/inputs/<rung>.parquet
PREREG_GEO2_MD = GEO2_RES / "PREREG_GEO2B.md"                 # analysis/geo2/prereg_geo2b.py (GA-4)
PREREG_GEO2_JSON = GEO2_RES / "prereg_geo2b.json"
C_TESTS_GEO = GEO2_RES / "C_tests_geo.json"                   # tests/run_tests.py geo (GA-7)
CODE_D0D1 = Path(DD.__file__).resolve()                       # GA-8 / GA-14
C_GEO_TRAINER_CODE = {"train_atoms.py": HERE / "train_atoms.py", "train_xgb.py": HERE / "train_xgb.py",
                      "inputs.py": HERE / "inputs.py", "lad_common.py": HERE / "lad_common.py",
                      "geo2/d0d1_splits.py": CODE_D0D1}       # C_tests_geo code_sha256 keys (GA-14)
P_BUILD_GEO = GEO2_RES / "P_build_geo.json"                   # inputs role, GEO2.md 6-1 P (GB-6)
P_GEO_REPORT = GEO2_RES / "P_geo_report.json"                 # inputs role, GEO2.md 7 (GB-6)
SMOKE_GEO = L.LAD / "smoke_geo"
GEO_LIMIT_GPU_H = 160.0                                       # GEO1b DECISIONS D0D1 '상한 조정' (GB-9; GEO2.md 8-4 had 80)
D0_GATE_SEED = 500                                            # GB-5: gate M-1' rng seed offset of the D0 training pairs
PRED_COLS = ["pair_id", "ref_kind", "pos", "fold", "rung", "arm", "model", "target", "y", "yhat", "base", "beta",
             "yhat_seed_sd"]
RUNGS = {   # spec 4-1 columns (token rows); the input table is $LAD/inputs/<rung>.parquet for every rung but geo
    "dft": dict(fu2_res=L.RES / "fu2", tokens=L.LAD / "fu2" / "tokens", tag="dft"),
    "g1": dict(fu2_res=F.FU2_RES, tokens=F.TOKENS, tag=""),
    "g2": dict(fu2_res=L.G2_DIR / "results" / "fu2", tokens=L.G2 / "fu2" / "tokens", tag="g2"),
    "geo": dict(fu2_res=GEO2_RES / "fu2_geo", tokens=GEO2 / "fu2_geo" / "tokens", tag="geo"),     # GEO2 8-1
}
# hashed once at import: the code this process runs (fu2 train.py review F3 rule)
CODE_SHA = {"train_atoms.py": L.sha256(Path(__file__).resolve()), "lad_common.py": L.sha256(Path(L.__file__).resolve()),
            **{(k if k == "d1pair_common.py" else f"fu2/{k}"): v for k, v in FT.CODE_SHA.items()}}
_GATE: dict = {}


# ================================================================ small helpers
def stop(msg):
    L.stop(msg)


def not_ready(msg):
    print(f"NOT READY: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def _abs(p):
    return Path(os.path.abspath(p))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    return o


def write_json(path, obj):
    return L.write_json(path, _jsonable(obj))


def norm(obj):
    """the value as it reads back from json (tuples -> lists, numpy -> python); fingerprints are compared this way."""
    return json.loads(json.dumps(_jsonable(obj)))


def inputs_path(rung):
    return GEO_INPUTS[rung] if rung in GEO_INPUTS else INPUTS / f"{rung}.parquet"


def res_out(name, stage):
    """results/<name> for stages 1 / 2; results/<stem>_geo<suffix> for stage 3 (GA-3: never the LADDER files)."""
    p = L.RES / name
    return p.with_name(f"{p.stem}_geo{p.suffix}") if int(stage) == GEO_STAGE else p


def collect_rungs(stage):
    """rungs of collect: every rung of the stages <= stage (LADDER); stage 3 = the geo rung only (GEO2 8-1)."""
    return tuple(STAGE_RUNGS[GEO_STAGE]) if int(stage) == GEO_STAGE else rungs_upto(stage)


def rung_split(rung):
    """GB-1: the split of a rung (geo -> T1D0, every other rung -> T1)."""
    return RUNG_SPLIT.get(rung, SPLIT)


def stage_labels(stage):
    """GB-3: the default label file of a stage (3 -> $GEO2B/F_pairs_d0d1.csv, else $LAD/F_pairs_cr2.csv)."""
    return Path(DD.LABELS) if int(stage) == GEO_STAGE else DEFAULT_LABELS


def uname(u):
    return f"L__{u['rung']}__{u['arm']}__{u['model']}__f{u['fold']}"


def unit_list(stage=1, rungs=None):
    rungs = STAGE_RUNGS[int(stage)] if rungs is None else rungs
    us = [dict(rung=r, arm=a, model=m, fold=k) for r in rungs for a in ARMS for m in MODELS for k in range(NF)]
    return sorted(us, key=lambda u: (MODEL_RANK[u["model"]], RUNG_ORDER.index(u["rung"]), ARMS.index(u["arm"]),
                                     u["fold"]))


def rungs_upto(stage):
    return tuple(r for s in sorted(STAGE_RUNGS) if s <= int(stage) for r in STAGE_RUNGS[s])


def opened_summary(paths):
    """L-4 record: per directory the number of files opened; the file names outside directories with > 20 files."""
    c = Counter(os.path.dirname(p) for p in paths)
    big = {d for d, n in c.items() if n > 20}
    return dict(dirs=dict(sorted(c.items())), files=sorted({p for p in paths if os.path.dirname(p) not in big}))


def _ids(D, rows):
    return [str(q) for q in D.P.index.to_numpy()[rows]]


def _num(df):
    """DataFrame -> float64 array; None / NA -> NaN (pd.to_numeric), anything non-numeric raises."""
    return df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)


def _maxdiff(a, b):
    """max |a - b| with equal NaN patterns required (inf when the NaN patterns differ)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    m = ~np.isnan(a)
    return float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0


# ================================================================ inputs
def versions_expected(rung):
    tag = RUNGS[rung]["tag"]
    sfx = f"-{tag}" if tag else ""
    return FTok.TOKEN_VERSION + sfx, FA.ATOM_VERSION_D1PAIR + sfx


def load_tokens(rung):
    """the rung's token reactions as fu2 train.Data arrays (same checks, same T_manifest row order)."""
    cfg = RUNGS[rung]
    res, tdir = Path(cfg["fu2_res"]), Path(cfg["tokens"])
    tpath, mpath, qpath, spath = (res / n for n in ("T_report.json", "T_manifest.csv", "Q_report.json", "S_splits.json"))
    miss = [str(p) for p in (tpath, mpath, qpath, spath) if not p.is_file()]
    if miss or not tdir.is_dir():
        not_ready(f"rung {rung}: token files missing {miss or [str(tdir)]}")
    trep, qrep = json.loads(tpath.read_text()), json.loads(qpath.read_text())
    if trep.get("stop") or qrep.get("stop"):
        stop(f"rung {rung}: T_report / Q_report stop")
    tv, av = versions_expected(rung)
    num_want = list(FTok.NUM_BASE) + (list(FTok.NUM_EXT) if trep.get("ext_used") else [])
    sha = dict(t_report=L.sha256(tpath), t_manifest=L.sha256(mpath), q_report=L.sha256(qpath),
               s_splits_rung=L.sha256(spath))
    checks = (("token_version", trep.get("token_version") == tv), ("atom_version", qrep.get("atom_version") == av),
              ("q_report_sha256", trep.get("q_report_sha256") == sha["q_report"]),
              ("splits_sha256 = rung S_splits.json", trep.get("splits_sha256") == sha["s_splits_rung"]),
              ("splits_sha256 = FU2 S_splits.json", trep.get("splits_sha256") == L.sha256(FU2_S_SPLITS)),
              ("manifest_sha256", trep.get("manifest_sha256") == sha["t_manifest"]),
              ("bin_cols", list(trep.get("bin_cols") or []) == list(FTok.BIN_COLS)),
              ("num_cols", list(trep.get("num_cols") or []) == num_want),
              ("glob_cols 182", len(trep.get("glob_cols") or []) == 182))
    bad = [k for k, ok in checks if not ok]
    if bad:
        stop(f"rung {rung}: token report does not match the current inputs / expected layout: {bad}")
    man = pd.read_csv(mpath)
    elements = [int(z) for z in trep["elements"]]
    num_cols, bin_cols = list(trep["num_cols"]), list(trep["bin_cols"])
    rids = man.rxn_id.astype(int).to_numpy()
    row = {int(r): k for k, r in enumerate(rids)}
    nr, nmax = len(rids), int(man.n_atoms.max())
    el = {z: k for k, z in enumerate(elements)}
    onehot = np.zeros((nr, nmax, len(elements)))
    binm = np.zeros((nr, nmax, len(bin_cols)))
    num = np.zeros((nr, nmax, len(num_cols)))
    pos = np.zeros((nr, nmax, 3))
    mask = np.zeros((nr, nmax), dtype=bool)
    etype = np.zeros((nr, nmax, nmax), dtype=np.int8)
    frag = np.zeros((nr, nmax), dtype=np.int8)
    glob = np.zeros((nr, len(trep["glob_cols"])))
    for k, (rid, s) in enumerate(zip(rids, man.sha256)):            # = fu2 train.Data.__init__
        p = tdir / f"{rid}.npz"
        if L.sha256(p) != s:
            stop(f"{p} sha256 differs from {mpath}")
        t = np.load(p)
        n = len(t["Z"])
        onehot[k, np.arange(n), [el[int(z)] for z in t["Z"]]] = 1.0
        binm[k, :n], num[k, :n], pos[k, :n] = t["bin"], t["num"], t["pos"]
        mask[k, :n] = True
        frag[k, :n] = t["bin"][:, 0]
        for (i, j), e in zip(t["edges"], t["etype"]):
            etype[k, i, j] = etype[k, j, i] = int(e)
        glob[k] = t["glob"]
    return dict(trep=trep, elements=elements, num_cols=num_cols, bin_cols=bin_cols, rids=rids, row=row, nmax=nmax,
                n=man.n_atoms.to_numpy(int), onehot=onehot, bin=binm, num=num, pos=pos, mask=mask, etype=etype,
                frag=frag, glob182=glob, man_sha=dict(zip(rids.tolist(), man.sha256.astype(str))), sha=sha,
                d_in=len(elements) + len(bin_cols) + len(num_cols),
                inputs=dict(t_report=str(tpath), t_manifest=str(mpath), q_report=str(qpath), s_splits_rung=str(spath),
                            tokens_dir=str(tdir), n_tokens=int(nr)))


def load_inputs(rung, tok):
    """$LAD/inputs/<rung>.parquet -> H 9 per token row (NaN where absent), usable reactions (MA-5), info."""
    path = inputs_path(rung)
    if not path.is_file():
        not_ready(f"{path} missing (Phase P)")
    X = pd.read_parquet(path)
    if "rxn_id" not in X.columns:
        stop(f"{path}: no rxn_id column (columns {list(X.columns)[:30]})")
    X = X.assign(rxn_id=X.rxn_id.astype(int))
    if X.rxn_id.duplicated().any():
        stop(f"{path}: duplicated rxn_id")
    sets = [pre for pre in H_PREFIXES if all(f"{pre}{t}" in X.columns for t in H_TARGETS)]
    if len(sets) != 1:
        stop(f"{path}: need exactly one complete set of H columns {[p + '<t>' for p in H_PREFIXES]} for {H_TARGETS}; "
             f"found sets {sets}; columns {list(X.columns)[:40]} ... (Phase P contract, train_atoms.py docstring)")
    hcols = [f"{sets[0]}{t}" for t in H_TARGETS]
    X = X.set_index("rxn_id")
    rids, nr = tok["rids"], len(tok["rids"])
    common = [int(r) for r in rids if int(r) in X.index]
    rows = np.array([tok["row"][r] for r in common], dtype=int)
    H9 = np.full((nr, len(H_TARGETS)), np.nan)
    in_tab = np.zeros(nr, dtype=bool)
    H9[rows] = _num(X.loc[common, hcols])
    in_tab[rows] = True
    ok182 = np.ones(nr, dtype=bool)
    gcols = list(tok["trep"]["glob_cols"])
    x182 = "columns absent: not cross-checked"
    if all(c in X.columns for c in gcols):
        G = _num(X.loc[common, gcols])
        fin = np.isfinite(G).all(1)
        ok182[rows] = fin
        g0 = tok["glob182"][rows]
        dev = np.abs(G - g0) > 1e-12 * np.maximum(1.0, np.abs(g0))
        nbad = int((dev & np.isfinite(G)).any(1).sum())
        if nbad:
            stop(f"{path}: {nbad} reactions whose 182 columns differ from their token glob (> 1e-12 relative)")
        x182 = "equal to the token glob"
    tsha = "column absent: not cross-checked"
    for col in ("token_sha256", "tok_sha256"):          # contract name / the name inputs.py writes (review P-5 / M-5)
        if col in X.columns:
            bad = [r for r in common if str(X.at[r, col]) != tok["man_sha"][r]]
            if bad:
                stop(f"{path}: {col} differs from T_manifest for {len(bad)} reactions, e.g. {bad[:5]}")
            tsha = f"{col} equal to T_manifest"
    usable = in_tab & np.isfinite(H9).all(1) & ok182
    uchk = "column absent: not cross-checked"
    if "usable" in X.columns:
        theirs = np.zeros(nr, dtype=bool)
        theirs[rows] = X.loc[common, "usable"].astype(bool).to_numpy()
        if (theirs != usable).any():
            stop(f"{path}: 'usable' differs from token + table row + finite H (+182) for "
                 f"{int((theirs != usable).sum())} token reactions")
        uchk = "equal"
    info = dict(path=str(path), sha256=L.sha256(path), n_rows=int(len(X)), n_token_rows_in_table=len(common),
                n_table_rows_without_token=int(len(X) - len(common)), h_cols=hcols, x182_check=x182,
                token_sha256_check=tsha, usable_check=uchk, n_usable=int(usable.sum()),
                n_nonfinite_h=int((in_tab & ~np.isfinite(H9).all(1)).sum()))
    return H9, usable, info


def _dd(fn, *a, **k):
    """a d0d1_splits rule -> exit 3 (Stop) / exit 2 (NotReady) with its message."""
    try:
        return fn(*a, **k)
    except DD.NotReady as e:
        not_ready(str(e))
    except DD.Stop as e:
        stop(str(e))


def label_provenance(path, sha):
    """MA-9: the label file must be a passed Q / V output or the original D1PAIR file (tests: others, recorded).
    GB-3: or the GEO2b D0 + D1 file whose sha256 S_splits_d0d1.json records (d0d1_splits.load_s_splits)."""
    if path == _abs(DD.LABELS):
        rec = _dd(DD.load_s_splits)[0]
        if (rec.get("outputs") or {}).get(str(path)) != sha:
            stop(f"{path}: sha256 differs from {DD.S_SPLITS.name} outputs")
        return (f"{DD.S_SPLITS.name} all_pass (preliminary {rec.get('preliminary')}), outputs sha256 equal; D0 = GEO1b "
                f"build {(rec.get('d0') or {}).get('g2_source')} (D0_pairs.csv {str((rec.get('d0') or {}).get('d0_pairs_sha256'))[:12]})")
    if path == _abs(ORIG_LABELS):
        want = json.loads(FU2_S_SPLITS.read_text()).get("inputs_sha256", {}).get("F_pairs")
        if sha != want:
            stop(f"{path} sha256 {sha[:12]} != FU2 S_splits.json inputs_sha256.F_pairs {str(want)[:12]}")
        return "D1PAIR F_pairs.csv (sha256 = FU2 S_splits.json inputs_sha256.F_pairs)"
    reports = {_abs(L.LAD / "F_pairs_cr2.csv"): L.RES / "V_report.json",
               _abs(L.LAD / "F_pairs_cr.csv"): L.RES / "Q_report.json"}
    rp = reports.get(path)
    if rp is None:
        return "unverified (not a Q / V output, not the original; tests only — the ADD3 gate covers production)"
    rep = json.loads(rp.read_text()) if rp.is_file() else {}
    if not rep.get("all_pass"):
        stop(f"{path}: {rp.name} is missing or not all_pass")
    rec = (rep.get("outputs") or {}).get(str(L.LAD / path.name))
    if rec != sha:
        stop(f"{path}: sha256 differs from {rp.name} outputs ({str(rec)[:12]} vs {sha[:12]})")
    return f"{rp.name} all_pass, outputs sha256 equal"


def load_labels(labels):
    """label file -> pair table (index pair_id) + info; default float parser (= d1_train.data, FU2 pairs.parquet)."""
    path = _abs(labels)
    if not path.is_file():
        not_ready(f"label file {path} missing")
    sha = L.sha256(path)
    prov = label_provenance(path, sha)
    P = pd.read_csv(path)
    need = ["pair_id", "ref_kind", "a_rxn", "b_rxn", "pos"] + [f"dd_{t}" for t in TARGETS]
    miss = [c for c in need if c not in P.columns]
    if miss:
        stop(f"{path}: columns missing {miss}")
    if P.pair_id.duplicated().any():
        stop(f"{path}: duplicated pair_id")
    l11 = "original labels"
    if path == _abs(DD.LABELS):                             # GB-3: L-11' on the D0 + D1 file
        l11 = l11_d0d1(path, P)
    elif path != _abs(ORIG_LABELS):                         # L-11 invariant on every load (MA-9)
        O = pd.read_csv(ORIG_LABELS)
        l11 = _l11_vs_orig(path, P, O)
    P = P.assign(pair_id=P.pair_id.astype(str), ref_kind=P.ref_kind.fillna("").astype(str),
                 pos=P.pos.fillna("").astype(str), a_rxn=P.a_rxn.astype(int), b_rxn=P.b_rxn.astype(int))
    return P.set_index("pair_id"), dict(path=str(path), sha256=sha, provenance=prov, l11=l11, n_pairs=int(len(P)))


def _l11_vs_orig(path, P, O):
    """L-11 (MA-9): a re-referenced label file = F_pairs.csv outside dd_d1 / dd_d2 / dd_barrier."""
    if list(P.columns) != list(O.columns) or P.pair_id.tolist() != O.pair_id.tolist():
        stop(f"{path}: columns / pair order differ from {ORIG_LABELS}")
    ch = {f"dd_{t}" for t in CHANGED_DD}
    other = [c for c in O.columns if not c.startswith("dd_") and not P[c].equals(O[c])]
    dmax = {t: _maxdiff(P[f"dd_{t}"], O[f"dd_{t}"]) for t in TARGETS if f"dd_{t}" not in ch}
    if other or max(dmax.values()) > 1e-12:
        stop(f"{path}: differs from F_pairs.csv outside dd_d1 / dd_d2 / dd_barrier: columns {other}, dd {dmax}")
    return dict(non_dd_columns_identical=True, unchanged_dd_max_abs=dmax)


def l11_d0d1(path, P):
    """GB-3 (L-11'): the D1 rows of the D0 + D1 file are F_pairs_cr2.csv byte for byte, F_pairs_cr2.csv passes the LADDER
    provenance (V_report) and L-11, the D0 rows are the F_pairs columns of D0_pairs.csv; ids D##### <-> ref_kind d0."""
    rec = _dd(DD.l11_d0d1, path=path)
    cr2 = _abs(DEFAULT_LABELS)
    if _abs(DD.CR2) != cr2:
        stop(f"d0d1_splits CR2 {DD.CR2} != {cr2}")
    prov2 = label_provenance(cr2, L.sha256(cr2))
    O = pd.read_csv(ORIG_LABELS)
    n1 = len(O)
    l11 = _l11_vs_orig(cr2, pd.read_csv(cr2), O)           # the D1 prefix IS this file (byte check above)
    d0 = P.ref_kind.fillna("").astype(str).eq("d0").to_numpy()
    ids = P.pair_id.astype(str)
    if len(P) != n1 + rec["n_d0_rows"] or d0[:n1].any() or not d0[n1:].all() \
            or (ids.str.startswith("D").to_numpy() != d0).any():
        stop(f"{path}: rows are not 'the {n1} D1 rows, then {rec['n_d0_rows']} D0 rows' (ref_kind d0 <-> D#####)")
    return dict(rec, F_pairs_cr2_provenance=prov2, F_pairs_cr2_l11=l11, n_rows=int(len(P)))


def load_folds(split=SPLIT):
    """the FU2 T1 split files (sha256 = FU2 S_splits.json), rung independent. split 'T1D0' (GB-1): the GEO2b D0 + D1
    split files through d0d1_splits.load_s_splits (every rule of its reader contract) -> (rec, folds, files)."""
    if split == SPLIT_D0D1:
        if _abs(DD.FU2_SPLITS) != _abs(F.SPLITS) or _abs(DD.FU2_S_SPLITS) != _abs(FU2_S_SPLITS):
            stop(f"d0d1_splits FU2 paths {DD.FU2_SPLITS}, {DD.FU2_S_SPLITS} != {F.SPLITS}, {FU2_S_SPLITS}")
        return _dd(DD.load_s_splits)
    if split != SPLIT:
        stop(f"split {split}: LADDER uses T1 only (spec 4-3)")
    srep = json.loads(FU2_S_SPLITS.read_text())
    folds, files = {}, {}
    for k in range(NF):
        name = f"{SPLIT}__f{k}.json"
        p = F.SPLITS / name
        if not p.is_file() or L.sha256(p) != srep["units"][name]["sha256"]:
            stop(f"{p} missing or changed after {FU2_S_SPLITS}")
        folds[k] = json.loads(p.read_text())
        files[f"split_T1_f{k}"] = str(p)
    return srep, folds, files


_BASE: dict = {}


def rung_base(rung, labels):
    """everything of a rung that both arms share (cached per process)."""
    key = (rung, str(_abs(labels)))
    if key in _BASE:
        return _BASE[key]
    if rung not in RUNGS:
        stop(f"unknown rung {rung}")
    t0, i0 = time.time(), len(_OPEN_LOG)
    split = rung_split(rung)
    if split == SPLIT_D0D1 and _abs(GEO2) != _abs(DD.GEO2):
        stop(f"$GEO2_SCRATCH roots differ: train_atoms {GEO2} vs d0d1_splits {DD.GEO2}")
    tok = load_tokens(rung)
    H9, ur, hinfo = load_inputs(rung, tok)
    P, linfo = load_labels(labels)
    srep, folds, sfiles = load_folds(split)
    s_index = FU2_S_SPLITS if split == SPLIT else Path(DD.S_SPLITS)
    fold_ids = {q for f in folds.values() for k in FOLD_KEYS for q in f[k]}
    miss = sorted(fold_ids - set(P.index))
    if miss:
        stop(f"{len(miss)} split pair ids are not in {linfo['path']}: {miss[:5]}")
    is_d0 = P.ref_kind.eq("d0").to_numpy()
    if split == SPLIT and is_d0[np.isin(P.index.to_numpy(), sorted(fold_ids))].any():
        stop(f"split T1 holds D0 pairs (ref_kind d0) of {linfo['path']}")
    y = P[[f"dd_{t}" for t in TARGETS]].to_numpy(float)
    row = tok["row"]
    ra = np.array([row.get(int(r), -1) for r in P.a_rxn])
    rb = np.array([row.get(int(r), -1) for r in P.b_rxn])
    has_tok = (ra >= 0) & (rb >= 0)
    usable_pair = has_tok & ur[np.maximum(ra, 0)] & ur[np.maximum(rb, 0)]
    H10 = np.zeros((len(tok["rids"]), len(TARGETS)))
    for jh, j in enumerate(H_IDX):
        H10[:, j] = np.where(ur, H9[:, jh], 0.0)                  # c_ghost column stays 0 (beta 0)
    dH = np.zeros_like(y)
    dH[usable_pair] = H10[ra[usable_pair]] - H10[rb[usable_pair]]
    pid = P.index.to_numpy()
    in_folds = np.isin(pid, sorted({q for f in folds.values() for q in f["train"] + f["test"]}))
    pair_list = sorted(pid[usable_pair & in_folds].tolist())
    inputs = dict(tok.pop("inputs"), inputs_table=hinfo["path"], labels=linfo["path"], labels_original_l11=(
        str(ORIG_LABELS) if linfo["path"] != str(_abs(ORIG_LABELS)) else None), split_s_splits=str(s_index),
        **sfiles)
    if split == SPLIT_D0D1:                                         # GB-3: the L-11' sources are inputs too (L-4)
        inputs.update(labels_cr2_l11=str(_abs(DEFAULT_LABELS)), labels_v_report=str(L.RES / "V_report.json"))
    sha = dict(tok.pop("sha"), inputs_table=hinfo["sha256"], labels=linfo["sha256"], s_splits=L.sha256(s_index))
    b = dict(tok, H9=H9, H10=H10, usable_rxn=ur, hinfo=hinfo, P=P, linfo=linfo, srep=srep, folds=folds, y=y,
             pid_row={p: k for k, p in enumerate(pid)}, ra=ra, rb=rb, has_tok=has_tok, usable_pair=usable_pair, dH=dH,
             pair_list=pair_list, pair_list_sha=C.list_sha([(q,) for q in pair_list]), inputs=inputs, sha=sha,
             labels_path=linfo["path"], opened=list(_OPEN_LOG[i0:]), load_seconds=time.time() - t0, split=split,
             is_d0=is_d0)
    n_d0 = int((usable_pair & in_folds & is_d0).sum())
    print(f"[data] rung {rung}: {len(tok['rids'])} token reactions, {hinfo['n_usable']} usable, {len(pair_list)} pairs "
          f"in P_{rung} ({int(usable_pair.sum())} usable of {len(P)}), labels {Path(linfo['path']).name}"
          + (f", split {split} ({n_d0} D0 pairs, nmax {tok['nmax']} atoms)" if split == SPLIT_D0D1 else "")
          + f" ({b['load_seconds']:.0f} s)", flush=True)
    _BASE[key] = b
    return b


class LadData(FT.Data):
    """fu2 train.Data for one rung x arm: the FU2 attributes (onehot, bin, num, pos, mask, etype, frag, n, glob, y, ra, rb,
    P, d_in, d_glob, static()) from the rung's tokens and the label file, plus H10 / dH (SH baseline) and the usable-pair
    filter. fu2 train.Data.__init__ (FU2 paths) is not called; static() is inherited."""

    def __init__(self, rung, arm, labels=DEFAULT_LABELS):
        if arm not in ARMS:
            stop(f"unknown arm {arm}")
        b = rung_base(rung, labels)
        for k in ("trep", "elements", "num_cols", "bin_cols", "rids", "row", "nmax", "n", "onehot", "bin", "num", "pos",
                  "mask", "etype", "frag", "d_in", "P", "y", "pid_row", "ra", "rb", "has_tok", "usable_rxn",
                  "usable_pair", "H10", "dH", "srep", "folds", "pair_list", "pair_list_sha", "inputs", "sha", "opened",
                  "hinfo", "linfo", "is_d0"):
            setattr(self, k, b[k])
        self.rung, self.arm, self.labels = rung, arm, b["labels_path"]
        self.split_name = b["split"]
        self.glob = b["glob182"] if arm == "S" else np.concatenate([b["glob182"], b["H10"][:, H_IDX]], 1)
        self.d_glob = self.glob.shape[1]
        self.s_splits_sha = self.sha["s_splits"]

    def split(self, a, b=None):
        """split(k) or split("T1", k) (= fu2 train.Data.split argument order) -> {train, test, inner_train, inner_val}:
        pair-table rows of the usable pairs + dropped counts, list sha256; leak checks. Rung geo: split T1D0 (GB-4)."""
        split, k = (self.split_name, a) if b is None else (a, b)
        if split != self.split_name:
            stop(f"split {split}: LADDER uses T1 only (spec 4-3)" if self.split_name == SPLIT else
                 f"split {split}: rung {self.rung} uses {self.split_name} only (GB-1)")
        rec = self.folds[int(k)]
        out, dropped = {}, {}
        for key in FOLD_KEYS:
            rows = np.array([self.pid_row[q] for q in rec[key]], dtype=int)
            keep = self.usable_pair[rows]
            out[key] = rows[keep]
            dropped[key] = dict(no_tokens=int((~self.has_tok[rows]).sum()), not_usable=int((~keep).sum()))
        st = {key: set(out[key].tolist()) for key in FOLD_KEYS}
        if st["test"] & st["train"]:
            stop(f"{self.rung} fold {k}: test pair in train")
        if st["inner_train"] | st["inner_val"] != st["train"] or st["inner_train"] & st["inner_val"]:
            stop(f"{self.rung} fold {k}: inner folds do not partition train")
        rx = lambda r: set(self.ra[r].tolist()) | set(self.rb[r].tolist())          # noqa: E731
        if rx(out["test"]) & rx(out["train"]):
            stop(f"{self.rung} fold {k}: test reaction in train")
        if self.split_name == SPLIT_D0D1:                         # GB-4: validation stays D1 only
            if self.is_d0[out["inner_val"]].any():
                stop(f"{self.rung} fold {k}: D0 pairs in inner_val")
            out["n_d0"] = {key: int(self.is_d0[out[key]].sum()) for key in FOLD_KEYS}
        ids = {key: _ids(self, out[key]) for key in FOLD_KEYS}
        out.update(dropped=dropped,
                   pairs_sha=C.list_sha([(q,) for q in ids["train"] + ["|"] + ids["test"] + ["|"] + ids["inner_train"]
                                         + ["|"] + ids["inner_val"]]),
                   val_sha=C.list_sha([(q,) for q in ids["inner_val"]]), test_sha=C.list_sha([(q,) for q in ids["test"]]))
        return out

    def d0_rows(self, rows):
        """GB-5: the D0 pairs among pair-table rows (empty for T1)."""
        rows = np.asarray(rows, dtype=int)
        return rows[self.is_d0[rows]] if len(rows) else rows


_DATA: dict = {}


def get_data(rung, arm, labels=None):
    """rung x arm data; labels None -> the rung's default label file (geo: $GEO2B/F_pairs_d0d1.csv, else cr2; GB-3)."""
    if labels is None:
        labels = stage_labels(GEO_STAGE) if rung_split(rung) == SPLIT_D0D1 else DEFAULT_LABELS
    return _get_data(rung, arm, labels)


def _get_data(rung, arm, labels):
    key = (rung, arm, str(_abs(labels)))
    if key not in _DATA:
        _DATA[key] = LadData(rung, arm, labels)
    return _DATA[key]


# ================================================================ arm SH (spec 4-2)
def beta_fit(dH, y):
    """per target beta_t = sum(dH_t * y_t) / sum(dH_t^2) (no intercept); c_ghost 0. dH, y [n, 10] (fit pairs only)."""
    dH, y = np.asarray(dH, float), np.asarray(y, float)
    num, den = (dH * y).sum(0), (dH * dH).sum(0)
    beta = np.zeros(len(TARGETS))
    for j in H_IDX:
        if not (np.isfinite(den[j]) and den[j] > 0 and np.isfinite(num[j])):
            stop(f"beta for {TARGETS[j]}: sum(dH^2) = {den[j]} on the fit pairs")
        beta[j] = num[j] / den[j]
    return beta


class StdSH(FT.Std):
    """fu2 train.Std (token / global scaling and SD of the original y from the fit's training pairs) + the SH baseline:
    beta and the residual SD from the same pairs (L-2: beta_rows is pair_rows)."""

    def __init__(self, D, pair_rows):
        super().__init__(D, pair_rows)
        y, dH = D.y[self.pair_rows], D.dH[self.pair_rows]
        self.beta = beta_fit(dH, y)
        self.beta_rows = self.pair_rows
        self.base_tr = self.beta * dH
        self.r_tr = y - self.base_tr
        self.sr = self.r_tr.std(0)                                     # ddof 0, no mean removed
        if not (np.all(np.isfinite(self.sr)) and np.all(self.sr > 0)):
            stop(f"SH residual SD not positive / finite on the fit pairs: {self.sr.tolist()}")


def loss_sh(yhat, r, sr, base, sy):
    """SH loss: mean |rhat - r| (standardised residual, 10 targets) + 0.1 * mean |kc_bar - sum_9 kc| / s_bar, kc = the
    final kcal prediction base + rhat * sr, s_bar = SD of the original dd_barrier of the fit's training pairs."""
    l1 = (yhat - r).abs().mean()
    kc = yhat * sr + base
    pen = ((kc[:, F.I_BAR] - kc[:, F.I_CH9].sum(1)).abs() / sy[F.I_BAR]).mean()
    return l1 + F.LAMBDA * pen, l1, pen


def final_sh(model, T, D, std, ra, rb, bs=256):
    """SH final prediction in kcal/mol (float64): beta * (H(a) - H(b)) + rhat * sr for reaction rows (ra, rb)."""
    return FT.predict(model, T, ra, rb, bs=bs) * std.sr + std.beta * (D.H10[ra] - D.H10[rb])


def fit_sh(D, model_name, cell, variant, tr, val=None, seed=F.SEED, n_epochs=F.MAX_EPOCHS, patience=F.PATIENCE,
           dropout=F.DROPOUT, batch_size=F.BATCH, log=None):
    """fu2 train.fit with the spec 4-2 SH changes only -> dict(model, std, sy, sr, beta, T, history, best_epoch,
    best_val, best_val_per_target, seconds)."""
    FT.seed_all(seed)
    rng = np.random.default_rng(seed)
    std = StdSH(D, tr)
    T = std.tensors(D)
    sr = torch.tensor(std.sr, dtype=torch.float32, device=FT.DEV)
    sy = torch.tensor(std.sy, dtype=torch.float32, device=FT.DEV)
    nt = len(F.TARGETS)
    # [standardised residual | baseline kcal]: fu2 epoch_batches flips both with the (a, b) order
    ytr = torch.tensor(np.concatenate([std.r_tr / std.sr, std.base_tr], 1), dtype=torch.float32, device=FT.DEV)
    model = Mo.build(model_name, cell, D.d_in, nt, D.d_glob, variant, dropout).to(FT.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=F.LR, weight_decay=F.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=F.MAX_EPOCHS)
    ra_all, rb_all = D.ra[tr], D.rb[tr]
    hist, best = [], dict(val=np.inf, epoch=0, state=None, per_target=None)
    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        model.train()
        tl = []
        for ra, rb, yb in FT.epoch_batches(rng, ra_all, rb_all, ytr, batch_size):
            R, ia, ib = FT.batch(T, ra, rb)
            loss, l1, pen = loss_sh(model(R, ia, ib), yb[:, :nt], sr, yb[:, nt:], sy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl.append((float(loss), float(l1), float(pen)))
        sched.step()
        rec = dict(epoch=ep, loss=float(np.mean([x[0] for x in tl])), l1=float(np.mean([x[1] for x in tl])),
                   pen=float(np.mean([x[2] for x in tl])), t=time.time() - t0)
        if not all(np.isfinite([x for t3 in tl for x in t3])):
            stop(f"training loss not finite at epoch {ep} ({model_name} {cell} {variant} SH)")
        if val is not None:
            yk = final_sh(model, T, D, std, D.ra[val], D.rb[val])
            mae_t = np.abs(yk - D.y[val]).mean(0) / std.sy                  # final prediction / SD(original y)
            rec["val"] = float(mae_t[F.I_PRIM].mean())
            if not np.isfinite(rec["val"]):
                stop(f"validation MAE not finite at epoch {ep} ({model_name} {cell} {variant} SH)")
            if rec["val"] < best["val"]:
                best = dict(val=rec["val"], epoch=ep, state=copy.deepcopy(model.state_dict()), per_target=mae_t.tolist())
        hist.append(rec)
        if log and (ep % log == 0 or ep == 1):
            print(f"    ep {ep} {rec}", flush=True)
        if val is not None and ep - best["epoch"] >= patience:
            break
    if val is not None and (best["state"] is None or best["epoch"] < 1):
        stop(f"no finite best epoch ({model_name} {cell} {variant} SH)")
    if val is not None:
        model.load_state_dict(best["state"])
    return dict(model=model, std=std, sy=std.sy, sr=std.sr, beta=std.beta, T=T, history=hist,
                best_epoch=best["epoch"] if val is not None else n_epochs, best_val=best["val"],
                best_val_per_target=best["per_target"], seconds=time.time() - t0)


def gate_m1_sh(model, D, std, rows, seed, n=F.ANTISYM_N):
    """gate M-1' (= fu2 train.gate_m1) on the SH final prediction incl. the baseline, float64."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(rows, size=min(n, len(rows)), replace=False)
    m64 = copy.deepcopy(model).double().eval()
    T64 = std.tensors(D, dtype=torch.float64)
    ra, rb = D.ra[pick], D.rb[pick]
    gab, gba, gaa = (final_sh(m64, T64, D, std, x, y, bs=32) for x, y in ((ra, rb), (rb, ra), (ra, ra)))
    anti, selfm = float(np.abs(gab + gba).max()), float(np.abs(gaa).max())
    return dict(n=len(pick), anti_max=anti, self_max=selfm, ok=bool(anti <= F.ANTISYM_TOL and selfm <= F.ANTISYM_TOL))


def arm_fns(arm):
    return (FT.fit, FT.gate_m1) if arm == "S" else (fit_sh, gate_m1_sh)


def gate_d0_train(gatef, r, D, rows, seed):
    """GB-5: gate M-1' (the arm's gate function, inputs only) on 200 random D0 training pairs of the fit; None when the
    fit has no D0 pairs (split T1)."""
    d0 = D.d0_rows(rows) if hasattr(D, "d0_rows") else np.zeros(0, dtype=int)
    if not len(d0):
        return None
    return gatef(r["model"], D, r["std"], d0, seed=seed)


def final_kcal(D, r, rows):
    """final prediction (kcal/mol, float64) of a fit for pair-table rows: S = FU2 (predict * sy), SH = baseline + residual."""
    if D.arm == "S":
        return FT.predict(r["model"], r["T"], D.ra[rows], D.rb[rows]) * r["sy"]
    return final_sh(r["model"], r["T"], D, r["std"], D.ra[rows], D.rb[rows])


def leak_check(D, std, fit_rows, others, test_rows):
    """L-2 / L-8 (spec 16 M): scaling / beta pairs = the fit's training pairs, disjoint from validation / test pairs; no
    test reaction among the scaling reactions."""
    used = set(np.asarray(std.pair_rows).tolist())
    if used != set(np.asarray(fit_rows).tolist()):
        stop("L-2 / L-8: scaling (and beta) pairs differ from the fit's training pairs")
    if D.arm == "SH" and set(np.asarray(std.beta_rows).tolist()) != used:
        stop("L-2: beta pairs differ from the fit's training pairs")
    for o in others:
        if used & set(np.asarray(o).tolist()):
            stop("L-2 / L-8: a validation / test pair is among the fit's scaling / beta pairs")
    trx = set(D.ra[test_rows].tolist()) | set(D.rb[test_rows].tolist())
    if trx & set(np.asarray(std.rxn_rows).tolist()):
        stop("L-8: a test reaction is among the fit's scaling reactions")
    out = dict(n_fit_pairs=len(used), n_scale_rxns=int(len(std.rxn_rows)), ok=True)
    if D.arm == "SH":
        out["beta_pairs_sha256"] = C.list_sha([(q,) for q in _ids(D, np.asarray(std.beta_rows))])
    return out


# ================================================================ gates / provenance
def _foreign_to(path, rungs):
    """True if inputs.check_paths puts `path` in another rung's spec 4-1 column for every rung of this process (MA-15)."""
    import inputs as IN
    return bool(rungs) and all(IN.check_paths(r, [path])["foreign"] for r in rungs)


def gate_add3(need, rungs=()):
    """spec 9 (= d1_split.gate_add1 / fu2 train.gate_add2); exit 3 otherwise. need = absolute input paths that
    prereg_d1pair_add3.json inputs_sha256 must list. Listed files of another rung's column only (e.g. the dft / g1
    tables in a g2 process) are not opened here (L-4, MA-15); their own rung's processes verify them."""
    def git(*a):
        return subprocess.run(["git", "-C", str(L.HERE), *a], capture_output=True, text=True)
    bad = []
    for p in (ADD3_MD, ADD3_JSON):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        elif git("ls-files", "--error-unmatch", str(p)).returncode != 0:
            bad.append(f"{p.name} not committed")
        elif git("diff", "--quiet", "HEAD", "--", str(p)).returncode != 0:
            bad.append(f"{p.name} modified since its commit")
    rec = {}
    if not bad:
        rec = json.loads(ADD3_JSON.read_text())
        if L.sha256(ADD3_MD) != rec.get("md_sha256"):
            bad.append(f"{ADD3_MD.name} sha256 differs from {ADD3_JSON.name} md_sha256")
        ins = rec.get("inputs_sha256") or {}
        lack = sorted(set(need) - set(ins))
        if lack:
            bad.append(f"{ADD3_JSON.name} inputs_sha256 lacks {lack}")
        skipped = []
        for path, s in ins.items():
            if path not in need and _foreign_to(path, rungs):
                skipped.append(path)
                continue
            if not Path(path).is_file() or L.sha256(path) != s:
                bad.append(f"input {path} changed or missing")
        _GATE["add3_not_opened_other_rungs"] = skipped
    if bad:
        print("PREREG ADD3 gate (spec 9: commit the addendum before any Phase M training):\n  " + "\n  ".join(bad),
              file=sys.stderr, flush=True)
        sys.exit(3)
    _GATE["add3"] = L.sha256(ADD3_MD)
    return dict(rec, git_head=git("rev-parse", "HEAD").stdout.strip())


def gate_p(rungs, labels):
    """MA-16 (review P-8): results/P_report.json all_pass, the rung tables and the label file have the sha256 that
    inputs.py pairs recorded (pairs accepts only current builds), so labels and H come from one consistent Phase P
    state; exit 2 while P_report is missing / running, 3 on a difference."""
    p = L.RES / "P_report.json"
    if not p.is_file():
        not_ready(f"{p} missing (inputs.py pairs)")
    rep = json.loads(p.read_text())
    if rep.get("running"):
        not_ready(f"{p} is being written")
    bad = [] if rep.get("all_pass") and not rep.get("stop") else [f"P_report all_pass {rep.get('all_pass')}"]
    for r in rungs:
        want = ((rep.get("rungs") or {}).get(r) or {}).get("table_sha256")
        if want is None or not inputs_path(r).is_file() or L.sha256(inputs_path(r)) != want:
            bad.append(f"{inputs_path(r)} sha256 != P_report rungs.{r}.table_sha256 ({want})")
    lp = str(_abs(labels))
    lab = (rep.get("labels") or {}).get(lp) or {}
    if not lab.get("ids_ok") or not Path(lp).is_file() or L.sha256(lp) != lab.get("sha256"):
        bad.append(f"label file {lp}: sha256 / ids differ from P_report labels ({lab.get('sha256')})")
    if bad:
        stop("Phase P state changed since results/P_report.json (rerun inputs.py build / pairs): " + "; ".join(bad))
    return dict(sha256=L.sha256(p), rungs=list(rungs), labels=lp)


def gate_c_stage2():
    """MA-17 (review C-1): stage 2 trains only after results/C_tests.json passed with the g2 rung tested."""
    c = json.loads((L.RES / "C_tests.json").read_text()) if (L.RES / "C_tests.json").is_file() else {}
    if not (c.get("all_pass") and c.get("g2_verified")):
        stop(f"stage 2 needs results/C_tests.json all_pass with g2_verified (rerun tests with the g2 table): "
             f"all_pass {c.get('all_pass')}, g2_verified {c.get('g2_verified')}")
    return dict(c_tests_sha256=L.sha256(L.RES / "C_tests.json"))


# ---------------------------------------------------------------- GEO2b stage 3 gates (GA-4 .. GA-7, GA-14, GB-2, GB-6)
def ladder_results_changed(rec):
    """GA-5: prereg_geo2b.json ladder_results_sha256 {repo-relative path: sha256} -> the paths whose file is missing or
    has another sha256 now (repo-relative keys: the snapshot does not depend on the worktree path, D1-G3 F-13)."""
    snap = rec.get("ladder_results_sha256")
    if not isinstance(snap, dict) or not snap:
        return [f"{PREREG_GEO2_JSON.name} has no ladder_results_sha256 snapshot"]
    bad = []
    for rel, s in sorted(snap.items()):
        p = L.REPO / rel
        if not p.is_file() or L.sha256(p) != s:
            bad.append(rel)
    return bad


def gate_prereg_geo2(need, rungs=()):
    """GEO2 spec 8-1 / 8-3 read for GEO2b (GA-4; = gate_add3 on the GEO2b files): analysis/geo2/results/PREREG_GEO2B.md
    and prereg_geo2b.json tracked by git and unchanged since HEAD, sha256(md) = json md_sha256, json rungs = ["geo"] and
    split = "T1D0", json inputs_sha256 {absolute path: sha256} contains `need` (the label file + $GEO2/inputs/geo.parquet)
    and every listed file still has its sha256 (files of other rungs' columns only are not opened, MA-15), and every
    LADDER result file of the json snapshot is unchanged (GA-5). Exit 3 otherwise."""
    def git(*a):
        return subprocess.run(["git", "-C", str(L.HERE), *a], capture_output=True, text=True)
    bad = []
    for p in (PREREG_GEO2_MD, PREREG_GEO2_JSON):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        elif git("ls-files", "--error-unmatch", str(p)).returncode != 0:
            bad.append(f"{p.name} not committed")
        elif git("diff", "--quiet", "HEAD", "--", str(p)).returncode != 0:
            bad.append(f"{p.name} modified since its commit")
    rec = {}
    if not bad:
        rec = json.loads(PREREG_GEO2_JSON.read_text())
        if L.sha256(PREREG_GEO2_MD) != rec.get("md_sha256"):
            bad.append(f"{PREREG_GEO2_MD.name} sha256 differs from {PREREG_GEO2_JSON.name} md_sha256")
        if list(rec.get("rungs") or []) != list(STAGE_RUNGS[GEO_STAGE]) or rec.get("split") != SPLIT_D0D1:
            bad.append(f"{PREREG_GEO2_JSON.name} rungs {rec.get('rungs')} / split {rec.get('split')} != "
                       f"{list(STAGE_RUNGS[GEO_STAGE])} / {SPLIT_D0D1}")
        ins = rec.get("inputs_sha256") or {}
        lack = sorted(set(need) - set(ins))
        if lack:
            bad.append(f"{PREREG_GEO2_JSON.name} inputs_sha256 lacks {lack}")
        skipped = []
        for path, s in ins.items():
            if path not in need and _foreign_to(path, rungs):
                skipped.append(path)
                continue
            if not Path(path).is_file() or L.sha256(path) != s:
                bad.append(f"input {path} changed or missing")
        _GATE["prereg_geo2_not_opened_other_rungs"] = skipped
        changed = ladder_results_changed(rec)
        if changed:
            bad.append(f"LADDER result files changed since the pre-registration (spec 14): {changed[:8]}")
    if bad:
        print("PREREG_GEO2B gate (GEO2 spec 8-3: commit PREREG_GEO2B.md + prereg_geo2b.json before any stage-3 training):"
              "\n  " + "\n  ".join(bad), file=sys.stderr, flush=True)
        sys.exit(3)
    _GATE["prereg_geo2"] = L.sha256(PREREG_GEO2_MD)
    return dict(rec, git_head=git("rev-parse", "HEAD").stdout.strip())


def gate_p_geo(rungs, labels):
    """GB-6 (the gate_p rule for the geo rung): analysis/geo2/results/P_build_geo.json all_pass, not stop, outputs sha256
    of $GEO2/inputs/geo.parquet = the file; P_geo_report.json (GEO2.md 7) stop false and made for that table (its
    inputs[<table>] = the sha256 now, GEO2b 연결 S-3). Exit 2 while a report is missing / being written / stale, 3 on a
    difference. The label file is checked by its own rule (GB-3, S_splits_d0d1.json)."""
    for p in (P_BUILD_GEO, P_GEO_REPORT):
        if not p.is_file():
            not_ready(f"{p} missing (GEO2b Phase P of the inputs role: GEO2.md 6-1 P / 7)")
    b = json.loads(P_BUILD_GEO.read_text())
    pr = json.loads(P_GEO_REPORT.read_text())
    if b.get("running") or pr.get("running"):
        not_ready(f"{P_BUILD_GEO.name} / {P_GEO_REPORT.name} is being written")
    bad = [] if b.get("all_pass") and not b.get("stop") else [f"P_build_geo all_pass {b.get('all_pass')}, stop "
                                                              f"{b.get('stop')}"]
    for r in rungs:
        t = inputs_path(r)
        want = (b.get("outputs") or {}).get(str(t))
        if want is None or not t.is_file() or L.sha256(t) != want:
            bad.append(f"{t} sha256 != {P_BUILD_GEO.name} outputs ({want})")
    stale = [str(t) for t in map(inputs_path, rungs) if t.is_file() and (pr.get("inputs") or {}).get(str(t)) != L.sha256(t)]
    if stale and not bad:
        not_ready(f"{P_GEO_REPORT.name} inputs do not record {stale} as it is now (stale: rerun feat_geo.py p_report)")
    if pr.get("stop"):
        bad.append(f"{P_GEO_REPORT.name} stop: {pr.get('stop_reasons') or pr.get('stop_reason')}")
    if bad:
        stop("GEO2b Phase P state differs from what training needs (rerun the inputs role's P step): " + "; ".join(bad))
    return dict(p_build_geo_sha256=L.sha256(P_BUILD_GEO), p_geo_report_sha256=L.sha256(P_GEO_REPORT),
                sha256=L.sha256(P_BUILD_GEO), rungs=list(rungs), labels=str(_abs(labels)))


def gate_splits_d0d1(labels):
    """GB-2: the stage-3 label file is the D0 + D1 file and S_splits_d0d1.json passes the d0d1_splits reader contract now
    (final, not preliminary; label + split files + D0 inputs unchanged). Exit 3 / 2 otherwise."""
    if _abs(labels) != _abs(DD.LABELS):
        stop(f"stage 3 trains on {DD.LABELS} only (GB-3), got --labels {labels}")
    rec, folds, files = load_folds(SPLIT_D0D1)
    return dict(s_splits_sha256=L.sha256(DD.S_SPLITS), preliminary=rec.get("preliminary"),
                n_d0_ok=(rec.get("d0") or {}).get("n_ok"), labels_sha256=(rec.get("labels") or {}).get("sha256"))


def gate_c_geo():
    """GA-7: stage 3 trains only after analysis/geo2/results/C_tests_geo.json passed (tests/run_tests.py geo).
    GA-14: and its code_sha256 of the trainer files (C_GEO_TRAINER_CODE keys) = the files now."""
    c = json.loads(C_TESTS_GEO.read_text()) if C_TESTS_GEO.is_file() else {}
    if not c.get("all_pass") or c.get("mode") not in (None, "geo"):
        stop(f"stage 3 needs {C_TESTS_GEO} all_pass in the full geo mode (tests/run_tests.py geo, GEO2 spec 8-2): "
             f"all_pass {c.get('all_pass')}, mode {c.get('mode')}")
    rec = c.get("code_sha256") if isinstance(c.get("code_sha256"), dict) else {}
    diff = [k for k, p in C_GEO_TRAINER_CODE.items() if rec.get(k) != L.sha256(p)]
    if diff:
        stop(f"{C_TESTS_GEO.name} code_sha256 differs from the code now for {diff}: the C tests did not run on the code "
             f"that would train (GA-14; rerun tests/run_tests.py geo gpu + cpu + merge)")
    return dict(c_tests_geo_sha256=L.sha256(C_TESTS_GEO), code_checked=list(C_GEO_TRAINER_CODE))


def guard_ladder_write(stage, what):
    """GA-13: the stage-1 / 2 commands smoke / run / collect and estimate write LADDER files (results/C_smoke_atoms.json,
    L_predictions.parquet, L_val.csv, L_report.json, M1_gpu_estimate.json, $LAD/smoke, $LAD/units). STOP (exit 3) before
    any write unless CODE_SHA = results/L_report.json code_sha256, the code that wrote the LADDER results (no such file =
    a fresh LADDER tree: the old behaviour). Stage 3 is not affected; `list` never calls this."""
    if int(stage) == GEO_STAGE:
        return
    ref = L.RES / "L_report.json"
    if not ref.is_file():
        return
    try:
        rec = json.loads(ref.read_text()).get("code_sha256")
    except (OSError, ValueError, AttributeError) as e:
        stop(f"{what} --stage {stage}: {ref} unreadable ({type(e).__name__}); refusing to write LADDER files (GA-13)")
    rec = rec if isinstance(rec, dict) else {}
    diff = sorted(k for k in set(rec) | set(CODE_SHA) if rec.get(k) != CODE_SHA.get(k))
    if diff:
        stop(f"{what} --stage {stage} refused: this code differs from the code of the LADDER results (results/"
             f"L_report.json code_sha256: {diff}); with the GEO2 additions the stage-1 / 2 commands other than list "
             f"would rewrite or invalidate LADDER files (GEO2 8-1 / 13 / 14; GA-13)")


def _check_root(root, smoke, rung=None):
    r = _abs(root)
    u = _abs(UNITS)
    inside = r == u or u in r.parents
    if smoke and inside:
        stop("--smoke never writes to $LAD/units")
    if inside and not _GATE.get("add3") and not (rung in GEO_INPUTS and _GATE.get("prereg_geo2")):
        stop("writing $LAD/units needs the ADD3 gate (train_atoms.py run)")
    if inside and rung in GEO_INPUTS and not _GATE.get("prereg_geo2"):          # GA-4: ADD3 does not open stage 3
        stop("writing geo units to $LAD/units needs the PREREG_GEO2B gate (train_atoms.py run --stage 3)")
    L.assert_path(r / "probe")
    return r


def fingerprint(D, u, sp, smoke=False):
    fp = _fingerprint_dict(D, u, sp, smoke)
    if D.rung in GEO_INPUTS:                    # GA-8: geo units also record the GEO2b pre-registration and D0D1 code
        fp["prereg_geo2b"] = None if smoke else (L.sha256(PREREG_GEO2_MD) if PREREG_GEO2_MD.is_file() else None)
        fp["d0d1_code"] = L.sha256(CODE_D0D1)
        fp["split"] = D.split_name
    return norm(fp)


def _fingerprint_dict(D, u, sp, smoke=False):
    sh = D.arm == "SH"
    return dict(
        code=CODE_VERSION, code_sha256=CODE_SHA, rung=D.rung, arm=D.arm, variant=VARIANT,
        inputs_table=D.sha["inputs_table"], labels=D.sha["labels"], pair_list=D.pair_list_sha, split_pairs=sp["pairs_sha"],
        s_splits=D.sha["s_splits"], t_report=D.sha["t_report"], t_manifest=D.sha["t_manifest"], q_report=D.sha["q_report"],
        add3=None if smoke else (L.sha256(ADD3_MD) if ADD3_MD.is_file() else None),
        grid=Mo.GRID[u["model"]], h_cols=D.hinfo["h_cols"] if sh else None,
        rules=dict(lam=F.LAMBDA, lr=F.LR, wd=F.WD, dropout=F.DROPOUT, batch=F.BATCH, max_epochs=F.MAX_EPOCHS,
                   patience=F.PATIENCE, seeds=F.FINAL_SEEDS, tune_seed=F.SEED, antisym_tol=F.ANTISYM_TOL,
                   antisym_n=F.ANTISYM_N, val="first inner fold (FU2 split files)",
                   sh=("beta = sum(dH dy) / sum(dH^2) no intercept on the fit pairs, c_ghost 0; residual / SD(train "
                       "residual); penalty and metric on the final prediction") if sh else None),
        **FT.versions())


def save_torch(path, obj):
    path = L.assert_path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ================================================================ one unit
def run_unit(D, u, root=UNITS, n_epochs=F.MAX_EPOCHS, seeds=F.FINAL_SEEDS, cells=None, log=50, smoke=False):
    """tune (grid cells on inner_train, early stopping on inner_val) -> final (seeds, best epoch, train) -> test
    predictions + gate M-1' for one unit (= fu2 train.run_unit + rung / arm / SH baseline / validation kcal MAE)."""
    root = _check_root(root, smoke, getattr(D, "rung", None))
    root.mkdir(parents=True, exist_ok=True)
    if (u["rung"], u["arm"]) != (D.rung, D.arm) or u["model"] not in MODELS:
        stop(f"unit {u} does not match the data ({D.rung}, {D.arm})")
    name = uname(u)
    sp = D.split(u["fold"])
    fp = fingerprint(D, u, sp, smoke)
    js = root / f"{name}.json"
    if js.is_file():
        old = json.loads(js.read_text())
        ofp = old.get("fingerprint") or {}
        if ofp != fp:
            stop(f"cache {js} has a different fingerprint: keys {sorted(k for k in set(fp) | set(ofp) if ofp.get(k) != fp.get(k))}")
        print(f"  {name}: cached", flush=True)
        return old
    t_unit = time.time()
    fitf, gatef = arm_fns(D.arm)
    sh = D.arm == "SH"
    full = Mo.GRID[u["model"]]
    idx = list(range(len(full))) if cells is None else [int(c) for c in cells]
    tr, te, itr, iva = sp["train"], sp["test"], sp["inner_train"], sp["inner_val"]
    if min(len(tr), len(te), len(itr), len(iva)) == 0:
        stop(f"{name}: empty split list (train {len(tr)}, test {len(te)}, inner {len(itr)} / {len(iva)})")
    tune = []
    for ci, gi in enumerate(idx):
        cell = full[gi]
        r = fitf(D, u["model"], cell, VARIANT, itr, iva, seed=F.SEED, n_epochs=n_epochs, log=log)
        lk = leak_check(D, r["std"], itr, (iva, te), te)
        g = gatef(r["model"], D, r["std"], te, seed=F.SEED + ci)            # inputs only, no labels (M-1)
        gd = gate_d0_train(gatef, r, D, itr, F.SEED + D0_GATE_SEED + ci)    # GB-5 (None for T1)
        vk = np.abs(final_kcal(D, r, iva) - D.y[iva]).mean(0)
        tune.append(dict(cell=cell, grid_index=gi, best_epoch=r["best_epoch"], val=r["best_val"],
                         val_per_target=r["best_val_per_target"], val_kcal_mae=dict(zip(TARGETS, vk.tolist())),
                         epochs_run=len(r["history"]), seconds=r["seconds"], gate_M1=g, sy=r["sy"].tolist(),
                         sr=r["sr"].tolist() if sh else None, beta=r["beta"].tolist() if sh else None, leak=lk))
        if gd is not None:
            tune[-1]["gate_M1_d0_train"] = gd
        print(f"  {name} cell {cell}: val {r['best_val']:.4f} at epoch {r['best_epoch']} ({r['seconds']:.0f} s) "
              f"gate {g}" + (f" gate D0 train {gd}" if gd is not None else ""), flush=True)
        if not g["ok"]:
            stop(f"gate M-1' failed ({name}, tuning cell {cell}): {g}")
        if gd is not None and not gd["ok"]:
            stop(f"gate M-1' failed on D0 training pairs ({name}, tuning cell {cell}): {gd}")
        del r
    best = min(range(len(tune)), key=lambda i: (tune[i]["val"], i))
    cell, ep = tune[best]["cell"], tune[best]["best_epoch"]
    if not (isinstance(ep, int) and ep >= 1 and np.isfinite(tune[best]["val"])):
        stop(f"{name}: chosen cell {cell} has best_epoch {ep} / validation {tune[best]['val']}")
    preds, finals, betas = [], [], []
    for s in seeds:
        r = fitf(D, u["model"], cell, VARIANT, tr, None, seed=s, n_epochs=ep, log=log)
        lk = leak_check(D, r["std"], tr, (te,), te)
        g = gatef(r["model"], D, r["std"], te, seed=s)
        gd = gate_d0_train(gatef, r, D, tr, s + D0_GATE_SEED)              # GB-5 (None for T1)
        if not g["ok"]:
            stop(f"gate M-1' failed ({name}, seed {s}): {g}")
        if gd is not None and not gd["ok"]:
            stop(f"gate M-1' failed on D0 training pairs ({name}, seed {s}): {gd}")
        preds.append(final_kcal(D, r, te))
        betas.append(r["beta"] if sh else np.zeros(len(TARGETS)))
        finals.append(dict(seed=s, epochs=ep, seconds=r["seconds"], gate_M1=g, sy=r["sy"].tolist(),
                           sr=r["sr"].tolist() if sh else None, beta=r["beta"].tolist() if sh else None, leak=lk,
                           history_tail=r["history"][-3:]))
        if gd is not None:
            finals[-1]["gate_M1_d0_train"] = gd
        st = dict(mu=r["std"].mu, sd=r["std"].sd, gmu=r["std"].gmu, gsd=r["std"].gsd, sy=r["sy"])
        if sh:
            st.update(sr=r["sr"], beta=r["beta"])
        save_torch(root / f"{name}__s{s - F.SEED}.pt", dict(state=r["model"].state_dict(), cell=cell, unit=u, std=st,
                                                            rung=D.rung, arm=D.arm, variant=VARIANT))
        del r
    if any(not np.array_equal(b, betas[0]) for b in betas):
        stop(f"{name}: beta differs between final seeds (same training pairs)")
    beta = np.asarray(betas[0], float)
    base = beta * D.dH[te] if sh else np.zeros((len(te), len(TARGETS)))
    Y = np.stack(preds)                                                    # [seeds, n_test, 10] kcal/mol, final
    pid = D.P.index.to_numpy()[te].astype(str)
    rows = [pd.DataFrame(dict(pair_id=pid, ref_kind=D.P.ref_kind.to_numpy()[te], pos=D.P.pos.to_numpy()[te],
                              fold=int(u["fold"]), rung=u["rung"], arm=u["arm"], model=u["model"], target=t,
                              y=D.y[te, j], yhat=Y[:, :, j].mean(0), base=base[:, j], beta=float(beta[j]),
                              yhat_seed_sd=Y[:, :, j].std(0))) for j, t in enumerate(TARGETS)]
    pred = pd.concat(rows, ignore_index=True)[PRED_COLS]
    pq = root / f"{name}.parquet"
    L.write_atomic(pq, lambda t: pred.to_parquet(t, index=False))
    rec = dict(unit=u, name=name, fingerprint=fp, smoke=bool(smoke), variant=VARIANT, tune=tune,
               chosen=dict(cell=cell, best_epoch=ep, index=best, grid_index=tune[best]["grid_index"]), final=finals,
               val_kcal_mae=tune[best]["val_kcal_mae"], val_pairs_sha256=sp["val_sha"], test_pairs_sha256=sp["test_sha"],
               n_val=len(iva), beta=beta.tolist(), n_train=len(tr), n_test=len(te), n_inner_train=len(itr),
               n_inner_val=len(iva), dropped=sp["dropped"], parquet_sha256=L.sha256(pq), inputs=D.inputs,
               inputs_table=D.hinfo, labels=D.linfo, opened_inputs=opened_summary(D.opened),
               wall_seconds=time.time() - t_unit, host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
    if "n_d0" in sp:                                                       # GB-4 (T1D0 only)
        rec.update(split=D.split_name, n_d0=sp["n_d0"])
    write_json(js, rec)
    return rec


def gates_ok(r):
    """every gate M-1' of a unit json ok (test pairs; GB-5 D0 training pairs where recorded)."""
    fits = list(r.get("tune") or []) + list(r.get("final") or [])
    return bool(fits) and all(f["gate_M1"]["ok"] for f in fits) and \
        all(f["gate_M1_d0_train"]["ok"] for f in fits if f.get("gate_M1_d0_train") is not None)


# ================================================================ commands
def smoke_estimate_geo(out):
    """GA-9 / GB-9 (GEO2 spec 8-4 with the GEO2b cap of 160 GPU-h; STOP above it is judged by prereg_geo2b.py): per
    arm x model and reference rung r in (g1, dft), the LADDER r atom units' wall hours (results/L_report.json, 5 folds;
    the full tuning + 5 final seeds with early stopping) x the ratio of the geo smoke tuning epoch seconds to the stage-1
    smoke r ones (results/C_smoke_atoms.json; mean of the smallest and the largest cell; GPU types may differ, both are
    recorded). Projection = the larger of the two totals (dft trained longest in LADDER). Also the hours of the busiest
    slice of `run i n --stage 3` for n = 1 .. 10 (the [i::n] run order; per-unit hours = the arm x model projection / 5).
    Read only; never changes the smoke verdict."""
    try:
        sm = json.loads((L.RES / "C_smoke_atoms.json").read_text())
        lr = json.loads((L.RES / "L_report.json").read_text())
        rows, tot = {}, {}
        for ref in ("g1", "dft"):
            tot[ref] = 0.0
            for arm in ARMS:
                for m in MODELS:
                    g = out["configs"][f"L__geo__{arm}__{m}__f0"]["tune_epoch_seconds"]
                    b = sm["configs"][f"L__{ref}__{arm}__{m}__f0"]["tune_epoch_seconds"]
                    eg = 0.5 * (float(g["smallest"]) + float(g["largest"]))
                    eb = 0.5 * (float(b["smallest"]) + float(b["largest"]))
                    hb = sum(float(lr["units"][f"L__{ref}__{arm}__{m}__f{k}"]["wall_seconds"]) for k in range(NF)) / 3600
                    rows.setdefault(f"{arm}__{m}", {})[ref] = dict(ladder_hours=hb, epoch_s_geo=eg, epoch_s_ref=eb,
                                                                   ratio=eg / eb, geo_hours=hb * eg / eb)
                    tot[ref] += hb * eg / eb
        use = max(tot, key=tot.get)
        unit_h = {k: v[use]["geo_hours"] / NF for k, v in rows.items()}
        us = unit_list(GEO_STAGE)
        slices = {str(n): float(max(sum(unit_h[f"{u['arm']}__{u['model']}"] for u in us[i::n]) for i in range(n)))
                  for n in range(1, 11)}
        rung_h = {}
        for r in ("dft", "g1", "g2"):
            rung_h[r] = float(sum(float(v.get("wall_seconds") or 0.0) for k, v in lr["units"].items()
                                  if k.startswith(f"L__{r}__") and "__XGB__" not in k) / 3600)
        h = float(tot[use])
        return dict(ok=True, geo_gpu_hours=h, by_reference=tot, reference_used=use, limit_gpu_hours=GEO_LIMIT_GPU_H,
                    over_limit=bool(h > GEO_LIMIT_GPU_H), per_arm_model=rows, busiest_slice_hours=slices,
                    ladder_rung_atom_gpu_hours=rung_h, gpu_geo_smoke=(out.get("versions") or {}).get("gpu"),
                    gpu_ladder_smoke=(sm.get("versions") or {}).get("gpu"),
                    rule="max over reference rungs (g1, dft) of sum over arm x model of LADDER unit wall hours x (geo / "
                         "reference smoke tuning epoch seconds); cap 160 GPU-h (GEO1b DECISIONS D0D1)",
                    sources=dict(c_smoke_atoms=str(L.RES / "C_smoke_atoms.json"),
                                 l_report=str(L.RES / "L_report.json")))
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as e:
        return dict(ok=False, why=f"{type(e).__name__}: {e}", limit_gpu_hours=GEO_LIMIT_GPU_H)


def cmd_smoke(stage, labels):
    """spec 8 smoke (FU2 smoke rules per config) -> results/C_smoke_atoms.json; writes only $LAD/smoke.
    Stage 3 (GA-3 / GA-9): results/C_smoke_atoms_geo.json, scratch $LAD/smoke_geo, + 'estimate_geo'."""
    guard_ladder_write(stage, "smoke")                 # GA-13: before the scratch folder is removed
    FT.require_gpu()
    smoke_dir = SMOKE_GEO if int(stage) == GEO_STAGE else SMOKE
    L.assert_path(smoke_dir / "probe")
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    out = dict(versions=FT.versions(), device=str(FT.DEV), labels=str(_abs(labels)), stage=int(stage), configs={},
               code=CODE_VERSION, code_sha256=CODE_SHA)
    ok = True
    for rung in STAGE_RUNGS[int(stage)]:
        for arm in ARMS:
            D = get_data(rung, arm, labels)
            for m in MODELS:
                u = dict(rung=rung, arm=arm, model=m, fold=0)
                t0 = time.time()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                rec = run_unit(D, u, root=smoke_dir, n_epochs=2, seeds=F.FINAL_SEEDS[:1],
                               cells=[0, len(Mo.GRID[m]) - 1], log=1, smoke=True)
                g_ok = gates_ok(rec)                     # test pairs + (GB-5) D0 training pairs where present
                ok &= g_ok
                eps = [t["seconds"] / max(1, t["epochs_run"]) for t in rec["tune"]]
                n_par = [sum(p.numel() for p in Mo.build(m, Mo.GRID[m][c], D.d_in, len(TARGETS), D.d_glob,
                                                         VARIANT).parameters()) for c in (0, len(Mo.GRID[m]) - 1)]
                out["configs"][uname(u)] = dict(
                    n_params=dict(smallest=n_par[0], largest=n_par[1]), seconds=time.time() - t0,
                    tune_epoch_seconds=dict(smallest=eps[0], largest=eps[-1]),
                    gpu_mem_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
                    gate_M1_tune=[t["gate_M1"] for t in rec["tune"]], gate_M1_final=rec["final"][0]["gate_M1"],
                    val=[t["val"] for t in rec["tune"]], val_kcal_mae=rec["val_kcal_mae"], n_train=rec["n_train"],
                    n_test=rec["n_test"], n_inner_train=rec["n_inner_train"], n_inner_val=rec["n_inner_val"],
                    dropped=rec["dropped"], beta=rec["beta"], d_glob=int(D.d_glob), pair_list_n=len(D.pair_list),
                    opened_dirs=rec["opened_inputs"]["dirs"], ok=g_ok)
                if rec.get("n_d0") is not None:          # GB-4 / GB-5 (T1D0 only)
                    out["configs"][uname(u)].update(
                        n_d0=rec["n_d0"], nmax_atoms=int(D.nmax),
                        gate_M1_d0_train_tune=[t.get("gate_M1_d0_train") for t in rec["tune"]],
                        gate_M1_d0_train_final=rec["final"][0].get("gate_M1_d0_train"))
                print(f"[smoke] {uname(u)}: ok {g_ok} {out['configs'][uname(u)]['seconds']:.0f} s", flush=True)
    # upper bound without early stopping (FU2 smoke formula): 4 tuning cells (middle ~ mean of the two measured) on the
    # inner-training pairs + 5 final seeds on all training pairs (x 4/3) with the largest cell, 300 epochs, x 5 folds
    ub = 0.0
    for c in out["configs"].values():
        s, lg = c["tune_epoch_seconds"]["smallest"], c["tune_epoch_seconds"]["largest"]
        ub += (s + lg + (s + lg)) * F.MAX_EPOCHS + 5 * lg * 4 / 3 * F.MAX_EPOCHS
    out["stage_upper_bound_gpu_hours"] = float(ub * NF / 3600)
    if int(stage) == GEO_STAGE:
        out["estimate_geo"] = smoke_estimate_geo(out)
    out["ok"] = bool(ok)
    write_json(res_out("C_smoke_atoms.json", stage), out)
    return 0 if ok else 3


def cmd_run(i, n, stage, labels):
    guard_ladder_write(stage, "run")                   # GA-13
    FT.require_gpu()
    i = os.environ.get("SLURM_ARRAY_TASK_ID", "") if str(i) == "A" else str(i)
    if not (i.isdigit() and str(n).isdigit() and 0 <= int(i) < int(n)):
        print(f"usage: run <i|A> <n> with 0 <= i < n (got i={i!r}, n={n!r}; '@I' is replaced by jobs/gpu.slurm)",
              file=sys.stderr)
        return 2
    i = int(i)
    us = unit_list(stage)[i::int(n)]
    if not us:
        print(f"[M] slice {i} of {n}: no units", flush=True)
        return 0
    rs = sorted({u["rung"] for u in us})
    need = {str(_abs(labels))} | {str(_abs(inputs_path(r))) for r in rs}
    if int(stage) == GEO_STAGE:      # GEO2 8-1 (GA-4 .. GA-7, GB-2, GB-6): PREREG_GEO2B, P, D0D1 splits, C_tests_geo
        if set(rs) - set(STAGE_RUNGS[GEO_STAGE]):
            stop(f"stage 3 holds the geo rung only, got {rs}")
        pre = gate_prereg_geo2(need, rs)
        gp = gate_p_geo(rs, labels)
        gs = gate_splits_d0d1(labels)
        gc = gate_c_geo()
        print(f"[M] PREREG_GEO2B ok (git {pre['git_head']}, md {_GATE['prereg_geo2'][:12]}), GEO2b P ok "
              f"({gp['p_build_geo_sha256'][:12]}), D0D1 splits ok ({gs['s_splits_sha256'][:12]}, {gs['n_d0_ok']} D0 "
              f"pairs), C_tests_geo ok ({gc['c_tests_geo_sha256'][:12]}), {FT.versions()}, slice {i}/{n}: "
              f"{len(us)} units", flush=True)
    else:
        add3 = gate_add3(need, rs)
        gp = gate_p(rs, labels)
        if int(stage) == 2:
            gate_c_stage2()
        print(f"[M] ADD3 ok (git {add3['git_head']}), P_report ok ({gp['sha256'][:12]}), {FT.versions()}, slice {i}/{n}: "
              f"{len(us)} units", flush=True)
    for u in us:
        t0 = time.time()
        D = get_data(u["rung"], u["arm"], labels)
        rec = run_unit(D, u, root=UNITS)
        print(f"[M] {uname(u)} done ({time.time() - t0:.0f} s) chosen {rec['chosen']}", flush=True)
    return 0


def _no_gpu(fp):
    return {k: v for k, v in (fp or {}).items() if k != "gpu"}


# fingerprint keys that train_xgb.py computes from the same files with the same rule (MA-14)
XGB_SHARED_KEYS = ("code", "rung", "arm", "inputs_table", "labels", "pair_list", "split_pairs", "s_splits", "t_report",
                   "t_manifest", "q_report", "add3", "h_cols", "prereg_geo2b", "d0d1_code", "split")  # last 3: geo (GA-8)


def check_xgb_report(xn, labels, stage=1):
    """MA-14 (review M-4): results/L_xgb_report.json of `train_xgb.py check` (which recomputes the full XGB fingerprints in
    env_espley) passed on the same labels and train_xgb.py, and no XGB unit json / parquet changed since.
    Stage 3: results/L_xgb_report_geo.json of `train_xgb.py check --stage 3` (GA-3)."""
    p = res_out("L_xgb_report.json", stage)
    if not p.is_file():
        not_ready(f"{p} missing (run train_xgb.py check before collect --with-xgb)")
    r = json.loads(p.read_text())
    bad = [] if r.get("ok") else [f"ok {r.get('ok')}: {(r.get('bad') or [])[:3]}"]
    if r.get("labels") != str(_abs(labels)):
        bad.append(f"labels {r.get('labels')} != {_abs(labels)}")
    if (r.get("code_sha256") or {}).get("train_xgb.py") != L.sha256(HERE / "train_xgb.py"):
        bad.append("train_xgb.py changed since the check")
    units = r.get("units") or {}
    for x in xn:
        e = units.get(x) or {}
        if e.get("json_sha256") != L.sha256(UNITS / f"{x}.json") or \
                e.get("parquet_sha256") != L.sha256(UNITS / f"{x}.parquet"):
            bad.append(f"{x}: not in the check or changed since")
    if bad:
        stop("XGB units not verified by a current train_xgb.py check: " + "; ".join(bad[:6]))
    return dict(file=str(p), sha256=L.sha256(p), n_units=len(xn))


def cmd_collect(stage, with_xgb, labels):
    """atom units (+ XGB units) of the rungs up to the stage -> results/L_predictions.parquet, L_val.csv, L_report.json.
    Stage 3 (GEO2 8-1, GA-3 / GA-5): the geo units only -> results/L_predictions_geo.parquet, L_val_geo.csv,
    L_report_geo.json; the LADDER result files of the PREREG_GEO2B snapshot must be unchanged."""
    geo = int(stage) == GEO_STAGE
    guard_ladder_write(stage, "collect")               # GA-13
    rungs = collect_rungs(stage)
    if geo:
        if not PREREG_GEO2_JSON.is_file():
            stop(f"{PREREG_GEO2_JSON} missing: stage-3 units cannot exist without the pre-registration")
        changed = ladder_results_changed(json.loads(PREREG_GEO2_JSON.read_text()))
        if changed:
            stop(f"LADDER result files changed since the GEO2b pre-registration (spec 14): {changed[:8]}")
    au = unit_list(rungs=rungs)
    names = [uname(u) for u in au]
    xn = [f"L__{r}__{a}__XGB__f{k}" for r in rungs for a in ARMS for k in range(NF)] if with_xgb else []
    missing = [x for x in names + xn if not ((UNITS / f"{x}.json").is_file() and (UNITS / f"{x}.parquet").is_file())]
    if missing:
        print(f"GATE: {len(missing)} units missing: {missing[:8]}", file=sys.stderr)
        return 2
    recs = {x: json.loads((UNITS / f"{x}.json").read_text()) for x in names + xn}
    for x, r in recs.items():
        if L.sha256(UNITS / f"{x}.parquet") != r.get("parquet_sha256"):
            stop(f"{x}.parquet changed after its json")
        if not isinstance(r.get("val_kcal_mae"), dict) or set(r["val_kcal_mae"]) != set(TARGETS):
            stop(f"{x}: val_kcal_mae missing or not per target")
    for x in names:
        r = recs[x]
        if r.get("smoke") or not gates_ok(r):            # GB-5: incl. the D0 training-pair gates of geo units
            stop(f"gate M-1' failed (or smoke unit) in {x}")
    xgb_check = None
    for u in au:                                     # every unit must carry the CURRENT fingerprint (FU2 collect rule)
        D = get_data(u["rung"], u["arm"], labels)
        now = fingerprint(D, u, D.split(u["fold"]))
        old = recs[uname(u)]["fingerprint"]
        if _no_gpu(old) != _no_gpu(now):
            stop(f"{uname(u)} was computed with other code / inputs: keys "
                 f"{sorted(k for k in set(old) | set(now) if k != 'gpu' and old.get(k) != now.get(k))}")
        if with_xgb:                                 # MA-14 (review M-4): the XGB unit of the same rung x arm x fold
            x = f"L__{u['rung']}__{u['arm']}__XGB__f{u['fold']}"
            xf = recs[x].get("fingerprint") or {}
            diff = [k for k in XGB_SHARED_KEYS if xf.get(k) != now.get(k)]
            if diff:
                stop(f"{x} was computed with other inputs than the atom units / the current files: keys {diff}")
    if with_xgb:
        xgb_check = check_xgb_report(xn, labels, stage)
    parts = []
    for x in names + xn:
        q = pd.read_parquet(UNITS / f"{x}.parquet")
        if not set(PRED_COLS) <= set(q.columns):
            stop(f"{x}.parquet lacks columns {sorted(set(PRED_COLS) - set(q.columns))}")
        parts.append(q[PRED_COLS])
    pred = pd.concat(parts, ignore_index=True)
    # spec 16 E: the models / arms of a rung and fold share one test-pair list; L-9: one validation list per rung x fold
    bad_te, bad_va = [], []
    for r in rungs:
        for k in range(NF):
            g = pred[(pred.rung == r) & (pred.fold == k) & (pred.target == TARGETS[0])]
            sets = {(a, m): frozenset(s.pair_id) for (a, m), s in g.groupby(["arm", "model"])}
            if len(set(sets.values())) != 1:
                bad_te.append(f"{r} f{k}")
            vs = {recs[x].get("val_pairs_sha256") for x in names + xn if f"L__{r}__" in x and x.endswith(f"__f{k}")}
            if len(vs) != 1:
                bad_va.append(f"{r} f{k}")
    if bad_te or bad_va:
        stop(f"test-pair lists differ between models / arms: {bad_te}; validation-pair lists differ (L-9): {bad_va}")
    if pred.duplicated(["pair_id", "rung", "arm", "model", "target"]).any():
        stop("duplicated prediction rows (a pair in two test folds?)")
    if geo and set(pred.rung) != set(rungs):
        stop(f"stage-3 predictions hold rungs {sorted(set(pred.rung))}, not {list(rungs)} only")
    out = res_out("L_predictions.parquet", stage)
    L.write_atomic(out, lambda t: pred.to_parquet(t, index=False))
    vrows = []
    for x in names + xn:
        r = recs[x]
        _, rung, arm, model, f = x.split("__")
        for t in TARGETS:
            vrows.append(dict(rung=rung, arm=arm, model=model, fold=int(f[1:]), target=t,
                              val_kcal_mae=float(r["val_kcal_mae"][t]), n_val=r.get("n_val"),
                              val_pairs_sha256=r.get("val_pairs_sha256")))
    V = pd.DataFrame(vrows)
    vout = res_out("L_val.csv", stage)
    L.write_atomic(vout, lambda t: V.to_csv(t, index=False))
    rep = dict(stage=int(stage), rungs=list(rungs), models=list(MODELS) + (["XGB"] if with_xgb else []),
               n_units=len(names) + len(xn), labels=str(_abs(labels)), code=CODE_VERSION, code_sha256=CODE_SHA,
               units={x: dict(chosen=recs[x].get("chosen"), n_train=recs[x].get("n_train"), n_test=recs[x].get("n_test"),
                              n_val=recs[x].get("n_val"), dropped=recs[x].get("dropped"), beta=recs[x].get("beta"),
                              val_kcal_mae=recs[x]["val_kcal_mae"], gpu=(recs[x].get("fingerprint") or {}).get("gpu"),
                              wall_seconds=recs[x].get("wall_seconds"),
                              gate_M1_max=dict(anti=max([t["gate_M1"]["anti_max"] for t in recs[x].get("tune", [])]
                                                        + [f["gate_M1"]["anti_max"] for f in recs[x].get("final", [])]
                                                        or [None]) if x in names else None))
                      for x in names + xn},
               gpu_hours_atoms=float(sum(recs[x].get("wall_seconds") or 0 for x in names) / 3600),
               xgb_check=xgb_check, predictions=str(out), predictions_sha256=L.sha256(out), val_csv=str(vout), val_csv_sha256=L.sha256(vout))
    if geo:
        d0 = pred.ref_kind.eq("d0")
        rep["geo2"] = dict(prereg_geo2b_md_sha256=L.sha256(PREREG_GEO2_MD) if PREREG_GEO2_MD.is_file() else None,
                           ladder_results_unchanged=True, spec="docs/specs/GEO2.md 8-1 (GEO2b: D0D1, split T1D0)",
                           split=SPLIT_D0D1, s_splits_d0d1_sha256=L.sha256(DD.S_SPLITS) if Path(DD.S_SPLITS).is_file()
                           else None,
                           n_test_pairs=dict(d1=int(pred.loc[~d0 & (pred.target == TARGETS[0])].pair_id.nunique()),
                                             d0=int(pred.loc[d0 & (pred.target == TARGETS[0])].pair_id.nunique())),
                           note="D0 test pairs: ref_kind / pos 'd0', pair_id D#####; report D1 and D0 separately")
    write_json(res_out("L_report.json", stage), rep)
    print(f"[collect] {rep['n_units']} units -> {out.name} ({len(pred)} rows)", flush=True)
    return 0


def cmd_estimate():
    """GPU hours of stage 1 from the 30 FU2 Phase M unit records (variant A+G; tuning + final fit seconds)."""
    guard_ladder_write(1, "estimate")                  # GA-13: writes results/M1_gpu_estimate.json (a LADDER file)
    recs = []
    for p in sorted(F.UNITS.glob("T1__*__f*.json")):
        r = json.loads(p.read_text())
        u = r["unit"]
        recs.append(dict(name=r["name"], model=u["model"], variant=u["variant"], fold=int(u["fold"]),
                         gpu=(r.get("fingerprint") or {}).get("gpu"), n_train=r.get("n_train"),
                         seconds=float(sum(t["seconds"] for t in r["tune"]) + sum(f["seconds"] for f in r["final"]))))
    df = pd.DataFrame(recs)
    if df.empty:
        print(f"no FU2 unit records under {F.UNITS}", file=sys.stderr)
        return 2
    ag = df[df.variant == VARIANT]
    per = {m: dict(n=int(len(g)), mean_h=float(g.seconds.mean() / 3600), max_h=float(g.seconds.max() / 3600),
                   by_gpu={k: float(v / 3600) for k, v in g.groupby("gpu").seconds.mean().items()})
           for m, g in ag.groupby("model")}
    mean_h = {m: v["mean_h"] for m, v in per.items()}
    rung_arm = float(sum(NF * mean_h[m] for m in MODELS))
    us = unit_list(1)
    slices = {str(n): [float(sum(mean_h[u["model"]] for u in us[i::n])) for i in range(n)] for n in (1, 2, 3, 4, 6)}
    rep = dict(source=str(F.UNITS), n_records=int(len(df)), variant=VARIANT, per_model_unit_hours=per,
               per_rung_arm_hours=rung_arm, n_rung_arms_stage1=len(STAGE_RUNGS[1]) * len(ARMS),
               stage1_gpu_hours=rung_arm * len(STAGE_RUNGS[1]) * len(ARMS), stage1_trainings=len(us) * 9,
               stage2_gpu_hours=rung_arm * len(STAGE_RUNGS[2]) * len(ARMS),
               slice_hours=slices, note="FU2 A+G wall seconds inside fit (tuning 4 cells + 5 final seeds); data loading "
                                        "and SH's 9 extra global columns not included")
    write_json(L.RES / "M1_gpu_estimate.json", rep)
    print(json.dumps(rep, indent=1), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], add_help=True)
    ap.add_argument("cmd", choices=("list", "smoke", "run", "collect", "estimate"))
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--labels", default=None)                 # GB-3: default = stage_labels(stage)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--with-xgb", action="store_true")
    a = ap.parse_args(argv)
    a.labels = str(stage_labels(a.stage)) if a.labels is None else a.labels
    if a.cmd == "list":
        print("\n".join(uname(u) for u in unit_list(a.stage)))
        return 0
    if a.cmd == "smoke" or (a.cmd == "run" and a.smoke):
        return cmd_smoke(a.stage, a.labels)
    if a.cmd == "run":
        if len(a.rest) != 2:
            print("usage: train_atoms.py run <i|A> <n> [--stage 1|2] [--labels P]  |  run --smoke", file=sys.stderr)
            return 2
        return cmd_run(a.rest[0], a.rest[1], a.stage, a.labels)
    if a.cmd == "collect":
        return cmd_collect(a.stage, a.with_xgb, a.labels)
    if a.cmd == "estimate":
        return cmd_estimate()
    return 2


if __name__ == "__main__":
    sys.exit(main())
