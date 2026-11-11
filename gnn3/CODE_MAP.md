# CODE_MAP — where every piece of GNN3 came from

GNN3 = the CH3M pair model of rung `lf3`, model `GNN` (cross-validation units `D__lf3__S__GNN__f{0..4}` in
`/gpfs/tmp_cpu2/yeseo1ee/ch3m/units`). The published numbers were produced by
`analysis/ch3m/train_m.py run GNN 3 @I 5` (array of 5 folds, `analysis/ch3m/jobs/gpu_m.slurm`, RTX 3090) and
evaluated by `analysis/ch3m/eval_m.py run` / `analysis/abs3/eval_abs.py run`. This package re-implements the functions
those runs actually called, with readable names, and calls the input-pipeline code unchanged.
Line numbers refer to the files at branch point `04642c1c` (abs3 HEAD). Equivalence: `results/E_equivalence.json`.

## 1. Function map

### 1.1 Data (`gnn3/data.py`, `gnn3/channels.py`)

| original file:function (lines) | purpose | new location / name |
|---|---|---|
| analysis/ch3/ch3_common.py: `M_ROWS` (43), `M` (52-55) | 10 labels → strain / inter / solv / barrier, 10×4 0/1 matrix | `channels.channel_matrix`, `channels.labels_to_channels` |
| analysis/ch3/data_ch3.py: `Ch3Data.__init__` (84-94) | `y = y10 @ M` on top of the deploy data | `data.PairDataset.__init__` (`targets_kcal`, `labels_kcal`) |
| analysis/ch3/data_ch3.py: `use_targets`, `MODE3` (57-77) | switch fu2 index constants to 3-channel mode | fixed constants `BARRIER_INDEX`, `CHANNEL_INDICES`, `VALIDATION_INDICES` |
| analysis/deploy/data_lf.py: `fill_token_arrays` (96-136) | token files → padded arrays (one-hot, flags, numeric, positions, edges, fragments, 182 globals) | `data.fill_token_arrays` |
| data_lf.py: `load_tokens` (139-190) | token build report checks (versions, columns, gates), manifest order, T-a flags | `data.ReactionTokens.from_manifest` |
| data_lf.py: `load_x182` (193-225) | 182-column table: same columns, finite rows, equal to the token copy (1e-12 rel.) | `data.check_global_table` |
| data_lf.py: `label_provenance` (237-247), `load_labels` (250-266) | label file sha256 = validated one; pair table | `data.load_pair_labels` |
| data_lf.py: `load_folds` (269-281) | 5 T1 split files, sha256 checked | `data.load_folds` |
| data_lf.py: `list_sha` (91-92) → d1pair_common.py: `list_sha` (235-237) | sha256 of an id list | `data.list_sha256` |
| data_lf.py: `DeployData.__init__` (290-340) | usable reactions ("lf" rule), usable pairs, pair rows, pair list | `data.PairDataset.__init__` |
| data_lf.py: `DeployData.split` (361-389), `_ids` (353-354) | fold lists (usable pairs only) + leak checks + list hashes | `data.PairDataset.fold_split`, `PairDataset.pair_ids`, `data.FoldSplit` |
| data_lf.py: `DeployData._split_all` (391-410) | all `in_T_main` pairs for the final model | `data.PairDataset.all_training_pairs` |
| data_lf.py: `DeployData.rxn_rows` (356-358) | reactions of a pair set | `data.PairDataset.reactions_of` |
| data_lf.py: `DeployData.layout` (343-346) | token layout stored with every model | `data.TokenLayout.as_saved` / `from_saved` |
| analysis/d1_pair/fu2/train.py: `Data.static` (110-121) | fit-independent tensors, cached | `data.ReactionTokens.static_tensors` |
| fu2/train.py: `Std.__init__` (143-153) | standardisation from the training reactions / pairs only | `data.Standardizer.from_training_pairs` |
| fu2/train.py: `Std.tensors` (155-160) | standardised model inputs | `data.Standardizer.tensors` |
| analysis/deploy/train_tf.py: `std_dict` (139-140), `std_sha` (149-155) | saved statistics, their hash | `Standardizer.as_saved`, `Standardizer.from_saved`, `Standardizer.sha256` |
| fu2/train.py: `DEV` (37) | CUDA if visible | `data.default_device` |
| analysis/deploy/predict_tf.py: `InputSet` (134-149) | token arrays of arbitrary (new) reactions | `data.ReactionTokens.from_files` |

### 1.2 Model (`gnn3/model.py`)

| original | purpose | new |
|---|---|---|
| analysis/d1_pair/fu2/models.py: `rbf` (29-33) | Gaussian radial basis of distances | `radial_basis` |
| models.py: `pdist` (41-44) | distance matrix | `pairwise_distances` |
| models.py: `mlp` (47-52) | Linear-SiLU-Dropout stack | `make_mlp` |
| models.py: `masked_sum` (55-56) | sum over real atoms | `masked_sum` |
| models.py: `AttnPool` (64-74) (`w`) | attention pooling | `AttentionPool` (`score`) |
| models.py: `MPLayer` (134-152) (`etype, edge, w_h, w_e, msg_out, upd, norm`) | one message-passing step | `MessagePassingLayer` (`edge_type_embedding, edge_distance_mlp, node_to_message, edge_to_message, message_output, node_update, layer_norm`) |
| models.py: `GNN` (155-179) (`embed, layers, pool, gproj, head`; `readout`) | encoder + antisymmetric head | `GNNPairModel` (`atom_embedding, message_layers, attention_pool, global_projection, pair_head`; `encode_reactions`) |
| models.py: `build` (283-284) | model of one grid cell, variant A+G | `build_model` |
| fu2/train.py: `batch` (163-170) | unique reactions of a pair batch, trimmed | `make_reaction_batch` |
| fu2/train.py: `predict` (185-193) | eval-mode prediction in batches of 256 | `predict_standardized` |
| (new) | load original weight names | `ORIGINAL_KEY_PREFIXES`, `ORIGINAL_LAYER_PARTS`, `rename_original_key`, `load_weights` |

Module creation order is unchanged, so the same seed gives the same initial weights (`tests/test_units.py`
`test_original_weights_same_seed_same_numbers`, all 4 grid cells).

### 1.3 Training (`gnn3/training.py`)

| original | purpose | new |
|---|---|---|
| fu2/train.py: cuDNN flags (38-39) | deterministic kernels | module top of `training.py` |
| fu2/train.py: `seed_all` (196-199) | numpy / torch / CUDA seeds | `seed_everything` |
| fu2/train.py: `orient` (173-175) | (a, b) → (b, a) with sign | `swap_pair_order` |
| fu2/train.py: `epoch_batches` (207-216) | shuffled, randomly flipped batches | `iterate_training_batches` |
| fu2/train.py: `loss_fn` (178-182) | L1 + 0.1 × identity penalty | `identity_penalty_loss` |
| fu2/train.py: `fit` (219-267) | one fit with early stopping | `train_one_fit` → `FitResult` |
| fu2/train.py: `gate_m1` (270-279) | float64 antisymmetry check (C-4) | `check_antisymmetry` |
| fu2/train.py: `versions` (283-286) | software / GPU versions | `environment` |
| train_tf.py: `kcal` (158-160) | prediction × target scale | `predictions_kcal` |
| train_tf.py: `leak_check` (163-178) | C-5 leak check | `check_no_leak` |
| analysis/ch3m/train_m.py: `run_unit` (189-274) | tune 4 cells, pick, 5 final seeds, predict test pairs | `run_cross_validation_unit` |
| train_m.py: `fingerprint` (166-176) | cache key of a unit | `UnitSettings.as_dict` |
| train_m.py: `pt_payload` (141-147), `save_torch` (100-110) | model file content, atomic save | `save_model_file` |
| train_m.py: `write_json` (84-85) | atomic json | `training.write_json` |
| train_m.py: `choose_final_grid` (538-555) | final cell (mode) and epochs (median) | `choose_final_setting` |
| train_m.py: `train_final` (558-627) | final 5-seed model on all training pairs | `train_final_model` (not run, D-9) |
| train_m.py: `main`, `cmd_run`, `unit_list`, `uname` (630-646, 408-426, 88-93) | command line, unit naming | `scripts/train_cv.py`, `scripts/train_final.py` |

### 1.4 Prediction (`gnn3/predict.py`, `scripts/predict.py`)

| original | purpose | new |
|---|---|---|
| analysis/deploy/predict_tf.py: `load_models` (106-130), `std_object` (98-103) | load model files with their statistics | `load_network`, `load_networks` |
| predict_tf.py: `check_reactions` (152-212) | input checks of new reactions | `check_new_reactions` |
| predict_tf.py: `predict_all` (216-223) | every network's ΔΔ | `predict_with_networks` |
| predict_tf.py: `run_predict` (299-381), direct read-out | seed mean / SD, out-of-range flag, excluded pairs | `summarize_networks`, `predict_pairs`, `scripts/predict.py` |
| analysis/ch3m/predict_m.py: `_thresholds` (53-64) → EM_ad.json `GNN3` | applicability thresholds | `APPLICABILITY_THRESHOLDS_KCAL` |

### 1.5 Evaluation (`gnn3/metrics.py`, `scripts/evaluate.py`)

| original | purpose | new |
|---|---|---|
| analysis/ch3/eval_ch3.py: `labels` (112-122) | 4-target truth + attributes | `channel_truth` |
| eval_ch3.py: `s_folds` (125-130) | evaluation attributes | `pair_sets` |
| eval_ch3.py: `assemble` (173-195) | long frame | `assemble` |
| eval_ch3.py: `set_mask` (198-208), `wide` (211-220) | set masks, one row per pair | `set_mask`, `wide` |
| eval_ch3.py: `pearson` (242-246), `sign_acc` (249-252) | r, sign accuracy | `pearson`, `sign_accuracy` |
| analysis/geo2/eval_geo2.py: `boot_diff` (41-54) | core bootstrap | `bootstrap_mae_difference` |
| eval_ch3.py: `metrics` (300-322) | MAE / RMSE / r / sign table | `error_table` |
| eval_ch3.py: `rank_families` (325-335), `rank_table` (338-377) | within-family barrier ranking | `_ranking_families`, `ranking_table` |
| eval_ch3.py: `dominant` (380-382), `ctype` (385-392), `agree` (395-416) | cause labels and agreement | `dominant_channel`, `cause_type`, `_agreement` |
| eval_ch3.py: `cause` (419-433), `cause_rows` (436-448) | cause agreement table | `cause_agreement`, `cause_table` |
| eval_ch3.py: `identity_residual` (459-464) | barrier − channel sum of predictions | `identity_residual` |
| analysis/ch3m/eval_m.py: `ad` (210-217) | 95th percentile of the seed spread | `applicability_thresholds` |
| eval_m.py: `direct4` (95-105) | GNN3 rows of KM_predictions | `scripts/evaluate.read_predictions` |
| (new) | MAE / R² per position and skeleton | `breakdown_table` |

### 1.6 Inputs (`gnn3/inputs/pipeline.py`, `scripts/prepare_inputs.py`, `jobs/prepare_inputs.sh`)

The input code is **called, not copied** (spec 3). The wrappers fix the order, arguments and environment.

| original (called unchanged) | stage | wrapper |
|---|---|---|
| analysis/deploy/g2_deploy.py `pool` (175-212) → `cmd_one` (128-172): autodE `stage_ts`, `stage_stereo`, `stage_ref`, then `finalize_lf` | ts-search (SMILES → TS) | `STAGES["ts-search"]` |
| analysis/deploy/finalize_lf.py `run` (276-304) → `finalize_lf` (138-224); `report` (428-524) | finalize (ORCA XTB2 ALPB(water) Freq at the TS) | `STAGES["finalize"]`; `reuse_finalized` copies finished folders |
| analysis/deploy/meta_lf.py `run` (127-199) | metadata | `STAGES["metadata"]` |
| analysis/deploy/feat_lf.py `slice` (164-251), `merge` (338-391) → espley_xtb_repro `xtb_slice.process_one`, `ext_features` blocks B1-B6, `aggregate.derive` | features (182 globals) | `STAGES["features"]` |
| analysis/deploy/tokens_lf.py `atoms` / `gate` / `tokens` (178-232) → d1_pair/fu2 `atoms.py`, `tokens.py build_one` (57-81) | tokens (29 wide) | `STAGES["tokens"]` |
| analysis/deploy/i3_chain.py: `i3_env` (175-181) | child environment inside one root | `root_environment` (port) |
| i3_chain.py: `check_env` (218-229) | job environment check | `check_interpreter` (port) |
| i3_chain.py: `classify_T` (321-338) | subset element rule of the tokens check | called unchanged by `_tokens_subset_exception` |

## 2. Constants

All in `gnn3/constants.py` (each with a comment naming its source).

| new name | value | unit | original |
|---|---|---|---|
| `LABEL_CHANNELS` | d1, d2, elst, pauli, oi, disp, cpcm, cds, c_ghost, barrier | kcal/mol | ch3_common.TARGETS10 (39) |
| `TARGETS` | strain, inter, solv, barrier | kcal/mol | ch3_common.TARGETS3 (40) |
| `CHANNEL_GROUPS` | strain = d1+d2; inter = elst+pauli+oi+disp+c_ghost; solv = cpcm+cds | — | ch3_common.GROUPS (41-42) |
| `BARRIER_INDEX`, `CHANNEL_INDICES`, `VALIDATION_INDICES` | 3; [0,1,2]; [0,1,2,3] | — | data_ch3.MODE3 (58) = fu2_common.I_BAR / I_CH9 / I_PRIM after `use_targets` |
| `IDENTITY_PENALTY_WEIGHT` | 0.1 | — | fu2_common.LAMBDA (40) |
| `LEARNING_RATE`, `WEIGHT_DECAY` | 1e-3, 1e-4 | — | fu2_common.LR, WD (41) |
| `DROPOUT` | 0.1 | — | fu2_common.DROPOUT (41) |
| `BATCH_SIZE_PAIRS` | 64 | pairs | fu2_common.BATCH (41) |
| `MAX_EPOCHS`, `PATIENCE_EPOCHS` | 300, 30 | epochs | fu2_common.MAX_EPOCHS, PATIENCE (41) |
| `BASE_SEED`, `FINAL_SEEDS` | 20261019; +0..4 | — | fu2_common.SEED (30) = d1pair_common.SEED (79); FINAL_SEEDS (42) |
| `SCALE_EPSILON` | 1e-12 | — | d1pair_common.SCALE_EPS (105) |
| `GRID` | n_layers {4, 6} × width {64, 128} | — | models.GRID["GNN"] (278) |
| `N_RADIAL_BASIS`, `RADIAL_BASIS_MAX_DISTANCE_ANGSTROM` | 32, 4.0 | Å | models.GNN defaults (156) |
| `N_EDGE_TYPES` | 3 (none, covalent, forming) | — | models.MPLayer (141) |
| `ATTENTION_MASK_VALUE` | −1e9 | — | models.NEG (26) |
| `VARIANT` | "A+G" | — | train_tf.VARIANT (66) |
| `PREDICT_BATCH_PAIRS`, `ANTISYMMETRY_BATCH_PAIRS` | 256, 32 | pairs | fu2/train.predict, gate_m1 |
| `ANTISYMMETRY_CHECK_PAIRS`, `ANTISYMMETRY_TOLERANCE_KCAL` | 200, 1e-4 | pairs, kcal/mol | fu2_common.ANTISYM_N, ANTISYM_TOL (43-44) |
| `ELEMENTS` | 1, 6, 7, 8, 9, 14, 16, 17, 35 | Z | dep_common.ELEMENTS (40) |
| `BINARY_COLUMNS`, `NUMERIC_COLUMNS` | 6 flags; 11 + 3 numeric | — | fu2/tokens.BIN_COLS, NUM_BASE, NUM_EXT (35-37) |
| `ATOM_TOKEN_WIDTH`, `N_GLOBAL_FEATURES` | 29, 182 | — | data_lf.D_IN (70); T_report glob_cols |
| `TOKEN_VERSION`, `ATOM_VERSION` | fu2-tokens-3-lf, fu2-atoms-3-lf | — | dep_common (41-42) |
| `GLOBAL_TABLE_RELATIVE_TOLERANCE` | 1e-12 | relative | data_lf.GLOB_REL_TOL (71) |
| `N_FOLDS`, `FOLD_LISTS` | 5; train, test, inner_train, inner_val | — | data_lf.NF, FOLD_KEYS (63-64) |
| `SIGN_THRESHOLD_KCAL`, `CELL_MIN_PAIRS`, `RANK_MIN_POINTS` | 1.0, 4, 8 | kcal/mol, pairs, points | eval_ch3.SIGN_MIN, CELL_MIN, RANK_MIN (46-48) |
| `RANK_BINS`, `POSITIONS` | [0.41,1), [1,2), [2,∞); all, p, m, o, direct | kcal/mol | eval_ch3 (49, 42) |
| `BOOTSTRAP_RESAMPLES`, `BOOTSTRAP_SEED` | 2000, 20261019 | — | eval_geo2.N_BOOT, SEED (35) |
| `APPLICABILITY_PERCENTILE` | 95 | % | eval_ch3.AD_PCT (50) |
| `predict.APPLICABILITY_THRESHOLDS_KCAL` | strain 0.98614, inter 0.58853, solv 0.50680, barrier 0.90211 | kcal/mol | analysis/ch3m/results/EM_ad.json `GNN3` |

## 3. Not ported, and why

| original | why not |
|---|---|
| train_m.py `gate_prereg`, `gate_c_tests`, `_check_root`, `mode_check`, `_outdim`, `_limit`, `CODE_SHA` | project bookkeeping of CH3M (pre-registration, write guards, test-gate, code hashes, smoke limits); the checks that protect the numbers (leak, antisymmetry, settings fingerprint, sha256 of every input) are ported |
| train_m.py `cmd_smoke`, `lad_unit`, `cmd_cr`, `load_units`, `cmd_collect`, `model_card`, `cmd_final`, `final_name` | smoke / regression / collection / Phase W steps of CH3M; not on the GNN3 result path |
| data_ch3.py `_bind_env`, `use_targets`, `current_mode`, `MODE10` | the environment and mode switches existed only to reuse the 10-target fu2 code; GNN3 is always 3-channel |
| fu2/train.py `require_gpu`, `Data.__init__`, `Data.split`, `gate_add2`, `run_unit`, `cmd_*` | env guard (README states the environment instead); the FU2 data class and Phase M driver are replaced by DeployData |
| fu2/models.py `AW`, `CFConv`, `TF`, `BiasedMHA`, `TFBlock`, `cosine_cutoff`, `masked_mean` | other model families |
| data_lf.py source `g2`, `_ladder_usable`, `input_columns`, `get_data` | C-2 / C-3 tests of the deploy project |
| predict_tf.py `graph_readout`, `_graph_nodes`, `write_outputs`, `cmd_run`, `cmd_check_i1/i2`; predict_m.py `check-i1`, `check-i2` | the TF / graph read-out and the self-tests of the TF10 final model (GNN3 has no final model, D-9) |
| eval_ch3.py `boot_rel`, `compare`, `adoption`, `lf_sum`, `ch3_frame`; eval_m.py `sum4`, `q1`, `q2`, `cmp_rows`, `cancel`; eval_abs.py `q_a`, `q_c`, `q_d` | comparisons of GNN3 against other models; they do not change any GNN3 number |
| ch3m_common / dep_common write helpers, `assert_write_ok` | replaced by plain atomic writes |
| the whole input pipeline (finalize_lf, meta_lf, feat_lf, tokens_lf, fu2 atoms / tokens, espley_xtb_repro, D1 pilot code) | called unchanged, by the spec; copying it would risk numbers |

## 4. External dependencies

| item | version / path | used by |
|---|---|---|
| env_d1fu2 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2`: Python 3.11.17, torch 2.2.1+cu118, CUDA 11.8, cuDNN 8.7.0 (8700), numpy 1.26.0, pandas 2.1.1, scipy 1.16.3, scikit-learn 1.3.1, pyarrow | training, prediction, evaluation, tests |
| GPU | NVIDIA GeForce RTX 3090 (partition gpu1) | bit-identical training (E-3) |
| D1 run env | `/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh`: Python 3.9.5 venv, **autodE 1.4.5**, RDKit 2024.3.2, **xtb 6.7.1**, **ORCA 6.1.1** (`$HOME/orca_6_1_1_avx2`), OpenMPI 4.1.8f (no XPMEM), `OMPI_MCA_pml=ob1`, `OMPI_MCA_btl=self,vader,tcp`, `OMPI_MCA_coll_hcoll_enable=0`, `UCX_TLS=tcp,self,sm` | ts-search, finalize, metadata |
| ORCA job header | `! XTB2 ALPB(water) Freq`, `%pal nprocs 4`, `%maxcore 3000` (g2_autode.FREQ_HEADER, 62) | finalize |
| venv5 | `/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/venv/bin/python`: Python 3.10.14, numpy 1.26.4, tblite 0.7.0, xtb 6.7.1 (edcfbbe) at `/home1/yeseo1ee/xtb-dist/bin/xtb`, dftd3, morfeus; Si patch `analysis/d1_pair/site_si/sitecustomize.py` | features, tokens |
| `$PILOT` | `/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/D1_build/d1_autode_pilot/` | ts-search, finalize |
| `$PILOT/orca_direct.py` | `write_inp` (121-123), `orca_bin` (127-128), `run` (131-139), `terminated` (143-144), `last_fspe` (151-153), `parse_hess` (164-183), `imag_and_mode` (186-192) | finalize |
| `$PILOT/run_ts.py` | `forming_share` (64-71), `run_autode` (88-128), `imag_gate` (183-197), `analyse` (201-279), `StageFail` (179-180) | finalize (forming_share), ts-search |
| `$PILOT/pilot_common.py` | `load_job` (31-37), `ade_name` (40-43), `job_dir` (56-59), `import_repo_modules` (66-74), `done` (78-79), `failed` (82-84), `mark_done` (87-89), `mark_fail` (92-94), `write_json` (97-100), `_jsonable` (103-116), `read_xyz` (119-127), `write_xyz` (130-132), `n_cores` (139-140), `configure_autode` (144-189) | ts-search, finalize |
| `$PILOT/make_reference.py`, `stereo_utils.py` | `coley_port` (148-268), `dipolarophile_check` (287-326); stereo checks (rdkit) | ts-search |
| D1-G2 folders | `/gpfs/tmp_cpu2/yeseo1ee/d1_g2/jobs/<job_id>/` (ts.xyz, ts_result.json, ref_*.xyz, ref_result.json, stereo_result.json), `g2_final/<rxn_id>/` | finalize (source of every D1 TS) |
| published inputs | `/gpfs/tmp_cpu2/yeseo1ee/ch3/lf/{X182_lf.parquet, fu2_lf/tokens/}` + reports `analysis/ch3/results/lf_inputs/` | training data |
| labels / splits | `/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/F_pairs_cr2.csv` (validated by `analysis/d1_ladder/results/V_report.json`), `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/splits/T1__f*.json` (`analysis/d1_pair/results/fu2/S_splits.json`), `analysis/d1_pair/results/S_folds.csv` | training, evaluation |

## 5. Problems found (발견한 문제) — recorded, not fixed

The original behaviour is ported unchanged; none of these changes a published number.

| # | where | problem |
|---|---|---|
| P-1 | analysis/deploy/finalize_lf.py `load_targets` (228-239), `load_manifest` (242-248), `cmd_run` (280-281); meta_lf.py `cmd_run` (134, 142) | The input chain after the TS search only accepts the 9,266 D1-G2 targets (fixed-size `G_status.csv`; `finalize_lf.py run` and `meta_lf.py` also need the D1 manifest row, manifest sha256 fixed). A reaction outside that set can get a TS (`g2_deploy.py`) but stops at the first command of the finalize stage (`finalize_lf.py run`) without a code change; `report` also needs the D1-G2 `g2_final` folder (gate L-2). |
| P-2 | g2_deploy.py (deployment TS search) | It finalises in a node-local work folder that is deleted afterwards, so `meta_lf.build` (reads `result.json source_dir`) would STOP for every deployed reaction; the deploy I-3 chain had to use its own fallback (`i3_chain.build_i3`). |
| P-3 | analysis/deploy/i3_chain.py `check_root` (147-152) | The only end-to-end new-reaction chain accepts roots under `/gpfs/tmp_cpu2/yeseo1ee/deploy/i3*` only; that folder was deleted (2026-11-09). |
| P-4 | analysis/d1_g2/jobs/g2.slurm (18) | Hard-codes `D1G2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_g2`: a new `g2_autode.py` run writes into the production D1-G2 folder. |
| P-5 | finalize_lf.py (ORCA XTB2 Freq) | Not bit-reproducible on rerun: of the 3 Hessian features, `nu_imag` moves by ~1e-7 (E-5 rerun 4.8e-8; CH3 V check: measured up to 7.4e-7, upper bound 8.6e-7), `mode_share` / `mode_async` by ~1e-11 (E-5 rerun 3.1e-11). The inputs can be regenerated exactly from the g2 folders only when the finalize folders are reused (E-5). |
| P-6 | analysis/deploy/predict_tf.py (module globals `MODEL="TF"`, 10 `TARGETS`); analysis/ch3m/predict_m.py (no `run`) | No command line in the existing code predicts new reactions with GNN3; only an in-process `PM.bind("GNN", 3)` works. `scripts/predict.py` fills this gap. |
| P-7 | analysis/ch3m/final_m.py `main` (41) | Writes `results/WM_gate.json` before `cmd_final` checks the GPU and the EM choice: `final_m.py GNN3` would overwrite the TF10 gate file and then STOP. |
| P-8 | train_m.py `train_final` (615-625), predict_m.py `_merge` (74-79) | `MODEL_CARD.json`, `MODEL_CARD_ch3m.json`, `WM_report.json` and the `IM_selftest.json` keys carry no candidate name, so a GNN3 final would overwrite TF10's; `check-i2 GNN3` would read the TF10 card and `_thresholds` would return TF10 thresholds. |
| P-9 | predict_m.py import → data_ch3 `_bind_env` (34-45) | Importing it STOPs unless `DEP_SCRATCH` = `$CH3/lf` and `DEP_RESULTS` = `analysis/ch3/results/lf_inputs` (each unset or resolving to that path; an already imported dep_common must point there too), so it cannot run inside another input root. |
| P-10 | train_m.py `choose_final_grid` (538-555) | No odd-number-of-units guard (train_tf.choose_final has one); harmless with 5 units. |
| P-11 | fu2/train.py `require_gpu` (293-304) | The message suggests `--gres=gpu:a6000:1` on gpu4 / gpu5, but every GNN3 run used gpu1 RTX 3090. |
| P-12 | d1pair_common.py `ANTISYM_TOL = 1e-6` (106) vs fu2_common.py `ANTISYM_TOL = 1e-4` (43) | Two constants with the same name and different values. The GNN3 path (fu2/train.py `gate_m1` 279, train_m.py 174) uses only fu2_common's 1e-4; d1pair_common's 1e-6 is used by the d1_pair / XGB gates (d1_train.py 560, d1_followup.py 286, d1_ladder/train_xgb.py 1039, ch3m/xgb_m.py 201). |
| P-13 | fu2/train.py `fit` (230) | `CosineAnnealingLR(T_max=300)` also in the final fits that run only `best_epoch` (38-76) epochs, so the learning rate never decays to its minimum there. Behaviour kept. |
| P-14 | eval_ch3.py `rank_families` (330) | `drop_duplicates("Xb")` silently keeps the lexicographically first pair_id when a family has two pairs with the same substituent. Kept. |
| P-15 | docs/specs/GNN3_CLEAN.md (167) | "cyclooctyne D R² 0.13" has no committed source. Recomputed here (`results/eval/breakdown.csv`): it is the barrier R² of skeleton D on all test pairs (0.129, n 541); on the P_main reference pairs it is 0.027 (n 458). The model card gives both. |
| P-16 | analysis/ch3m/results/MODEL_CARD_ch3m.json | Describes the TF10 final model, not GNN3; GNN3 has no final model (only the 25 cross-validation networks). |
