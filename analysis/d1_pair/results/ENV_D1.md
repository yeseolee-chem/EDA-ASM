# ENV_D1 — environment of the D1PAIR G / F jobs (§7-2)

Every `os.environ` read found in `g1_geom.py`, `xtb_slice.py`, `ext_features.py`, `aggregate.py`, `r6_partition.py`, `r6_recover.py`, `rev5_common.py`, `rev6_common.py`, `fragmenter.py`, `train_ml_single.py`, `r6_features.py`, `r6_nn.py` and the D1 snapshot modules (`run_ts.py`, `orca_direct.py`, `rmsd.py`, ...), with the value the D1 jobs use (`jobs/g_env.sh` exports the G / F values; the F0 / M jobs do not import the G / F modules' paths).
Write locations are checked against the rev 4-6 scratch / results folders (`d1pair_common.FORBIDDEN_WRITE`) by `check_env` before any G / F computation.

| variable | read at | D1PAIR value |
|---|---|---|
| `D1P_CONFIG` | pilot_common.py:26 | `**not set by D1PAIR** (module default)` |
| `D1P_MANIFEST` | pilot_common.py:32 | `**not set by D1PAIR** (module default)` |
| `D1P_RDKIT_WARNINGS` | stereo_utils.py:36 | `**not set by D1PAIR** (module default)` |
| `D1_SNAPSHOT` | g1_geom.py:47 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/d1_snapshot_d573111e` |
| `ESPLEY_ARMS` | train_ml_single.py:74 | `unused` |
| `ESPLEY_EXT` | train_ml_single.py:75 | `unused` |
| `ESPLEY_FEAT` | train_ml_single.py:64 | `unused (train_ml_single imported for GRIDS / FEATURE_SETS only)` |
| `ESPLEY_GEOM` | train_ml_single.py:68 | `unused` |
| `ESPLEY_LABELS` | g1_geom.py:55, xtb_slice.py:90, rev5_common.py:27 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/G/labels_d1_g.json` |
| `ESPLEY_META` | g1_geom.py:58, xtb_slice.py:96, rev5_common.py:25 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/G/input_meta_d1.csv` |
| `ESPLEY_ML_OUT` | train_ml_single.py:65 | `unused` |
| `ESPLEY_NJOBS` | train_ml_single.py:71 | `unused` |
| `ESPLEY_NSLICES` | aggregate.py:22 | `unused (aggregate.main not run)` |
| `ESPLEY_OUT` | aggregate.py:20, rev5_common.py:22, train_ml_single.py:63 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/xtb` |
| `ESPLEY_PARTITION` | xtb_slice.py:99 | `label` |
| `ESPLEY_PROF` | g1_geom.py:56, r6_partition.py:21, rev5_common.py:24 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/prof` |
| `ESPLEY_PROTOCOLS` | train_ml_single.py:70 | `unused` |
| `ESPLEY_REPO_DATA` | rev5_common.py:26 | `read-only Espley repository data (rev5_common.ESP_DATA); never read by D1PAIR code` |
| `ESPLEY_ROWS` | train_ml_single.py:67 | `unused` |
| `ESPLEY_SMILES` | g1_geom.py:57 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/G/smiles_d1.csv` |
| `ESPLEY_TARGET_SET` | train_ml_single.py:430 | `unused` |
| `G1_ROOT` | g1_geom.py:59, xtb_slice.py:94, rev5_common.py:23, r6_features.py:69 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/g1` |
| `G1_ROOT_R6` | rev6_common.py:33 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/g1_rec` |
| `G1_ROOT_REV4` | r6_recover.py:40 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/g1` |
| `G1_WORKERS` | g1_geom.py:379 | `unset (cpus / 2)` |
| `G2_ROOT` | xtb_slice.py:95 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/g2_unused` |
| `ORCA_BIN` | g1_geom.py:60, orca_direct.py:128 | `binary, $HOME/orca_6_1_1_avx2/orca (same as rev 4)` |
| `R4_SCRATCH` | rev6_common.py:24 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/r4` |
| `R5_SCRATCH` | rev5_common.py:21 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/r5` |
| `R6_B1_TOL` | ext_features.py:167 | `unset (rev 5 B1 gate 1e-6 for every rxn)` |
| `R6_ONLY` | r6_recover.py:525 | `unset` |
| `R6_SCRATCH` | rev6_common.py:22 | `/gpfs/tmp_cpu2/yeseo1ee/d1_pair/r6` |
| `R6_SKIP` | r6_recover.py:531 | `unset` |
| `R6_WORKERS` | r6_recover.py:536 | `unset (cpus / 2)` |
| `SLURM_CPUS_PER_TASK` | g1_geom.py:356, g1_geom.py:379, ext_features.py:1181, r6_recover.py:536 … | `SLURM` |
| `TF_CPP_MIN_LOG_LEVEL` | r6_nn.py:21 | `1 (r6_nn default)` |
| `VAL_QUEUE` | orca_direct.py:42 | `**not set by D1PAIR** (module default)` |
| `XTB_BIN` | g1_geom.py:61, xtb_slice.py:91 | `binary, xtb 6.7.1 (same as rev 4)` |
| `XTB_THREADS` | xtb_slice.py:132, ext_features.py:245 | `1` |

Variables without a D1PAIR value: ['D1P_CONFIG', 'D1P_MANIFEST', 'D1P_RDKIT_WARNINGS', 'VAL_QUEUE'].

Set by `jobs/g_env.sh` for D1PAIR itself (not read by the rev 4-6 modules):
- `PYTHONPATH=analysis/d1_pair/site_si`, `D1PAIR_ESP_DIR=analysis/espley_xtb_repro`: Si added to fragmenter.COV (1.11 A) / Z (14), rev5_common.BONDI (2.10 A) / GFN2_VALENCE (4) (user 2026-10-19); `check_env` exits 3 unless the four Si entries are active.
- `PYTHONDONTWRITEBYTECODE=1`; `D1PAIR_SCRATCH` = the scratch root; the ORCA / OpenMPI / xtb variables of rev 4 `r4_env.sh` unchanged.
