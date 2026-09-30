#!/bin/bash
#SBATCH --job-name=r4_feat_agg
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
# Phase 2: merge the 18 slices of GEOM (dft -> xtb_features_g0.parquet, g1 -> xtb_features_g1.parquet); gates in
# aggregate.py (exit != 0 = gate failed, nothing written). An existing output is skipped while all 18 slices exist
# and none is newer; otherwise it is moved to <out>.stale and rebuilt.
#   sbatch --partition=cpu1,cpu2 --export=ALL,GEOM=dft --output=$R4_SCRATCH/logs/feat_agg_dft.%j.out r4_feat_aggregate.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
GEOM=${GEOM:-dft}
python aggregate.py --geom "$GEOM"
