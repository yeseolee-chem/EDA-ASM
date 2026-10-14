#!/bin/bash
#SBATCH --job-name=r4_feat_agg
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
# Phase 2: merge the 18 slices of GEOM (g1 default -> xtb_features_g1.parquet; g2 likewise); gates in aggregate.py
# (exit != 0 = gate failed, nothing written). An existing output is skipped while all 18 slices exist and none is
# newer; otherwise it is moved to <out>.stale and rebuilt.
#   sbatch --partition=cpu1,cpu2 --export=ALL,GEOM=g1 --output=$R4_SCRATCH/logs/feat_agg_g1.%j.out r4_feat_aggregate.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
GEOM=${GEOM:-g1}
python aggregate.py --geom "$GEOM"
