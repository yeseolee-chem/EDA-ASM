#!/bin/bash
#SBATCH --job-name=r4_feat_all
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
# Phase 2 in one queue slot: the 18 slices of GEOM (g1 default, or g2; xtb_slice.py, 1 core each, $SLURM_CPUS_PER_TASK
# at a time), then aggregate.py. Same scripts as the array route (r4_feat_array.sh / r4_feat_aggregate.sh); exit code
# = the aggregate gate's.
#   sbatch --partition=cpu1,cpu2 --export=ALL,GEOM=g1 --output=$R4_SCRATCH/logs/feat_all_g1.%j.out r4_feat_all.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export GEOM=${GEOM:-g1}
seq 0 8 | xargs -P "${SLURM_CPUS_PER_TASK:-1}" -I{} env SLURM_ARRAY_TASK_ID={} bash "$SLURM_SUBMIT_DIR/r4_feat_array.sh"
bash "$SLURM_SUBMIT_DIR/r4_feat_aggregate.sh" || { echo "### aggregate gate failed"; exit 1; }
