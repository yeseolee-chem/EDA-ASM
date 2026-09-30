#!/bin/bash
#SBATCH --job-name=r4_feat_all
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
# Phase 2 in one queue slot: the 18 slices of GEOM (xtb_slice.py, 1 core each, $SLURM_CPUS_PER_TASK at a time), then
# aggregate.py, then (GEOM=dft) the 1e-8 verification against the OLD rev 3 parquet. Same scripts as the array route
# (r4_feat_array.sh / r4_feat_aggregate.sh / r4_feat_verify.sh); exit code of the last step is the job's.
#   sbatch --partition=cpu1 --export=ALL,GEOM=dft --output=$R4_SCRATCH/logs/feat_all_dft.%j.out r4_feat_all.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export GEOM=${GEOM:-dft}
seq 0 8 | xargs -P "${SLURM_CPUS_PER_TASK:-1}" -I{} env SLURM_ARRAY_TASK_ID={} bash "$SLURM_SUBMIT_DIR/r4_feat_array.sh"
bash "$SLURM_SUBMIT_DIR/r4_feat_aggregate.sh" || { echo "### aggregate gate failed"; exit 1; }
if [ "$GEOM" = dft ]; then bash "$SLURM_SUBMIT_DIR/r4_feat_verify.sh"; exit $?; fi
