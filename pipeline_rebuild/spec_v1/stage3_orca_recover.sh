#!/bin/bash
# ORCA-based recovery of the 98 xTB-failed reactions. 8-way sharded.

#SBATCH --job-name=st3_orca
#SBATCH --array=0-15%9
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/st3_orca_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/st3_orca_%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export PATH=/home1/yeseo1ee/orca_6_1_1_avx2:$PATH
# ORCA MPI env — required for %pal nprocs > 1
export LD_LIBRARY_PATH=/home1/yeseo1ee/orca_6_1_1_avx2:${LD_LIBRARY_PATH:-}

python -u pipeline_rebuild/spec_v1/stage3_orca_recover.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards 16 --ncpu 1
