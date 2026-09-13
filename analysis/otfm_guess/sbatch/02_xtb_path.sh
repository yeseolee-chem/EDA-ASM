#!/bin/bash
# xTB reaction-path search. Sharded across array index.
#
# Submit examples:
#   XTB_MODE=pilot sbatch --array=0-9%10 sbatch/02_xtb_path.sh
#   XTB_MODE=full  sbatch --array=0-19%20 sbatch/02_xtb_path.sh
#
# Each array task gets --shard $SLURM_ARRAY_TASK_ID --nshard $SLURM_ARRAY_TASK_COUNT.
# After all shards finish, run sbatch/02b_merge.sh to consolidate.
#
#SBATCH --job-name=xtbpath
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_guess/logs/02_%A_%a.out

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_guess

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export XTB_MODE="${XTB_MODE:-pilot}"
export XTB_GFN="${XTB_GFN:-2}"
export XTB_THREADS="${XTB_THREADS:-4}"
export XTB_TIMEOUT_S="${XTB_TIMEOUT_S:-600}"
export OMP_NUM_THREADS="$XTB_THREADS"
export MKL_NUM_THREADS="$XTB_THREADS"

# Determine total shard count from array spec (SLURM sets this).
# Fallback to 1 if not an array job.
NSHARD="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

cd "$BASE"
python 02_xtb_path.py --mode "$XTB_MODE" --shard "$SHARD" --nshard "$NSHARD"
