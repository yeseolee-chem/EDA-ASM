#!/bin/bash
#SBATCH --job-name=v9_d2628
#SBATCH --partition=cpu2
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_d2628_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_d2628_%A_%a.err
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
python -u spec/spec05_d25_sum/code/from_spec06/compute_d26_28.py --shard $SLURM_ARRAY_TASK_ID --nshards 6
