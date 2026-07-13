#!/bin/bash
#SBATCH --job-name=v9_merge
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_merge.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_merge.%j.err
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
python -u spec/spec05_d25_sum/code/merge_d25.py 2>&1 || true
python -u spec/spec05_d25_sum/code/from_spec06/merge_d26_28.py 2>&1 || true
