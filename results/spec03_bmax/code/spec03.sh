#!/bin/bash
#SBATCH --job-name=spec03
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec03_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec03_%j.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

# xgboost is not part of the base reactot env; install it inside the sbatch
# (compute node only). Idempotent: skips the install if already present.
python -c "import xgboost" 2>/dev/null || pip install --quiet --user xgboost

python -u pipeline_rebuild/spec_v1/spec03_bmax.py
