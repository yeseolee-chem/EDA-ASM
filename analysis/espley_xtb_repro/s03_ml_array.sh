#!/bin/bash
# Array element = one target (0..10). Each runs Protocol A (Ridge/KRR/SVR/XGB)
# + Protocol B (Linear/Ridge/RF/GBR/XGB) on both E5 and ESPLEY54 feature sets.
# 11 targets -> array of 11, %10 concurrent (MaxJobs=10 UBAI limit).
#SBATCH --job-name=xtb_ml
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --array=0-10%10
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/logs/ml_target_%a.%j.out
set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
python -c "import xgboost" 2>/dev/null || pip install --user --quiet xgboost
cd /home1/yeseo1ee/projects/eda-asm-prediction/analysis/espley_xtb_repro
export ESPLEY_NJOBS=${SLURM_CPUS_PER_TASK}
python train_ml_single.py ${SLURM_ARRAY_TASK_ID}
