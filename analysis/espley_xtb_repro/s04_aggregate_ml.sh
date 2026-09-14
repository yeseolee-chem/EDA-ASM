#!/bin/bash
#SBATCH --job-name=xtb_ml_agg
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/logs/ml_agg.%j.out
set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /home1/yeseo1ee/projects/eda-asm-prediction/analysis/espley_xtb_repro
python aggregate_ml.py
