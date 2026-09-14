#!/bin/bash
#SBATCH --job-name=xtb_pairs
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/logs/pairs.%j.out
set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /home1/yeseo1ee/projects/eda-asm-prediction/analysis/espley_xtb_repro
python evaluate_pairs.py results /gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv
# outputs go to CWD (this dir)
ls -la split_metrics.csv margin_calibration.csv mmp_pairs_v2.csv 2>&1
