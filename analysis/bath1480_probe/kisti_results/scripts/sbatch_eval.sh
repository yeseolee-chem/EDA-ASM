#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/eval/eval.%j.log
set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
SC=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/kisti_results/scripts
echo "=== 1. export ==="; python3 -u $SC/export_predictions.py
echo "=== 2. metrics ==="; python3 -u $SC/eval_metrics.py
echo "=== 3. figures ==="; python3 -u $SC/eval_figures.py
echo "=== done ==="
