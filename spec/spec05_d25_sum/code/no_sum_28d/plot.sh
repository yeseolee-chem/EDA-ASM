#!/bin/bash
#SBATCH --job-name=s5_28d_plot
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/s5_28d_plot.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/s5_28d_plot.%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
python -u spec/spec05_d25_sum/code/no_sum_28d/plot_comparison.py
