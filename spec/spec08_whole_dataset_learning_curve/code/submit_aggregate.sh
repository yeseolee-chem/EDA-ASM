#!/bin/bash
# SPEC_08 whole-dataset LC — sbatch wrapper for aggregate_lc.py.
# CLAUDE.md: matplotlib rendering must run on compute node.

#SBATCH --job-name=s08w_agg
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec08w_agg_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec08w_agg_%j.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[node] $(hostname)  [start] $(date '+%F %T')"
python -u spec/spec08_whole_dataset_learning_curve/code/aggregate_lc.py
echo "[done] $(date '+%F %T')"
