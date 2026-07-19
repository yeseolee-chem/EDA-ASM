#!/bin/bash
# SPEC_10 — sbatch wrapper for aggregate_lc_family.py.
# CLAUDE.md: matplotlib rendering (and any python) must run on a compute
# node, never on gate1.hpc. Even a small learning-curve plot goes through
# sbatch. 48h walltime per repo rule.

#SBATCH --job-name=s10_agg
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec10_agg_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec10_agg_%j.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[node] $(hostname)  [start] $(date '+%F %T')"
python -u spec/spec10_family_learning_curve/code/aggregate_lc_family.py
echo "[done] $(date '+%F %T')"
