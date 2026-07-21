#!/bin/bash
# SPEC_11 - side-by-side 28-d (spec06) vs 33-d (spec11) comparison figures.
# CPU-only, cheap; per CLAUDE.md still --time=48:00:00.
#SBATCH --job-name=s11_compare
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec11_compare_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec11_compare_%j.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

python -u spec/spec11_electronic_33d/code/compare_28d_vs_33d.py
