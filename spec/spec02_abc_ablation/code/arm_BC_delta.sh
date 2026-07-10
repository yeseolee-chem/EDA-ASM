#!/bin/bash
# Arms B/C delta trainer array: task 0..4 -> ridge_delta fold0..4, task 5..9 -> xgb_delta fold0..4
#SBATCH --job-name=spec02_BC
#SBATCH --array=0-9%3
#SBATCH --partition=gpu1,gpu3,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec02_BC_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec02_BC_%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
python -u spec/spec02_abc_ablation/code/arm_BC_delta.py
