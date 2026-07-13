#!/bin/bash
#SBATCH --job-name=v9_spec02BC
#SBATCH --partition=gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH --array=0-9%10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_spec02BC_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/v9_spec02BC_%A_%a.err
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
export BUNDLE_PT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/bundles_v9/features_v6_delta_m3.pt
export SPLIT_ROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9
TASK=$SLURM_ARRAY_TASK_ID
if [ "$TASK" -lt 5 ]; then ARM=ridge; FOLD=$TASK; else ARM=xgb; FOLD=$((TASK-5)); fi
python -u spec/spec02_abc_ablation/code/arm_BC_delta.py --task $TASK --arm $ARM --fold $FOLD
