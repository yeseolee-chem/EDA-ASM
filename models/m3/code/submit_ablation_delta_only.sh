#!/bin/bash
# m3 spec2 ablation: delta_only (baseline pinned to 0).
# 5 array tasks = fold0 x members 0..4. Runs concurrently on 3 GPU partitions.

#SBATCH --job-name=abl_m3_do
#SBATCH --array=0-4%5
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/abl_m3_do_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/abl_m3_do_%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

# Make sure the v9_all subsamples symlink is present (fold0/size_626.json).
mkdir -p outputs/asr_v1/phase3/subsamples
ln -sfn /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/subsamples_v9 \
        outputs/asr_v1/phase3/subsamples/v9_all

export MODE=delta_only
export BASELINE=xtb_geom6_plus_v2
export SUBSAMPLES_TAG=v9_all
export SIZE_FULL=626
export OUT_TAG=ablation
export FOLD=0
export EPOCHS_MAX=100000
export PATIENCE=10000

echo "[$(date)] === m3 ablation delta_only member=${SLURM_ARRAY_TASK_ID} start ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true

python -u models/m3/code/runner_ablation_m3.py --fold 0 --member ${SLURM_ARRAY_TASK_ID}

echo "[$(date)] === m3 ablation delta_only member=${SLURM_ARRAY_TASK_ID} done ==="
