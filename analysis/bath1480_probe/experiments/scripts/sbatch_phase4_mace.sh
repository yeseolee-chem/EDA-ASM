#!/bin/bash
# Phase 4 prep: MACE-OFF23 TS feature extraction. 3-way parallel on GPU.
#
#SBATCH --job-name=exp_p4mace
#SBATCH --time=48:00:00
#SBATCH --partition=gpu1,gpu2,gpu3,gpu4,gpu5,gpu6
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/results/phase4_mace.%A_%a.log

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/scripts
echo "=== MACE TS task=$SLURM_ARRAY_TASK_ID host=$(hostname) start=$(date -Is) ==="
python3 -u $SCRIPTS/03_extract_mace_ts.py
echo "=== end $(date -Is) ==="
