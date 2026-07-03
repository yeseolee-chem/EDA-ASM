#!/bin/bash
# One-time: cache MACE-OFF features + descriptors for the ENTIRE unlabeled
# dipolar + qmrxn pool. Required before submit_al.sh.
#
# Submit: sbatch scripts/asr_v1/submit_al_cache_pool.sh

#SBATCH --job-name=asr_v1_al_cache
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/asr_v1/al/logs/al-cache-%j.out
#SBATCH --error=outputs/asr_v1/al/logs/al-cache-%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"
mkdir -p "$PROJECT_DIR/outputs/asr_v1/al/logs"
cd "$PROJECT_DIR"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[$(date)] === AL pool-feature cache start ==="
python -u scripts/asr_v1/al_cache_pool.py --device cuda
echo "[$(date)] === AL pool-feature cache done ==="
