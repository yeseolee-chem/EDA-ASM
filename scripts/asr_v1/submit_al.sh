#!/bin/bash
# AL orchestrator: 5 rounds × 40 picks via Δ-learning M1 uncertainty.
# Pre-req: outputs/asr_v1/al/pool_features.pt (run al_cache_pool.py first).
#
# Submit: sbatch scripts/asr_v1/submit_al.sh
#
# The orchestrator runs on a GPU node because each round trains a Δ-M1
# 5-ensemble. It then idles (sbatch --wait) while the CPU ADF batch runs.
# 5 rounds × (~30 min training + ~6h ADF avg) ≈ 33 h wall.

#SBATCH --job-name=asr_v1_al
#SBATCH --partition=gpu1,gpu3,gpu4,gpu5,gpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/asr_v1/al/logs/al-%j.out
#SBATCH --error=outputs/asr_v1/al/logs/al-%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"
mkdir -p "$PROJECT_DIR/outputs/asr_v1/al/logs"
cd "$PROJECT_DIR"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[$(date)] === AL orchestrator start (job $SLURM_JOB_ID) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import mace, torch; print(f'mace-torch {mace.__version__}  torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

START_ROUND="${START_ROUND:-1}"
END_ROUND="${END_ROUND:-5}"
N_PICK="${N_PICK:-40}"

python -u scripts/asr_v1/run_al.py \
    --start-round "$START_ROUND" \
    --end-round "$END_ROUND" \
    --n-pick "$N_PICK"

echo "[$(date)] === AL orchestrator done ==="
