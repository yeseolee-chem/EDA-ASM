#!/bin/bash
# Run build_xtb_extra_cache_v2.py on a CPU compute node — NEVER on login.
# 789 reactions × 1 GFN2-xTB SP, 16 workers parallel → ~10-15 min wall.

#SBATCH --job-name=xtb_extra_v2
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=analysis/exp_6arm_redesign_v2/slurm/logs/xtb_extra_v2-%j.out
#SBATCH --error=analysis/exp_6arm_redesign_v2/slurm/logs/xtb_extra_v2-%j.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p analysis/exp_6arm_redesign_v2/slurm/logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[$(date)] === xtb_extra_v2 cache build start on $(hostname) ==="
python -u analysis/exp_6arm_redesign_v2/build_xtb_extra_cache_v2.py --rebuild --workers 16
echo "[$(date)] === xtb_extra_v2 cache build done ==="

echo "[$(date)] === m3 bundle build (uses v2 cache) start ==="
python -u analysis/exp_6arm_redesign_v2/build_v2_bundle_m3.py
echo "[$(date)] === m3 bundle build done ==="
