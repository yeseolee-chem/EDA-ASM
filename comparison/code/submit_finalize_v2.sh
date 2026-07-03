#!/bin/bash
# Aggregate + plot comparison m1 / m2 / m3-v2 (member 0, no-OOD + parity outliers removed).

#SBATCH --job-name=fin_v2
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=analysis/exp_6arm_redesign_v2/slurm/logs/fin_v2-%j.out
#SBATCH --error=analysis/exp_6arm_redesign_v2/slurm/logs/fin_v2-%j.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[$(date)] === finalize v2 start on $(hostname) ==="
python -u analysis/exp_6arm_redesign_v2/finalize_compare_m1_m2_m3_member0_noOutliers_v2.py
echo "[$(date)] === finalize v2 done ==="
