#!/bin/bash
# Post-run: aggregate metrics + build plots + emit REPORT.md.
# CPU-only, ≤5 min actual work but 48h walltime per CLAUDE.md.

#SBATCH --job-name=abc_final
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=experiments/abc_ablation/logs/final-%j.out
#SBATCH --error=experiments/abc_ablation/logs/final-%j.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[$(date)] === aggregate metrics ==="
python -u experiments/abc_ablation/aggregate.py

echo "[$(date)] === build plots ==="
python -u experiments/abc_ablation/plots.py

echo "[$(date)] === REPORT.md ==="
python -u experiments/abc_ablation/report.py

echo "[$(date)] === abc_final done ==="
