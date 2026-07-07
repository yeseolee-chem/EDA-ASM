#!/bin/bash
# One-shot bootstrap: build splits + run Arm A (xgb_direct OOF).
# Both are CPU-only and fast. 48h walltime per CLAUDE.md.
#SBATCH --job-name=abc_setup
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=experiments/abc_ablation/logs/setup-%j.out
#SBATCH --error=experiments/abc_ablation/logs/setup-%j.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p experiments/abc_ablation/logs experiments/abc_ablation/results

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[$(date)] === abc_setup: build splits ==="
python -u experiments/abc_ablation/build_splits.py

echo "[$(date)] === abc_setup: arm A (xgb_direct OOF) ==="
python -u experiments/abc_ablation/arm_A_run.py --descriptor-set m3

echo "[$(date)] === abc_setup done ==="
