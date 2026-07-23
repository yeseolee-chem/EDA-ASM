#!/bin/bash
#SBATCH --job-name=espley_s1
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/espley_s1_%j.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/espley_s1_%j.err

# spec18_espley_s1_labels — build 2-channel DIAS labels on LOCKED_778.
# Pure CPU arithmetic + schema construction. Idempotent on resubmit.
# CLAUDE.md: --time=48:00:00 mandatory even for a few-minute job.

set -euo pipefail

STAGE="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref Comparison/spec18_espley_s1_labels"

mkdir -p /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

cd "$STAGE"

echo "=== [$(date -Is)] build_2ch_labels.py ==="
python "$STAGE/code/build_2ch_labels.py"

echo "=== [$(date -Is)] compare_to_ds3.py ==="
python "$STAGE/code/compare_to_ds3.py"

echo "=== [$(date -Is)] aggregate.py ==="
python "$STAGE/code/aggregate.py"

echo "=== [$(date -Is)] DONE ==="
