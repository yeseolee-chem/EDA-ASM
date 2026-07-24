#!/bin/bash
#SBATCH --job-name=espley_s1r1
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/espley_s1r1_%j.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/espley_s1r1_%j.err

# spec18r1_espley_s1_labels_fix — build → verify → aggregate.
# Order matters: build writes .pkl → verify reloads from disk → aggregate reads .pkl.
# CLAUDE.md: --time=48:00:00 mandatory even for a minutes-long job.

set -euo pipefail

STAGE="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref Comparison/spec18r1_espley_s1_labels_fix"
mkdir -p /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

cd "$STAGE"

echo "=== [$(date -Is)] build_2ch_labels.py ==="
python "$STAGE/code/build_2ch_labels.py"

echo "=== [$(date -Is)] verify_artifact.py ==="
python "$STAGE/code/verify_artifact.py"

echo "=== [$(date -Is)] aggregate.py ==="
python "$STAGE/code/aggregate.py"

echo "=== [$(date -Is)] DONE ==="
