#!/bin/bash
#SBATCH --job-name=espley_s22_build
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/espley_s22_build_%j.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/espley_s22_build_%j.err

# Generate pilot inputs (5 rxns × 3 job types = 15 ORCA inputs).
# CLAUDE.md: --time=48:00:00 mandatory.

set -euo pipefail

STAGE="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref Comparison/spec22_relabel400_wb97x3c"
mkdir -p /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

python "$STAGE/code/build_pilot_inputs.py"
echo "=== [$(date -Is)] DONE ==="
