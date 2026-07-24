#!/bin/bash
#SBATCH --job-name=s23_build
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/s23_build_%j.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/s23_build_%j.err

# Build all 1200 ORCA input directories + job_manifest.csv.
# CLAUDE.md: --time=48:00:00 mandatory.

set -euo pipefail
STAGE="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref Comparison/spec23_relabel400_production"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

python "$STAGE/code/build_inputs.py"
echo "=== [$(date -Is)] build DONE ==="
