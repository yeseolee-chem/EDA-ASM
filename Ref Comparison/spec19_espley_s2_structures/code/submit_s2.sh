#!/bin/bash
#SBATCH --job-name=espley_s2
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/espley_s2_%j.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/espley_s2_%j.err

# spec19_espley_s2_structures — full chain: discover → build → common_atoms → verify → crosscheck → aggregate.
# CLAUDE.md: --time=48:00:00 mandatory even for a minutes-long job.

set -euo pipefail

STAGE="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref Comparison/spec19_espley_s2_structures"
mkdir -p /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

cd "$STAGE"

echo "=== [$(date -Is)] discover_geometry_sources.py ==="
python "$STAGE/code/discover_geometry_sources.py"

echo "=== [$(date -Is)] build_structures.py ==="
python "$STAGE/code/build_structures.py"

echo "=== [$(date -Is)] build_common_atoms.py ==="
python "$STAGE/code/build_common_atoms.py"

echo "=== [$(date -Is)] verify_structures.py ==="
python "$STAGE/code/verify_structures.py"

echo "=== [$(date -Is)] diassep_crosscheck.py ==="
python "$STAGE/code/diassep_crosscheck.py"

echo "=== [$(date -Is)] aggregate.py ==="
python "$STAGE/code/aggregate.py"

echo "=== [$(date -Is)] DONE ==="
