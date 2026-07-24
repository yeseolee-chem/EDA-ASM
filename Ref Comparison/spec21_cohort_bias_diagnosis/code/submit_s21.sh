#!/bin/bash
#SBATCH --job-name=espley_s21
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/espley_s21_%j.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/espley_s21_%j.err

# spec21: cohort bias diagnosis. No compute — pure diagnostic.
# All steps sequential inside one sbatch. CLAUDE.md: --time=48:00:00.

set -euo pipefail

STAGE="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref Comparison/spec21_cohort_bias_diagnosis"
mkdir -p /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

cd "$STAGE"

echo "=== [$(date -Is)] join_cohort.py (G21-A) ==="
python "$STAGE/code/join_cohort.py"

echo "=== [$(date -Is)] d1_reactivity_position.py ==="
python "$STAGE/code/d1_reactivity_position.py"

echo "=== [$(date -Is)] d2_scaffold_composition.py (G21-C) ==="
python "$STAGE/code/d2_scaffold_composition.py"

echo "=== [$(date -Is)] d3_geometry_provenance.py (G21-B) ==="
python "$STAGE/code/d3_geometry_provenance.py"

echo "=== [$(date -Is)] aggregate.py ==="
python "$STAGE/code/aggregate.py"

echo "=== [$(date -Is)] DONE ==="
