#!/bin/bash
#SBATCH --job-name=espley_s20_discover
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/espley_s20_%j.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/espley_s20_%j.err

# spec20 G20-0: discovery + aggregate ONLY. Production compute is BLOCKED
# by §7 until G20-0 finding is reviewed. --time=48:00:00 per CLAUDE.md
# even for a minutes-long CPU job.

set -euo pipefail

STAGE="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref Comparison/spec20_locked778_fragment_relax"
mkdir -p /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

cd "$STAGE"

echo "=== [$(date -Is)] discover_protocol.py ==="
python "$STAGE/code/discover_protocol.py"

echo "=== [$(date -Is)] aggregate.py ==="
python "$STAGE/code/aggregate.py"

echo "=== [$(date -Is)] DONE ==="
