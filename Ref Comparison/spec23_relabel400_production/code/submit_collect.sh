#!/bin/bash
#SBATCH --job-name=s23_collect
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/s23_collect_%j.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/s23_collect_%j.err

# Interim collector — scans workdir, builds label parquet from whatever
# reactions currently have all 3 job outputs (eda + fragA_opt + fragB_opt).
# Safe to re-run at any time.

set -euo pipefail
STAGE="/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref Comparison/spec23_relabel400_production"
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
python "$STAGE/code/collect_and_build_labels.py"
echo "=== [$(date -Is)] DONE ==="
