#!/bin/bash
#SBATCH --job-name=switch_R
#SBATCH --partition=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/switch_R.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/switch_R.%j.err

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

echo "=== 1) Add R coords to any replacement .pt files still missing them ==="
python scripts/add_R_to_replacements.py

echo ""
echo "=== 2) Re-run fragment partitioning: dipolar/rgd1 at R, qmrxn20 at TS ==="
python scripts/refine_fragments.py --geom family

echo ""
echo "=== 3) Write replacement .pt geoms ==="
python scripts/write_replacement_geoms.py

echo ""
echo "=== 4) done ==="
