#!/bin/bash
#SBATCH --job-name=sv1_st3
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/st3_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/st3_%j.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "=== st3a: fragment partitions ==="
python -u pipeline_rebuild/spec_v1/fragment_partition.py \
    labels/adf/adf_labels_v6_multifamily.parquet \
    /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/partitions.json

echo ""
echo "=== st3b: xTB descriptors d1..d24 ==="
python -u pipeline_rebuild/spec_v1/stage3_xtb_and_descriptors.py
