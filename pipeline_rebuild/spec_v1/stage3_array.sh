#!/bin/bash
# 8-way parallel Stage 3 (xTB descriptors) — each shard covers ~100 reactions.

#SBATCH --job-name=sv1_st3a
#SBATCH --array=0-7%8
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/st3a_%A_%a.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/st3a_%A_%a.err

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

# Partitions must exist before shards. Task 0 builds them if missing;
# other tasks wait briefly.
if [ ! -f /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/partitions.json ]; then
  if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    python -u pipeline_rebuild/spec_v1/fragment_partition.py \
      labels/adf/adf_labels_v6_multifamily.parquet \
      /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/partitions.json
  else
    while [ ! -f /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/partitions.json ]; do
      sleep 5
    done
  fi
fi

python -u pipeline_rebuild/spec_v1/stage3_xtb_and_descriptors.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards 8
