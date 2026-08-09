#!/bin/bash
# Phase 3: run xtb_channel_generator per shard (7 xTB channels for each rxn)
# Depends on Phase 2 (xtb_prep) having produced shard manifests + xyz files
#SBATCH --job-name=xtb_ch
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-9%10
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/xtb_ch.%A_%a.log

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results
SHARD=$SLURM_ARRAY_TASK_ID
MANIFEST=$BASE/xtb_input/manifest_shard$(printf '%02d' $SHARD).csv
OUT=$BASE/xtb_channels/channels_shard$(printf '%02d' $SHARD).parquet

if [ ! -f "$MANIFEST" ]; then
    echo "SKIP: no manifest at $MANIFEST"
    exit 0
fi

if [ -f "$OUT" ]; then
    echo "SKIP: $OUT exists"
    exit 0
fi

mkdir -p $BASE/xtb_channels

echo "=== xtb channels shard $SHARD  host=$(hostname)  time=$(date -Is) ==="
python3 -u /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/kisti_results/scripts/xtb_channel_generator.py \
    --manifest $MANIFEST \
    --out $OUT \
    --xtb xtb \
    --solvent water \
    --workers 4
echo "=== end  time=$(date -Is) ==="
