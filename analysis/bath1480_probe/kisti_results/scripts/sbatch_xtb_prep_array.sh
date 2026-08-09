#!/bin/bash
# Phase 2: XYZ file prep + xtb --opt for GS fragments (parallel shards)
# 3504 rxns / 10 shards = ~350 rxns/shard, each rxn ~30s = ~3h/shard
#SBATCH --job-name=xtb_prep
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-9%10
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/xtb_prep.%A_%a.log

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results
SCRIPT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/kisti_results/scripts/build_xtb_manifest.py

echo "=== xtb prep shard $SLURM_ARRAY_TASK_ID/10  host=$(hostname)  time=$(date -Is) ==="
python3 -u $SCRIPT \
    --root $BASE/data \
    --out $BASE/xtb_input \
    --shard $SLURM_ARRAY_TASK_ID \
    --nshards 10 \
    --xtb xtb \
    --solvent water
echo "=== end  time=$(date -Is) ==="
