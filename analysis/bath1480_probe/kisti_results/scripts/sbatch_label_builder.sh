#!/bin/bash
#SBATCH --job-name=label_build
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/label_build.%j.log

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results
SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/kisti_results/scripts
PKL=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/gh_repo/tt_ml/manual_tt_solvent.pkl

echo "=== label_builder start  host=$(hostname)  time=$(date -Is) ==="
python3 -u $SCRIPTS/label_builder.py \
    --eda-root $BASE/data \
    --glob '*/eda.out' \
    --pkl $PKL \
    --out $BASE/manual_tt_7ch.pkl \
    --audit $BASE/label_audit.json
echo "=== end  time=$(date -Is) ==="
ls -la $BASE/manual_tt_7ch.pkl $BASE/label_audit.json 2>&1
