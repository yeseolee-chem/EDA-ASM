#!/bin/bash
# Phase A: re-parse ORCA outputs (gate 0.50) + rebuild manual_tt_7ch.pkl
#SBATCH --job-name=prep_7ch
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/pipeline_prep.%j.log

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results
SCRIPTS=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/kisti_results/scripts

echo "=== 1. re-parse (gate 0.50) ==="
python3 -u $SCRIPTS/orca_eda_parser.py --batch $BASE/data --glob '*/eda.out' --out $BASE/labels_6ch.parquet

echo ""
echo "=== 2. label_builder → manual_tt_7ch.pkl ==="
python3 -u $SCRIPTS/label_builder.py \
    --eda-root $BASE/data --glob '*/eda.out' \
    --pkl /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/gh_repo/tt_ml/manual_tt_solvent.pkl \
    --out $BASE/manual_tt_7ch.pkl \
    --audit $BASE/label_audit.json

echo ""
echo "=== 산출물 ==="
ls -la $BASE/labels_6ch.parquet $BASE/manual_tt_7ch.pkl $BASE/label_audit.json
