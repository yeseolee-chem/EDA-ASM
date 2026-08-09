#!/bin/bash
#SBATCH --job-name=eda_parse
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/parse_batch.%j.log

set -euo pipefail

BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results
SCRIPT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/kisti_results/scripts/orca_eda_parser.py
OUT=$BASE/labels_6ch.parquet

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "=== parser batch start  host=$(hostname)  time=$(date -Is) ==="
python3 -u "$SCRIPT" --batch "$BASE/data" --glob '*/eda.out' --out "$OUT"
echo "=== end  time=$(date -Is) ==="
ls -la "$OUT" ${OUT%.parquet}.failures.csv 2>&1
