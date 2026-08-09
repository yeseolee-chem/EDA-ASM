#!/bin/bash
#SBATCH --job-name=tt_verify
#SBATCH --time=48:00:00
#SBATCH --partition=REPLACE_ME_PARTITION
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/verify.%j.log

set -euo pipefail
PKG_ROOT="${PKG_ROOT:-$(pwd)}"
mkdir -p "$PKG_ROOT/results" "$PKG_ROOT/logs"

python3 "$PKG_ROOT/scripts/verify_vs_espley.py" \
    "$PKG_ROOT/results/tt_eda_5channel.parquet" \
    "$PKG_ROOT/data/espley_reference.parquet" \
    "$PKG_ROOT/results/verification_report.md"
