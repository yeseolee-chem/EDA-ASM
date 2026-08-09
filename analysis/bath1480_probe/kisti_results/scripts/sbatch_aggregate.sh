#!/bin/bash
#SBATCH --job-name=tt_aggregate
#SBATCH --time=48:00:00
#SBATCH --partition=REPLACE_ME_PARTITION
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/aggregate.%j.log

set -euo pipefail

PKG_ROOT="${PKG_ROOT:-$(pwd)}"
mkdir -p "$PKG_ROOT/results" "$PKG_ROOT/logs"

# Assumes conda env with pandas, pyarrow is active (see env/setup_env.sh)
python3 "$PKG_ROOT/scripts/aggregate_channels.py" \
    "$PKG_ROOT/data" \
    "$PKG_ROOT/results/tt_eda_5channel.parquet"
