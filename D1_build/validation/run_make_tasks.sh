#!/bin/bash
# run_make_tasks.sh — sample + manifest + task list on a compute node (make_tasks.py).
# With XLSX=<path to D1_설계_v8.xlsx> exported, V6a is drawn first (sample_pilot.py, seed 20260930,
# 2 per panel, pilot rows excluded) into v6a_manifest.csv; without it V6a is left out.
#SBATCH --job-name=d1v_tasks
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
set -uo pipefail
VAL_DIR=${VAL_DIR:?}
export PILOT_DIR=$(cd "$VAL_DIR/../d1_autode_pilot" && pwd)
export D1P_CONFIG=$VAL_DIR/config_val.yaml
source "$PILOT_DIR/pilot_env.sh"
cd "$VAL_DIR" || exit 1
if [ -n "${XLSX:-}" ] && [ -f "$XLSX" ]; then
    python "$PILOT_DIR/sample_pilot.py" --xlsx "$XLSX" --out v6a_manifest.csv --seed 20260930 --n 10 \
        --controls 0 --exclude-ts "$PILOT_DIR/pilot_manifest.csv" --id-prefix V6a_ || exit 1
fi
python make_tasks.py
