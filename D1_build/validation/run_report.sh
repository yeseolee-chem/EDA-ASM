#!/bin/bash
# run_report.sh — final analysis after the worker array (afterany): results/VALIDATION_REPORT.md.
#SBATCH --job-name=d1v_report
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
set -uo pipefail
VAL_DIR=${VAL_DIR:?}
export PILOT_DIR=$(cd "$VAL_DIR/../d1_autode_pilot" && pwd)
export D1P_CONFIG=$VAL_DIR/config_val.yaml
source "$PILOT_DIR/pilot_env.sh"
cd "$VAL_DIR" && python analysis/validation_report.py --final
