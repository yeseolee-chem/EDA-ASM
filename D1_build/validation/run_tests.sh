#!/bin/bash
# run_tests.sh — Phase F unit tests on a compute node (VALIDATION_SPEC §3). Pilot / D0 data read-only.
#SBATCH --job-name=d1v_tests
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
set -uo pipefail
VAL_DIR=${VAL_DIR:?}
export PILOT_DIR=$(cd "$VAL_DIR/../d1_autode_pilot" && pwd)
export D1P_CONFIG=$VAL_DIR/config_val.yaml
source "$PILOT_DIR/pilot_env.sh"
cd "$VAL_DIR" && python tests/test_units.py
