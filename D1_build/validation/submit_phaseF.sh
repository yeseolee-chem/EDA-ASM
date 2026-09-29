#!/bin/bash
# submit_phaseF.sh — Phase F (VALIDATION_SPEC §3, §9 step 0): unit tests -> S0 preflight on config_val.yaml
# (S0 starts only if the tests pass: afterok + kill-on-invalid-dep). One-shot from the login node.
set -euo pipefail
source "$(dirname "$0")/_submit_common.sh"
need_slots 2
T=$(sbatch --parsable -p "$P" -o "$SCR/logs/tests_%j.log" --export=ALL,VAL_DIR="$VAL" run_tests.sh)
S=$(sbatch --parsable -p "$P" --dependency=afterok:"$T" --kill-on-invalid-dep=yes -o "$SCR/logs/s0_%j.log" \
      --export=ALL,PILOT_DIR="$PILOT",D1P_CONFIG="$VAL/config_val.yaml" "$PILOT/s0_preflight.sh")
echo "partition=$P  TESTS=$T  S0=$S   -> $SCR/logs/tests_$T.log, $SCR/s0/S0_REPORT.md"
