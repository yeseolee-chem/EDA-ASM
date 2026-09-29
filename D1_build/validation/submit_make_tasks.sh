#!/bin/bash
# submit_make_tasks.sh [XLSX] — val_d0_set.csv, val_manifest.csv, tasks.csv (+ v6a_manifest.csv if the
# D1 design workbook is given). One-shot from the login node. Commit the outputs before submit_validation.sh.
set -euo pipefail
source "$(dirname "$0")/_submit_common.sh"
need_slots 1
X=${1:-}
[ -z "$X" ] || [ -f "$X" ] || { echo "STOP: $X not found"; exit 1; }
J=$(sbatch --parsable -p "$P" -o "$SCR/logs/make_tasks_%j.log" --export=ALL,VAL_DIR="$VAL",XLSX="$X" run_make_tasks.sh)
echo "partition=$P  MAKE_TASKS=$J  -> $SCR/logs/make_tasks_$J.log"
