#!/bin/bash
# submit_validation.sh — Phase V (VALIDATION_SPEC §9 step 2): worker pool (10) + final report (afterany) = 11 tasks.
# Refuses unless the pre-registration files are committed and unchanged. Re-run after a 48 h wall:
# workers pick up the remaining tasks (claims of dead workers are taken over), the report is re-queued.
set -euo pipefail
source "$(dirname "$0")/_submit_common.sh"
for f in config_val.yaml val_d0_set.csv tasks.csv val_manifest.csv; do
    git ls-files --error-unmatch "$f" > /dev/null 2>&1 || { echo "STOP: $f not committed (pre-registration, §2)"; exit 1; }
    git diff --quiet HEAD -- "$f" || { echo "STOP: $f differs from its committed version"; exit 1; }
done
[ -f "$SCR/queue/STOP" ] && { echo "STOP file present: $(head -1 "$SCR/queue/STOP")"; exit 1; }
grep -qE '^[[:space:]]+no_ts_unconfirmed_policy:[[:space:]]*TBD' config_val.yaml && \
    { echo "STOP: prereg A1.no_ts_unconfirmed_policy is TBD (user decision D-1)"; exit 1; }
need_slots 11
W=$(sbatch --parsable -p "$P" --array=0-9%10 -o "$SCR/logs/w_%A_%a.log" --export=ALL,VAL_DIR="$VAL" worker.sh)
R=$(sbatch --parsable -p "$P" --dependency=afterany:"$W" -o "$SCR/logs/report_%j.log" --export=ALL,VAL_DIR="$VAL" run_report.sh)
echo "partition=$P  WORKERS=$W (0-9%10)  REPORT=$R  prereg=$(git log -1 --format=%h -- config_val.yaml)"
