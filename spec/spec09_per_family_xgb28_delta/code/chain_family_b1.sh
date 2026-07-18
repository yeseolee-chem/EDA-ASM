#!/bin/bash
# SPEC_06 B1 — poller: submit 20 (family, fold) bundles one at a time as
# MaxSubmit=20 slots free. Run under nohup on login node (shell loop only).
#
# Strategy: keep trying sbatch; if AssocMaxSubmitJobLimit trips (queue at 20),
# sleep 5 min and retry. `train_family_xgb28_delta.py` is idempotent, so no
# risk from spurious resubmit.

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

LOG=spec/spec09_per_family_xgb28_delta/code/chain_family_b1.log
: > "$LOG"

FAMILIES=(dipolar qmrxn20_e2 qmrxn20_sn2 rgd1)
FOLDS=(0 1 2 3 4)

for FAM in "${FAMILIES[@]}"; do
  for F in "${FOLDS[@]}"; do
    while true; do
      OUT=$(sbatch --export=ALL,FAMILY="${FAM}",FOLD="${F}" \
            spec/spec09_per_family_xgb28_delta/code/submit_fold_family.sh 2>&1)
      RC=$?
      TS=$(date '+%F %T')
      if [ "${RC}" -eq 0 ]; then
        echo "[${TS}] OK  ${FAM} fold=${F}: ${OUT}" >> "$LOG"
        break
      fi
      # If AssocMaxSubmitJobLimit — pause and retry
      if echo "${OUT}" | grep -q "AssocMaxSubmitJobLimit"; then
        N=$(squeue -u yeseo1ee -r -h 2>/dev/null | wc -l)
        echo "[${TS}] FULL slots=${N}/20 waiting for ${FAM} fold=${F}" >> "$LOG"
        sleep 300
        continue
      fi
      # Any other error — log and abort (avoid busy loop on real config issue)
      echo "[${TS}] ERR ${FAM} fold=${F} rc=${RC}: ${OUT}" >> "$LOG"
      exit 1
    done
  done
done

echo "[$(date '+%F %T')] all 20 (family, fold) bundles submitted" >> "$LOG"
