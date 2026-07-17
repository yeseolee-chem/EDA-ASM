#!/bin/bash
# SPEC_06 — background poller: submit member arrays 2, 3, 4 as SLURM
# submit-slots free up (MaxSubmit=20 for user yeseo1ee).
# Run under nohup on login node (shell loop only, no compute).

set -uo pipefail
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

LOG=spec/spec06_2step_xgb28_delta/code/chain_members.log
: > "$LOG"

for M in 2 3 4; do
    while true; do
        # Count currently-submitted array tasks + regular jobs.
        N=$(squeue -u yeseo1ee -r -h 2>/dev/null | wc -l)
        # Need 5 free slots for a 5-fold array; leave 1 slack -> require N <= 14.
        if [ "${N}" -le 14 ]; then
            echo "[$(date '+%F %T')] slots=${N}/20, submitting MEMBER=${M}" >> "$LOG"
            OUT=$(sbatch --export=ALL,MEMBER=${M} spec/spec06_2step_xgb28_delta/code/submit_train.sh 2>&1)
            RC=$?
            echo "[$(date '+%F %T')] sbatch rc=${RC}: ${OUT}" >> "$LOG"
            if [ "${RC}" -eq 0 ]; then
                break
            fi
        else
            echo "[$(date '+%F %T')] slots=${N}/20 full, waiting" >> "$LOG"
        fi
        sleep 300
    done
done

echo "[$(date '+%F %T')] all members 2/3/4 submitted" >> "$LOG"
