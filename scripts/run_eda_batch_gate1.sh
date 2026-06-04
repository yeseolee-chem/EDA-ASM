#!/bin/bash
# Parallel launcher for EDA-ASM batch on gate1 login node.
#
# License is locked to gate1 (compute nodes reject). Each reaction
# runs as a separate `amspython run_eda_one.py --rxn_id <ID>` process
# with NSCM=1 (no MPI). `xargs -P` controls concurrency.
#
# Idempotent: run_eda_one.py skips reactions whose eda_result.json
# already has status="ok". Safe to re-run after kill/crash.
#
# Usage:
#   nohup bash scripts/run_eda_batch_gate1.sh 4 \
#       > outputs/stage5b/batch_runner.log 2>&1 &
#   echo $! > outputs/stage5b/batch_runner.pid

CONCURRENCY="${1:-4}"

cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

# Environment for AMS — required because nohup'd processes don't inherit
# the user's interactive shell config.
source /home1/yeseo1ee/.bashrc 2>/dev/null
module load mpi/2021.9.0 2>/dev/null
source $HOME/ams2026.103/amsbashrc.sh

mkdir -p outputs/stage5b/per_reaction outputs/stage5b/batch_logs

RXN_LIST=outputs/stage5b/rxn_id_list.txt
N_TOTAL=$(wc -l < "$RXN_LIST")

# Count already-done up front
N_DONE=$(find outputs/stage5b/per_reaction -name eda_result.json 2>/dev/null \
         | xargs -I{} grep -l '"status": "ok"' {} 2>/dev/null | wc -l)
N_REMAINING=$((N_TOTAL - N_DONE))

echo "==============================================================="
echo " gate1 parallel EDA-ASM launcher"
echo " started:     $(date)"
echo " total:       $N_TOTAL"
echo " done:        $N_DONE"
echo " remaining:   $N_REMAINING"
echo " concurrency: $CONCURRENCY"
echo " host:        $(hostname)"
echo "==============================================================="

# nice -n 10 = lower CPU priority (be polite on shared login node)
# Each worker writes its own log under batch_logs/<rxn_id>.log
run_one() {
    local rxn="$1"
    local log="outputs/stage5b/batch_logs/${rxn}.log"
    local out_dir="outputs/stage5b/per_reaction/${rxn}"
    mkdir -p "$out_dir"
    {
        echo "[$(date '+%H:%M:%S')] START $rxn on $(hostname)"
        NSCM=1 nice -n 10 \
            $AMSBIN/amspython scripts/run_eda_one_v3.py --rxn_id "$rxn"
        local rc=$?
        echo "[$(date '+%H:%M:%S')] END   $rxn rc=$rc"
        if [ $rc -ne 0 ]; then
            echo "{\"rxn_id\": \"$rxn\", \"rc\": $rc, \"timestamp\": \"$(date -Iseconds)\"}" \
                >> outputs/stage5b/failed_reactions.jsonl
        fi
        echo "$rxn $rc $(date +%s)" >> outputs/stage5b/batch_progress.tsv
    } >> "$log" 2>&1
}
export -f run_one
export AMSBIN

# Stream rxn IDs, run in parallel with xargs -P
cat "$RXN_LIST" | xargs -I{} -P "$CONCURRENCY" bash -c 'run_one "$@"' _ {}

echo "==============================================================="
echo " batch finished: $(date)"
N_FINAL=$(find outputs/stage5b/per_reaction -name eda_result.json 2>/dev/null \
          | xargs -I{} grep -l '"status": "ok"' {} 2>/dev/null | wc -l)
echo " completed: $N_FINAL / $N_TOTAL"
echo "==============================================================="
