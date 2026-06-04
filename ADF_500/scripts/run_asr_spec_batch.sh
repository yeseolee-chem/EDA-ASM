#!/bin/bash
# Spec-compliant ADF batch (BP86/D3BJ/TZ2P/Good per ASR_Fragmentation_Spec.md).
# Re-computes the 500 original cohort with 11 ADF jobs per reaction.
# Wall-time estimate: ~30-60 min per reaction, ~30-60 hours total at 4 concurrent.
set -e
CONCURRENCY="${1:-4}"
RXN_LIST="${2:-outputs/asr_spec_rxn_list.txt}"

cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction
source /home1/yeseo1ee/.bashrc 2>/dev/null
module load mpi/2021.9.0 2>/dev/null
source $HOME/ams2026.103/amsbashrc.sh
export PYTHONPATH=src:.

mkdir -p outputs/asr_spec outputs/asr_spec/logs

N_TOTAL=$(wc -l < "$RXN_LIST")
N_DONE=$(find outputs/asr_spec -maxdepth 1 -name '*.json' \
         -exec grep -l '"status_at_queue": "AUTO_ACCEPT_CANDIDATE"\|"status_at_queue": "MANUAL_REVIEW_REQUIRED"' {} \; 2>/dev/null | wc -l)
echo "==============================================================="
echo " ASR-spec ADF batch (BP86/D3BJ/TZ2P/Good)"
echo " start:    $(date)"
echo " total:    $N_TOTAL    done already: $N_DONE"
echo " concur:   $CONCURRENCY"
echo " host:     $(hostname)"
echo "==============================================================="

run_one() {
    local rxn="$1"
    local log="outputs/asr_spec/logs/${rxn}.log"
    {
        echo "[$(date '+%H:%M:%S')] START $rxn"
        NSCM=1 nice -n 10 \
            $AMSBIN/amspython scripts/run_asr_spec.py --rxn_id "$rxn"
        echo "[$(date '+%H:%M:%S')] END $rxn rc=$?"
    } >> "$log" 2>&1
}
export -f run_one
export AMSBIN PYTHONPATH

cat "$RXN_LIST" | xargs -I{} -P "$CONCURRENCY" bash -c 'run_one "$@"' _ {}

N_FINAL=$(find outputs/asr_spec -maxdepth 1 -name '*.json' \
         -exec grep -l '"status_at_queue"' {} \; 2>/dev/null | wc -l)
echo "==============================================================="
echo " done: $(date)  completed=$N_FINAL/$N_TOTAL"
echo "==============================================================="
