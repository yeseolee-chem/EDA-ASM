#!/bin/bash
# Phase 3 — ADF EDA batch driver on gate1 (license is gate1-locked).
# Runs N substrates concurrently via xargs -P. Each substrate runs ~5 ADF jobs.
# Usage:
#   bash scripts/run_eda_batch_gate1.sh "nme2 nh2 oh ..." [concurrency]
set -o pipefail

IDS="${1:-nme2 nh2 oh ome me ph f i br cl cf3 ac cn no2}"
CONCURRENCY="${2:-3}"

PROJECT=/gpfs/home1/yeseo1ee/projects/v1-claisen-asr
cd "$PROJECT"
source $HOME/ams2026.103/amsbashrc.sh
export NSCM=1
export OMP_NUM_THREADS=1

mkdir -p _out/eda_logs

START=$(date +%s)
echo "[$(date -Is)] EDA batch start: $(echo $IDS | wc -w) substrates, concurrency=$CONCURRENCY"

run_one() {
    local id="$1"
    local log="_out/eda_logs/${id}.log"
    local wd="/gpfs/tmp_cpu2/yeseo1ee_plams/${id}_batch"
    if [ -f "runs/$id/eda/.done_eda" ]; then
        echo "[$(date -Is)] SKIP $id (already .done_eda)"
        return 0
    fi
    rm -rf "$wd"
    mkdir -p "$wd"
    echo "[$(date -Is)] START $id (wd=$wd)"
    nice -n 15 "$AMSBIN/amspython" scripts/run_eda.py --id "$id" --workdir "$wd" > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -f "runs/$id/eda/asr_vector.json" ]; then
        # one-line summary
        ea=$(python3 -c "import json; d=json.load(open('runs/$id/eda/asr_vector.json'))
print(f'strain={d[\"E_strain\"]:+.2f} Pauli={d[\"E_Pauli\"]:+.2f} elst={d[\"E_elstat\"]:+.2f} oi={d[\"E_oi\"]:+.2f} disp={d[\"E_disp\"]:+.2f} int={d[\"E_int\"]:+.2f}')" 2>/dev/null)
        echo "[$(date -Is)] OK    $id  $ea"
    else
        echo "[$(date -Is)] FAIL  $id  rc=$rc — see $log"
    fi
}
export -f run_one
export AMSBIN

echo "$IDS" | tr ' ' '\n' | xargs -P "$CONCURRENCY" -I {} bash -c 'run_one "$@"' _ {}

END=$(date +%s)
echo "[$(date -Is)] batch complete in $((END-START))s"

# Final summary
echo
echo "=== ASR vectors summary ==="
for id in h $IDS; do
  asr="runs/$id/eda/asr_vector.json"
  if [ -f "$asr" ]; then
    python3 -c "
import json
d = json.load(open('$asr'))
print(f'{\"$id\":>6s}  strain={d[\"E_strain\"]:+7.2f}  Pauli={d[\"E_Pauli\"]:+7.2f}  elst={d[\"E_elstat\"]:+7.2f}  oi={d[\"E_oi\"]:+7.2f}  disp={d[\"E_disp\"]:+7.2f}  int={d[\"E_int\"]:+7.2f}')"
  else
    printf '%6s  (no asr_vector.json)\n' "$id"
  fi
done
