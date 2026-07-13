#!/bin/bash
# Watcher: m3 완료 → 현재 실행 m1/m2 fold 잡을 "다음 member 시작 전"에 안전 취소
# → spec 잡들이 자연 슬롯 확보 → spec 완료 후 취소된 m1/m2 fold 재제출 (idempotent skip)
#
# 안전성:
#   - 각 fold의 진행 중 member는 완료될 때까지 대기 (로그에 "OK $MK fold=$F member=$M" 나타남)
#   - 그 직후 (다음 python-init 단계에서) scancel → 실제 훈련 시작 전에 종료
#   - 이미 완료된 member는 member{N}.json에 저장되어 idempotent 재개 가능

#SBATCH --job-name=watcher
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/watcher.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/watcher.%j.err

set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

LOG_DIR=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs

echo "[$(date +%H:%M:%S)] Phase 1: wait for m3 v9 completion"
while true; do
  n_m3=$(squeue -u yeseo1ee -h -o '%j' 2>/dev/null | grep -c v9m3_train || true)
  [ "$n_m3" = "0" ] && break
  sleep 30
done
echo "[$(date +%H:%M:%S)] m3 v9 all done"

# List of currently-running m1/m2 fold jobs to watch
declare -A JOB_META  # JID -> "mk fold"
mapfile -t RUNNING < <(squeue -u yeseo1ee -h -t RUNNING -o '%i %j' | awk '$2=="v9m12_tr" {print $1}')
echo "[$(date +%H:%M:%S)] m1/m2 running jobs to safe-cancel: ${#RUNNING[@]} — ${RUNNING[*]}"

# Record what fold each running job belongs to (for later resubmit)
declare -A CANCEL_TARGETS  # JID -> "MK FOLD"
for JID in "${RUNNING[@]}"; do
  LOG="$LOG_DIR/train_v9m12_$JID.out"
  # Read the first line to determine MK/FOLD
  # Example: "[HH:MM:SS] v9 m1 fold=3 BASELINE=..."
  info=$(grep -m1 "^\[.*v9 m[12] fold=" "$LOG" 2>/dev/null | sed -E 's/.*v9 (m[12]) fold=([0-9]+).*/\1 \2/')
  if [ -n "$info" ]; then
    CANCEL_TARGETS[$JID]=$info
  fi
done

# Phase 2: watch for "OK $MK fold=$F member=$M" then immediate scancel
echo "[$(date +%H:%M:%S)] Phase 2: watch m1/m2 logs for member-complete markers"
for JID in "${!CANCEL_TARGETS[@]}"; do
  LOG="$LOG_DIR/train_v9m12_$JID.out"
  (
    # tail from CURRENT END - only new "OK" lines
    tail -n 0 -f "$LOG" 2>/dev/null | while read -r line; do
      if echo "$line" | grep -qE "^OK v9m12|OK m[12] fold="; then
        echo "[$(date +%H:%M:%S)] watched OK on $JID (${CANCEL_TARGETS[$JID]}) → scancel"
        scancel "$JID"
        exit 0
      fi
    done
  ) &
done
wait

echo "[$(date +%H:%M:%S)] Phase 2 done — all watched m1/m2 jobs cancelled after member-complete."

# Phase 3: wait for spec chain to complete (spec_trigger submits spec5/4/2, they run)
echo "[$(date +%H:%M:%S)] Phase 3: wait for spec chain completion"
while true; do
  n_spec=$(squeue -u yeseo1ee -h -o '%j' 2>/dev/null | grep -cE "v9_spec|spec_trig" || true)
  [ "$n_spec" = "0" ] && break
  sleep 60
done
echo "[$(date +%H:%M:%S)] spec chain done"

# Phase 4: resubmit cancelled m1/m2 folds (idempotent — skips completed members)
echo "[$(date +%H:%M:%S)] Phase 4: resubmit m1/m2 fold jobs"
for JID in "${!CANCEL_TARGETS[@]}"; do
  read MK FOLD <<< "${CANCEL_TARGETS[$JID]}"
  NEW=$(sbatch --parsable --export=ALL,MK="$MK",FOLD="$FOLD" \
    scripts/v9_ml/train_v9_m1m2_fold.sh 2>&1)
  echo "  resubmit $MK fold=$FOLD → $NEW"
done
echo "[$(date +%H:%M:%S)] all done"
