#!/bin/bash
# Master driver — runs all batches sequentially on gate1.
# Per-batch concurrency PAR (default 4) controls how many reactions run at once.
PAR="${1:-4}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
for batch_dir in "$ROOT"/batch_*; do
    [[ -d "$batch_dir" ]] || continue
    echo ""
    echo "=== running $(basename $batch_dir) at $(date -Iseconds) ==="
    bash "$REPO/adf_outputs/run_batch.sh" "$batch_dir" "$PAR"
done
echo ""
echo "=== all batches done at $(date -Iseconds) ==="
