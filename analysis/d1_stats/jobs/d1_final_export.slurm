#!/bin/bash
# D1 본계산 종료 후 확정 라벨 export — 누리온 코드 aggregate_run.py --final 그대로 (D1 venv · env.sh).
#   → D1_build/d1_run/results/{labels_d1.json, d1_status.csv, geometries/, RUN_REPORT.md, RETRIES.md}  (SPEC_D1_ALL Phase A 의 입력)
#   RELAY_DONE 이 없으면 아무것도 하지 않는다 (진행 중 export 는 ubai/snapshot_labels.slurm 이 따로 한다).
#   sbatch -p cpu1,cpu2 -o analysis/d1_stats/logs/d1_final_export_%j.out analysis/d1_stats/jobs/d1_final_export.slurm
#SBATCH --job-name=d1fin
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G
set -uo pipefail
B=/gpfs/tmp_cpu2/yeseo1ee/d1; R=$B/d1_run
[ -f "$B/d1_scratch/queue/RELAY_DONE" ] || { echo "RELAY_DONE 없음 — D1 본계산이 끝나지 않았다"; cat "$B/d1_scratch/queue/RELAY_STATUS" 2>/dev/null; exit 1; }
cd "$R" || exit 1
source "$R/nurion/env.sh" || exit 2
cat "$B/d1_scratch/queue/RELAY_DONE"
"$D1_PY" aggregate_run.py --final || { echo "aggregate_run.py --final 실패"; exit 1; }
ls -la "$R/results/labels_d1.json" "$R/results/d1_status.csv" "$R/results/RUN_REPORT.md"
sha256sum "$R/results/labels_d1.json" "$R/results/d1_status.csv" "$R/d1_manifest.csv"
