#!/bin/bash
#SBATCH --job-name=mlt_quota
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/mlts/logs/mlt_quota_%j.out
# analysis/mlts/jobs/quota.sh — disk-use estimate for the launcher's 30 GB rule (user 2026-11-05: "job에서 du로 추정";
# mmlsquota works only on the login node, which now runs only sbatch and git). The GPFS user quota (300 GB) covers
# /home1/yeseo1ee and /gpfs/tmp_cpu2/yeseo1ee, so free = 300 - du(both). Writes $MLT/QUOTA_FREE_GB '<GB> <ISO time>'
# atomically, appends a line to $MLT/status/quota_history.tsv, then re-queues itself 12 h later (the launcher wants a
# reading younger than 24 h). Submit once from analysis/mlts:  sbatch jobs/quota.sh   (stop: touch $MLT/QUOTA_STOP)
set -uo pipefail
MLT=/gpfs/tmp_cpu2/yeseo1ee/mlts
mkdir -p "$MLT/status"
cd "${SLURM_SUBMIT_DIR:-/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-mlts/analysis/mlts}"
t0=$(date +%s)
h=$(du -s --block-size=1M /home1/yeseo1ee 2>/dev/null | awk '{print $1}')
g=$(du -s --block-size=1M /gpfs/tmp_cpu2/yeseo1ee 2>/dev/null | awk '{print $1}')
if [ -n "$h" ] && [ -n "$g" ]; then
  used=$(( (h + g + 1023) / 1024 ))
  free=$(( 300 - used ))
  now=$(date '+%FT%T')
  printf '%s %s\n' "$free" "$now" > "$MLT/.QUOTA_FREE_GB.tmp$$" && mv -f "$MLT/.QUOTA_FREE_GB.tmp$$" "$MLT/QUOTA_FREE_GB"
  printf '%s\t%s\t%s\t%s\t%s\t%ss\n' "$now" "$free" "$used" "$h" "$g" "$(( $(date +%s) - t0 ))" >> "$MLT/status/quota_history.tsv"
  echo "[quota] used ${used} GB (home ${h} MB, tmp_cpu2 ${g} MB), free ${free} GB, $(( $(date +%s) - t0 )) s"
else
  echo "[quota] du failed (home='$h' tmp='$g'); QUOTA_FREE_GB not changed"
fi
if [ ! -f "$MLT/QUOTA_STOP" ]; then
  sbatch --parsable --begin=now+12hours jobs/quota.sh && echo "[quota] next run queued (+12 h)"
fi
