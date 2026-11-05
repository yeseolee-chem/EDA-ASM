#!/bin/bash
#SBATCH --job-name=mlt_status
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/mlts/logs/mlt_status_%j.out
# analysis/mlts/jobs/status.sh — one status snapshot for the lead (user 2026-11-05: the login node runs only one-shot
# sbatch and git; every check happens here). Writes $MLT/status/latest.txt (atomic) and keeps one copy per job.
# Submit from analysis/mlts:  sbatch jobs/status.sh
set -uo pipefail
MLT=/gpfs/tmp_cpu2/yeseo1ee/mlts
RES=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-mlts/analysis/mlts/results
OUT=$MLT/status
mkdir -p "$OUT"
T=$OUT/.latest.tmp$$
{
  echo "== status $(date '+%FT%T') job ${SLURM_JOB_ID:-} host $(hostname)"
  echo "== QUOTA_FREE_GB file (du estimate by jobs/quota.sh): $(cat $MLT/QUOTA_FREE_GB 2>/dev/null)"
  tail -2 $MLT/status/quota_history.tsv 2>/dev/null
  echo "== squeue"
  squeue -u "$USER" -h -o "%i %j %T %M %R" 2>&1 | grep -v "mlt_status"
  echo "== PIPELINE_ATTENTION: $(cat $MLT/PIPELINE_ATTENTION 2>/dev/null | cut -c1-300)"
  echo "== PIPELINE_PAUSE: $([ -f $MLT/PIPELINE_PAUSE ] && echo "present: $(cat $MLT/PIPELINE_PAUSE)" || echo absent)"
  echo "== open STOP files"
  ls $RES/STOP_*.md 2>/dev/null | grep -v resolved || echo none
  echo "== PIPELINE_LOG tail"
  tail -6 $RES/PIPELINE_LOG.md 2>/dev/null | cut -c1-260
  echo "== FT train progress"
  for f in $(ls -t $MLT/logs/ft_p_*.out $MLT/logs/ftfull_*.out $MLT/logs/ft_abl_*.out $MLT/logs/ftall_now_*.out 2>/dev/null | head -14); do
    echo "-- $(basename $f): $(grep '\[train\] epoch' $f | tail -1 | cut -c1-170)"; done
  for f in $(ls -t $MLT/logs/ft_spec_data_*.out 2>/dev/null | head -1); do
    echo "-- $(basename $f): $(tail -2 $f | cut -c1-200 | tr '\n' ' ')"; done
  echo "== leak recheck (mlt_audit_leak2)"
  for f in $(ls -t $MLT/logs/mlt_audit_leak2_*.out 2>/dev/null | head -1); do tail -4 $f | cut -c1-260; done
  echo "== FT-2 summaries"
  for f in $(ls $MLT/ft2/*/*/summary.json 2>/dev/null); do
    echo "-- ${f#$MLT/ft2/}: $(python3 -c "import json,sys; a=json.load(open(sys.argv[1])).get('all',{}); print({k:a.get(k) for k in ('n_attempted','same_ts_frac','dd_median_A','dd_p90_A','rmsd_median_A','nimag1_frac')})" $f 2>&1 | cut -c1-260)"; done
} > "$T" 2>&1
cp "$T" "$OUT/status_${SLURM_JOB_ID:-x}.txt"
mv -f "$T" "$OUT/latest.txt"
