#!/bin/bash
#SBATCH --job-name=mltp_launch
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/mlts/logs/mltp_launch_%j.out
# analysis/mlts/jobs/pipeline.slurm — DEPLOY-V2 (MLTS) automatic stage launcher (pipeline.py; operator notes PIPELINE.md).
# CLAUDE.md option 2: the launcher is itself an sbatch job on a small CPU partition; nothing runs on the login node.
# The lead submits it ONCE, from analysis/mlts:      sbatch jobs/pipeline.slurm
# Each run, in this order:
#   1. stop (no pass, no successor) if $MLT/PIPELINE_PAUSE contains 'exit' or results/STOP_S.md exists (normal end)
#   2. queue the successor FIRST (same script, --begin=now+10minutes), so a crashing pass still has one; only when no
#      other pending mltp_launch exists (squeue retried 3 x; if it keeps failing the successor is queued anyway: a
#      duplicate is harmless — --dependency=singleton keeps launchers serial and the next run sees it pending).
#   3. self-check: pipeline.py and mlt_common.py tracked and equal to HEAD, else no pass (PIPELINE_ATTENTION says why)
#   4. one pass `pipeline.py once` (MLTENV=reactot, stdlib only; seconds; timeout 600 s); a non-zero exit (crash,
#      import error, timeout 124) is written to $MLT/PIPELINE_ATTENTION with the last output line.
# The launcher never deletes data and never cancels jobs.
# Test hooks: PIPELINE_NO_PASS=1 skips 3-4; PIPELINE_SKIP_SELFCHECK=1 skips 3; PIPELINE_TEST_PY=<exe> replaces env.sh/$PY.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
MLT=${MLT_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/mlts}
RES=${MLT_RESULTS:-results}
mkdir -p "$MLT/pipeline"
attn() {   # one line, overwritten atomically (the lead's hourly check reads it)
  printf '%s %s\n' "$(date '+%FT%T')" "$*" > "$MLT/.PIPELINE_ATTENTION.tmp$$" && mv -f "$MLT/.PIPELINE_ATTENTION.tmp$$" "$MLT/PIPELINE_ATTENTION"
}
echo "[mltp] host=$(hostname) job=${SLURM_JOB_ID:-} $(date '+%F %T')"
if [ -f "$MLT/PIPELINE_PAUSE" ] && [ "$(tr -d '[:space:]' < "$MLT/PIPELINE_PAUSE")" = "exit" ]; then
  echo "[mltp] PIPELINE_PAUSE = exit: no pass, no successor"
  exit 0
fi
if [ -f "$RES/STOP_S.md" ]; then
  echo "[mltp] $RES/STOP_S.md present (명령서 S: 멈춤): no pass, no successor"
  exit 0
fi
others=""
for k in 1 2 3; do
  if out=$(squeue -u "$USER" -h -n mltp_launch -t PD -o %i 2>/dev/null); then
    others=$(printf '%s\n' "$out" | grep -vx "${SLURM_JOB_ID:-none}" | grep -c .)
    break
  fi
  echo "[mltp] squeue failed (try $k)"
  [ "$k" -lt 3 ] && sleep 20
done
if [ -z "$others" ]; then
  echo "[mltp] squeue failed 3 times: queueing the successor anyway (singleton keeps launchers serial)"
  others=0
fi
if [ "${others:-0}" -eq 0 ]; then
  if succ=$(env -u MLT_JOB_T0 sbatch --parsable --begin=now+10minutes jobs/pipeline.slurm 2>&1); then
    echo "[mltp] successor $succ"
  else
    echo "[mltp] successor NOT queued: $succ (pipeline.py retries it and writes PIPELINE_ATTENTION)"
  fi
else
  echo "[mltp] successor already pending ($others)"
fi
if [ -n "${PIPELINE_NO_PASS:-}" ]; then
  echo "[mltp] PIPELINE_NO_PASS set: no pass"
  exit 0
fi
if [ -z "${PIPELINE_SKIP_SELFCHECK:-}" ]; then
  for f in pipeline.py mlt_common.py; do
    if ! git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$f" 2>/dev/null; then
      echo "[mltp] $f is not committed or modified since HEAD: pass skipped"
      attn "launcher pass skipped: analysis/mlts/$f is not committed or modified since HEAD (the launcher runs only committed code; commit it and the next run passes)"
      exit 0
    fi
  done
fi
if [ -n "${PIPELINE_TEST_PY:-}" ]; then
  PY=$PIPELINE_TEST_PY
else
  export MLTENV=reactot
  source jobs/env.sh || { attn "launcher: jobs/env.sh failed (MLTENV=reactot)"; exit 0; }
fi
timeout 600 "$PY" pipeline.py once 2>&1 | tee "$MLT/pipeline/last_pass.out"
rc=${PIPESTATUS[0]}
echo "[mltp] pass exit $rc $(date '+%F %T')"
if [ "$rc" -ne 0 ]; then
  attn "launcher pass exit $rc$([ "$rc" -eq 124 ] && echo ' (timeout 600 s)'): $(grep -v '^\s*$' "$MLT/pipeline/last_pass.out" 2>/dev/null | tail -n 1 | cut -c1-200) — log $MLT/logs/mltp_launch_${SLURM_JOB_ID:-}.out"
fi
exit 0
