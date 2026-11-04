#!/bin/bash
# analysis/mlts/jobs/sbx.sh — run one short test command INSIDE a sandbox allocation (compute node), never on login:
#   bash jobs/sbx.sh <reactot|mlts|d1> <script.py> <args...>        (from analysis/mlts/)
# CPU sandbox = running job named mlt_sandbox; GPU sandbox (SBX_GPU=1) = running job named mlt_gsandbox (1 GPU).
# Env: MLT_SANDBOX (job id override), SBX_C cores (4), SBX_T seconds (560), SBX_MEM (8G), SBX_GPU (unset = CPU).
# srun --overlap returns when the command ends; the command is killed after SBX_T seconds.
set -uo pipefail
ENVN=$1; shift
NAME=mlt_sandbox; GRES=()
if [ -n "${SBX_GPU:-}" ]; then NAME=mlt_gsandbox; GRES=(--gres=gpu:1); fi
SB=${MLT_SANDBOX:-$(squeue -u "$USER" -h -n "$NAME" -t R -o %i | head -1)}
# CPU tests fall back to the GPU sandbox (16 CPUs) when no CPU sandbox runs (CPU sandbox cancelled 11-04, user OK)
[ -n "$SB" ] || [ -n "${SBX_GPU:-}" ] || SB=$(squeue -u "$USER" -h -n mlt_gsandbox -t R -o %i | head -1)
[ -n "$SB" ] || { echo "no running $NAME job"; exit 2; }
HERE=$(cd "$(dirname "$0")/.." && pwd)
exec srun --jobid="$SB" --overlap -n1 -c"${SBX_C:-4}" --mem="${SBX_MEM:-8G}" "${GRES[@]}" timeout "${SBX_T:-560}" \
  bash -c "cd '$HERE' && export MLTENV='$ENVN' && source jobs/env.sh && \"\$PY\" \"\$@\"" _ "$@"
