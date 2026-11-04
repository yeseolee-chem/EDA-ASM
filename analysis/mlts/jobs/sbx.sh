#!/bin/bash
# analysis/mlts/jobs/sbx.sh — run one short test command INSIDE the sandbox allocation (compute node), never on login:
#   bash jobs/sbx.sh <reactot|mlts|d1> <script.py> <args...>        (from analysis/mlts/)
# Env: MLT_SANDBOX (default: the running job named mlt_sandbox), SBX_C cores (4), SBX_T seconds (560), SBX_MEM (8G).
# srun --overlap returns when the command ends; the command is killed after SBX_T seconds.
set -uo pipefail
ENVN=$1; shift
SB=${MLT_SANDBOX:-$(squeue -u "$USER" -h -n mlt_sandbox -t R -o %i | head -1)}
[ -n "$SB" ] || { echo "no running mlt_sandbox job"; exit 2; }
HERE=$(cd "$(dirname "$0")/.." && pwd)
exec srun --jobid="$SB" --overlap -n1 -c"${SBX_C:-4}" --mem="${SBX_MEM:-8G}" timeout "${SBX_T:-560}" \
  bash -c "cd '$HERE' && export MLTENV='$ENVN' && source jobs/env.sh && \"\$PY\" \"\$@\"" _ "$@"
