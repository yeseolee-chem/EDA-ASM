#!/bin/bash
#SBATCH --job-name=g2r_d1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/g2r/logs/%x_%j.out
# analysis/g2r/jobs/cpu_g2r.slurm — D1-G2R steps in the D1 run env (= analysis/deploy/jobs/cpu_d1.slurm: d1_run/nurion/
# env.sh -> venv python 3.9.5 + autodE 1.4.5 + rdkit, xtb 6.7.1, ORCA 6.1.1, OpenMPI 4.1.8f) with
# D1G2_SCRATCH=$G2R, so g2_common.G2 (and g2_autode's ts_templates folder) is the new root, never $G2.
# Submit from the repository root:
#   sbatch -p cpu1|cpu2 --time=48:00:00 [--cpus-per-task=N --mem=M -J name] analysis/g2r/jobs/cpu_g2r.slurm \
#          analysis/g2r/<script>.py <args...> [ ::: <script>.py <args...> ]
# ' ::: ' runs the next command only if the previous one exited 0. "@I" -> $SLURM_ARRAY_TASK_ID.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
[ $# -gt 0 ] || { echo "usage: analysis/g2r/jobs/cpu_g2r.slurm <script.py> <args...> [ ::: <script.py> ... ]"; exit 2; }
source /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh
PY="${D1_PY:?}"
export D1G2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/g2r PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DEP_JOB_T0=$(date +%s)
echo "[g2r_d1] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}" \
     "git=$(git rev-parse --short HEAD 2>/dev/null) D1G2_SCRATCH=$D1G2_SCRATCH xtb=${XTB_BIN:-?} $($PY -V 2>&1) $(date '+%F %T')"
cmd=(); first=0
run() { echo "=== ${cmd[*]}  $(date '+%F %T')"; "$PY" "${cmd[@]}"; rc=$?; echo "=== exit $rc $(date '+%F %T')"
        [ $first -eq 0 ] && [ $rc -ne 0 ] && first=$rc; return $rc; }
for a in "$@"; do
  if [ "$a" = ":::" ]; then run || exit $first; cmd=(); else cmd+=("${a//@I/${SLURM_ARRAY_TASK_ID:-@I}}"); fi
done
[ ${#cmd[@]} -gt 0 ] && run
exit $first
