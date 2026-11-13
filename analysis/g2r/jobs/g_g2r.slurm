#!/bin/bash
#SBATCH --job-name=g2r_g
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/g2r/logs/%x_%j.out
# analysis/g2r/jobs/g_g2r.slurm — copy of analysis/d1_pair/jobs/g.slurm (unchanged) for D1-G2R Phases F and T
# (docs/specs/D1G2R.md 8-1, 9) plus, after jobs/g_env.sh (which overwrites PYTHONPATH with site_si):
#   PYTHONPATH = analysis/g2r/site_g2r first (its sitecustomize runs site_si, then the D-5 tolerances) and
#   G2R_TOL_PATCH=1; analysis/g2r/f_preflight.py runs before the commands and exits 3 unless the job process and a spawned
#   interpreter have Si (1.11, 14, 2.10, 4), GATE_E_EH = 1e-4, TOL_DIP_AU = 1e-2 and every other tolerance unchanged.
# Submit from analysis/d1_pair (env.sh / g_env.sh are found via $SLURM_SUBMIT_DIR):
#   sbatch -p cpu1|cpu2 --time=48:00:00 --output=/gpfs/tmp_cpu2/yeseo1ee/g2r/logs/%x_%A_%a.out \
#     --export=ALL,PYENV=venv5,D1PAIR_PHASE=F,DEP_SCRATCH=..,DEP_RESULTS=..,DEP_G2_JOBS=.. \
#     ../g2r/jobs/g_g2r.slurm ../deploy/feat_lf.py slice @I 2         (' ::: ' separates commands, '@I' = array task)
set -uo pipefail
source "$SLURM_SUBMIT_DIR/jobs/env.sh"
if [ "${D1PAIR_PHASE:-G}" = F ] && [ "${PYENV:-reactot}" != venv5 ]; then echo "STOP: D1PAIR_PHASE=F needs PYENV=venv5 (tblite 0.7.0)"; exit 3; fi
source "$SLURM_SUBMIT_DIR/jobs/g_env.sh"
export PYTHONPATH=$(cd "$CODE/../g2r/site_g2r" && pwd):$PYTHONPATH G2R_TOL_PATCH=1
echo "[g2r_g] PYTHONPATH=$PYTHONPATH G2R_TOL_PATCH=$G2R_TOL_PATCH DEP_SCRATCH=${DEP_SCRATCH:-} DEP_RESULTS=${DEP_RESULTS:-}"
"$PY" ../g2r/f_preflight.py "${SLURM_JOB_NAME:-g2r_g}_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-x}" || exit 3
cmd=()
run() { echo "=== ${cmd[*]}  $(date '+%F %T')"; "$PY" "${cmd[@]}"; rc=$?; echo "=== exit $rc $(date '+%F %T')"; return $rc; }
for a in "$@"; do
  if [ "$a" = ":::" ]; then run || exit $?; cmd=(); else cmd+=("${a//@I/${SLURM_ARRAY_TASK_ID:-0}}"); fi
done
run; exit $?
