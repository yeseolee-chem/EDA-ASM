#!/bin/bash
#SBATCH --job-name=app_pilot
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/mlts/logs/%x_%j.out
# A'' pilot (user 11-05 18:0x (2), results/DECISIONS.md), one manual job OUTSIDE the frozen launcher (app_pilot.py).
# Submit from analysis/mlts (the job starts there; or pass --chdir=<worktree>/analysis/mlts):
#   pilot: sbatch -p cpu2 jobs/app_pilot.slurm                         select -> prep -> run (8 ORCA x 8 ranks) -> report
#   smoke: sbatch -p cpu1,cpu2 -J app_smoke -c 8 --mem=32G jobs/app_pilot.slurm smoke
#          (select -> prep --smoke -> smoke: the smallest selected reaction, MaxIter 2, no Freq -> $MLT/app_pilot/SMOKE.txt)
# Each step runs in its own environment (jobs/env.sh): select / prep MLTENV=mlts (pandas, torch, AIMNet2 on the CPU);
# run / smoke / report MLTENV=d1 (D1 venv + opt.d1() ORCA environment: OpenMPI 4.1.8f without XPMEM, OMPI_MCA_pml=ob1,
# OMPI_MCA_coll_hcoll_enable=0, UCX_TLS=tcp,self,sm, binding none). Idempotent: after a wall clip resubmit the same
# line (finished reactions are skipped). Progress: $MLT/app_pilot/SUMMARY.txt.
set -uo pipefail
[ -f app_pilot.py ] && [ -f jobs/env.sh ] || { echo "start the job in analysis/mlts (cwd $(pwd))"; exit 2; }
export MLT_JOB_T0=${MLT_JOB_T0:-$(date +%s)}
MODE=${1:-pilot}
step() {
  local envn=$1; shift
  echo "=== [$envn] $*  $(date '+%F %T')"
  ( export MLTENV=$envn; source jobs/env.sh || exit 2; exec "$PY" "$@" )
  local r=$?
  echo "=== exit $r $(date '+%F %T')"
  return $r
}
echo "[app_pilot] mode=$MODE host=$(hostname) job=${SLURM_JOB_ID:-} cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?} tmp=${TMPDIR:-/tmp} git=$(git rev-parse --short HEAD 2>/dev/null) $(date '+%F %T')"
case "$MODE" in
  smoke)
    step mlts app_pilot.py select || exit $?
    step mlts app_pilot.py prep --smoke || exit $?
    step d1 app_pilot.py smoke
    exit $? ;;
  pilot)
    step mlts app_pilot.py select || exit $?
    step mlts app_pilot.py prep || exit $?
    step d1 app_pilot.py run; rc=$?
    [ $rc -ne 3 ] || exit 3
    step d1 app_pilot.py report
    exit $rc ;;
  *)
    echo "unknown mode $MODE (smoke | pilot)"; exit 2 ;;
esac
