#!/bin/bash
#SBATCH --job-name=adf_asr_cand
#SBATCH --output=Validate/refrag/candidate_logs/%x-%j.log
#SBATCH --error=Validate/refrag/candidate_logs/%x-%j.log
#SBATCH --partition=cpu1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Submit a single ADF candidate run on a compute node.
#
# Usage:
#   sbatch --export=RXN_ID=<synthetic_rid> Validate/refrag/slurm_run_candidate.sbatch
#
# For batch submission, use a wrapper script that loops over rids:
#   for rid in $(ls Validate/refrag/candidates_stage5a/per_reaction/); do
#     out="Validate/refrag/candidate_results/${rid}.json"
#     [ -f "$out" ] && continue
#     sbatch --export=RXN_ID="$rid" Validate/refrag/slurm_run_candidate.sbatch
#   done

set -eo pipefail
RXN_ID="${RXN_ID:?RXN_ID must be passed via --export=RXN_ID=<rid>}"
ROOT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
mkdir -p "$ROOT/Validate/refrag/candidate_logs"

source "$HOME/ams2026.103/amsbashrc.sh"
# amsbashrc only sets AMSHOME/AMSBIN; the LD_LIBRARY_PATH setup lives in
# setenv.sh (added to PATH/LD_LIBRARY_PATH by amspython only). Source it
# explicitly here so PLAMS's child subprocesses (ams.exe) inherit the right
# env on compute nodes — system libstdc++ is too old (6.0.25 vs needed 6.0.27).
. "$AMSBIN/setenv.sh"
# Belt-and-suspenders: also prepend the bundled libs.
export LD_LIBRARY_PATH="$AMSBIN/lib/libstdc++:$AMSBIN/lib:$AMSBIN/intelmpi/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# Force NSCM=1 (serial) and SERIAL_PROGRAM=1 so AMS does NOT spawn srun from
# inside the SLURM allocation. The ams shell wrapper takes the direct-exec
# branch only when SERIAL_PROGRAM=1, otherwise inside SLURM it uses srun and
# the resulting nested-step env strips/breaks LD_LIBRARY_PATH.
export NSCM=1
export SERIAL_PROGRAM=1
unset SLURM_NTASKS SLURM_NPROCS SLURM_TASKS_PER_NODE
"$AMSBIN/amspython" Validate/refrag/run_candidate_wrapper.py \
  --rxn_id "$RXN_ID"
