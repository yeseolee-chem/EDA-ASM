#!/bin/bash
# Phase 3 — ADF EDA at TS for ONE substrate (passed via --export=ID=<id>)
# Matches the ADF_500 pattern that worked: NSCM=1 single-process, no SLURM srun.
#SBATCH --nodes=1
#SBATCH --partition=cpu1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH -o ./_out/%x.%j.out
#SBATCH -e ./_err/%x.%j.err

set -uo pipefail
: "${ID:?ID must be set via --export=ID=<substrate_id>}"

# Activate AMS only (avoid orca-env / openmpi/4.1.5 to prevent srun double-wrap)
source "$HOME/ams2026.103/amsbashrc.sh"
export NSCM=1                  # single-process ADF (proven pattern from ADF_500)
export OMP_NUM_THREADS=1
unset SLURM_CPU_BIND SLURM_TASK_PID SLURM_NTASKS SLURM_PROCID 2>/dev/null || true

PROJECT="/gpfs/home1/yeseo1ee/projects/v1-claisen-asr"
WORKDIR="/gpfs/tmp_cpu2/yeseo1ee_plams/${ID}_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$PROJECT"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
log "=== ADF EDA Phase 3, ID=$ID ==="
log "host=$(hostname)  NSCM=$NSCM  WORKDIR=$WORKDIR"
log "ams=$AMSBIN/ams"

"$AMSBIN/amspython" scripts/run_eda.py --id "$ID" --workdir "$WORKDIR"
rc=$?
log "amspython exit=$rc"

# Move PLAMS workdir back into project for audit (only essential rkfs)
ARCHIVE="runs/$ID/eda/plams_workdir"
mkdir -p "$ARCHIVE"
# Save only the .rkf files (not all temp); they're audit trail
find "$WORKDIR" -name '*.rkf' -exec cp -p {} "$ARCHIVE/" \; 2>/dev/null || true
log "rkf files archived to $ARCHIVE"

# Optional: clean up tmp workdir (uncomment when confident)
# rm -rf "$WORKDIR"

log "=== Phase 3 DONE for $ID, rc=$rc ==="
exit $rc
