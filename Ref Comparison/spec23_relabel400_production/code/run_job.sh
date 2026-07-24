#!/bin/bash
#SBATCH --job-name=s23_orca
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/s23_%A_%a.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/s23_%A_%a.err

# spec23 production ORCA runner. One task per (rid, jobtype) row in $MANIFEST.
# Idempotent, restart-aware. jobtype ∈ {eda, fragA_opt, fragB_opt}.

set -uo pipefail

MANIFEST="${MANIFEST:?MANIFEST env var required}"
NPROCS="${NPROCS:-4}"
MAXCORE_MB="${MAXCORE_MB:-3500}"
IDX="${SLURM_ARRAY_TASK_ID:?not a SLURM array task}"

# CSV: reaction_id, sub_source, reaction_number, jobtype, workdir, input, out.
# Strip any \r that snuck in from a CRLF-emitting CSV writer (wave manifests
# from Python's csv.DictWriter default to \r\n).
row=$(awk -F',' -v i=$((IDX + 2)) 'NR==i' "$MANIFEST" | tr -d '\r')
if [[ -z "$row" ]]; then
  echo "[error] no row at index $IDX in $MANIFEST" >&2
  exit 1
fi
IFS=',' read -r RID SUB RN JOBTYPE WORKDIR INPUT OUT <<< "$row"

echo "=== [$(date -Is)] task=$IDX rid=$RID jobtype=$JOBTYPE workdir=$WORKDIR ==="
cd "$WORKDIR"

# Idempotent skip
if [[ -f "$OUT" ]] && grep -q '\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*' "$OUT" 2>/dev/null; then
  echo "[skip] $OUT already terminated normally"
  exit 0
fi

# --- MPI setup ------------------------------------------------------------
# /opt/ohpc/pub/openmpi/4.1.5 has a broken mpirun (unknown option -np). The
# working OpenMPI at this site is /opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.6,
# combined with /opt/ohpc/pub/libs/hwloc for libhwloc.so.15.
OPENMPI=/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.6
HWLOC=/opt/ohpc/pub/libs/hwloc
ORCA_ROOT=/home1/yeseo1ee/orca_6_1_1_avx2

export PATH="$OPENMPI/bin:$ORCA_ROOT:$PATH"
export LD_LIBRARY_PATH="$OPENMPI/lib:$HWLOC/lib:$ORCA_ROOT:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_mca_base_env_list="LD_LIBRARY_PATH;PATH"

echo "[env] mpirun: $(which mpirun) $(mpirun --version 2>&1 | head -1)"
echo "[env] orca:   $(which orca)"

# --- Restart semantics per job type ---------------------------------------
INP_BASENAME=$(basename "$INPUT")
OUT_BASENAME=$(basename "$OUT")

if [[ "$JOBTYPE" == "eda" ]]; then
  # EDA has no mid-module checkpoint; the surviving .gbw acts as SCF warm start.
  # Wipe scratch from previous attempt but keep .gbw for guess reuse.
  rm -f -- *.tmp *.densities *.bas[0-9]* 2>/dev/null || true
else
  # fragA_opt / fragB_opt — resume from last accepted xyz if present.
  BASE="${JOBTYPE}"
  RESTART_XYZ="${BASE}.xyz"
  if [[ -s "$RESTART_XYZ" ]] && [[ ! -s "${BASE}_start.xyz" ]]; then
    cp -f "$INPUT" "${INP_BASENAME}.first"
    cp -f "$RESTART_XYZ" "${BASE}_resume.xyz"
    # Rewrite input coord block from the surviving optimised geom.
    python3 - <<'PY_REWRITE'
import os, re, sys
inp   = os.environ["INPUT"]
rxyz  = os.environ.get("BASE","frag") + "_resume.xyz"
# Load resume xyz
with open(rxyz) as f:
    lines = f.read().splitlines()
n = int(lines[0].strip())
atoms = lines[2:2+n]
# Read original input, replace the atom block between `* xyz ... *`
with open(inp) as f:
    src = f.read()
m = re.search(r"^\* *xyz.*?\n(.*?)\n\*\s*$", src, re.M | re.S)
if not m:
    sys.exit(0)
new_block = "\n".join(f"  {ln}" for ln in atoms)
new_src = src[:m.start(1)] + new_block + src[m.end(1):]
with open(inp, "w") as f:
    f.write(new_src)
PY_REWRITE
  fi
  rm -f -- *.tmp *.densities *.bas[0-9]* 2>/dev/null || true
fi

# --- Run ORCA -------------------------------------------------------------
cleanup() {
  echo "[$(date -Is)] cleanup: removing scratch"
  rm -f -- *.tmp *.densities *.bas[0-9]* *.SHARKINP.tmp 2>/dev/null || true
}
trap cleanup EXIT USR1

"$ORCA_ROOT/orca" "$INP_BASENAME" > "$OUT_BASENAME" 2>&1

# --- Post-check -----------------------------------------------------------
if grep -q '\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*' "$OUT" 2>/dev/null; then
  echo "[ok] $OUT terminated normally"
  # G23-E: verify MPI was actually engaged if we asked for it
  if [[ "$NPROCS" -gt 1 ]]; then
    if ! grep -qE "MPI parallel run.*$NPROCS|Program running with $NPROCS parallel processes|MPI PROCESSES.*$NPROCS" "$OUT"; then
      echo "[warn] output does not confirm $NPROCS parallel processes"
    fi
  fi
  exit 0
else
  echo "[fail] $OUT does not end with normal termination"
  tail -20 "$OUT" || true
  exit 1
fi
