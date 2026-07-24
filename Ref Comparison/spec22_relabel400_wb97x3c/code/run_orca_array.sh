#!/bin/bash
#SBATCH --job-name=espley_s22_orca
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/orca_%A_%a.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/orca_%A_%a.err

# Array task: run one ORCA job selected by SLURM_ARRAY_TASK_ID from a manifest CSV.
# Usage:  sbatch --array=0-N%K --export=MANIFEST=<path> run_orca_array.sh
# CLAUDE.md: --time=48:00:00 fixed; idempotent — skip if <jobtype>.out ends with `****ORCA TERMINATED NORMALLY****`.

set -euo pipefail

MANIFEST="${MANIFEST:?MANIFEST env var required}"
IDX="${SLURM_ARRAY_TASK_ID:?not a SLURM array task}"

# Read one CSV row (1-indexed relative to header)
row=$(awk -F',' -v i=$((IDX + 2)) 'NR==i' "$MANIFEST")
if [[ -z "$row" ]]; then
  echo "[error] no row at index $IDX in $MANIFEST" >&2
  exit 1
fi

# CSV columns: reaction_id, sub_source, reaction_number, jobtype, workdir, input, out
IFS=',' read -r RID SUB RN JOBTYPE WORKDIR INPUT OUT <<< "$row"

echo "=== [$(date -Is)] task=$IDX rid=$RID jobtype=$JOBTYPE workdir=$WORKDIR ==="
cd "$WORKDIR"

# Idempotent skip
if [[ -f "$OUT" ]] && grep -q '\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*' "$OUT" 2>/dev/null; then
  echo "[skip] $OUT already terminated normally"
  exit 0
fi

# Ensure OpenMPI 4.1.5 (ABI-compatible with ORCA's 4.1.8 build target).
# Explicit path — module load has been unreliable in compute-node context.
OPENMPI=/opt/ohpc/pub/openmpi/4.1.5
ORCA=/home1/yeseo1ee/orca_6_1_1_avx2/orca
export PATH="$OPENMPI/bin:/home1/yeseo1ee/orca_6_1_1_avx2:$PATH"
export LD_LIBRARY_PATH="$OPENMPI/lib:/home1/yeseo1ee/orca_6_1_1_avx2:${LD_LIBRARY_PATH:-}"

INP_BASENAME=$(basename "$INPUT")
"$ORCA" "$INP_BASENAME" > "$(basename "$OUT")" 2>&1 || true

if grep -q '\*\*\*\*ORCA TERMINATED NORMALLY\*\*\*\*' "$OUT" 2>/dev/null; then
  echo "[ok] terminated normally"
else
  echo "[fail] $OUT does not end with normal termination"
  tail -30 "$OUT" || true
fi
