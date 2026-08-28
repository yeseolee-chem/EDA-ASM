#!/bin/bash
# SPEC14 Step 3 — ORCA EDA single-points for 218 sampled reactions.
# Wave submission: 20-cap allows up to 20 submitted at once.
# Practical layout: --array=0-217%10 (10 running concurrent). Submit in
# two waves (0-19 and 20-217 chained), OR submit as 3 arrays split by
# partition. Simplest: 12 tasks per array, 18 arrays chained by afterany,
# OR one 20-task array + waves triggered from a launcher sbatch (§CLAUDE.md).
#
# NOTE ON EXECUTION: this script is a TEMPLATE. Actual submission strategy
# is decided at execution time (see logs/README on submit day). The wall
# limit here is 48h per CLAUDE.md — a single EDA calc runs in minutes to
# hours depending on atom count; a 48h array element accommodates the
# largest reactions in the L size bin.
#
#SBATCH --job-name=geomval
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation/logs/%A_%a.out

set -uo pipefail

BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation
mapfile -t DIRS < <(ls -d $BASE/inputs/rxn_*/ | sort)
D="${DIRS[$SLURM_ARRAY_TASK_ID]}"
D="${D%/}"

# Skip if already terminated normally (idempotent)
if [ -f "$D/eda.out" ] && grep -q "ORCA TERMINATED NORMALLY" "$D/eda.out"; then
    echo "SKIP $D (already complete)"
    exit 0
fi

# ORCA binary path — user's ORCA 6.1.1 install
export ORCA_BIN=/home1/yeseo1ee/orca_6_1_1_avx2/orca
export PATH=$(dirname $ORCA_BIN):$PATH
export LD_LIBRARY_PATH=$(dirname $ORCA_BIN):${LD_LIBRARY_PATH:-}

cd "$D"
echo "=== ORCA EDA $D $(date -Is) ==="
$ORCA_BIN eda.inp > eda.out 2> eda.err
rc=$?
echo "=== rc=$rc $(date -Is) ==="
exit $rc
