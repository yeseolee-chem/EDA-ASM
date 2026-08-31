#!/bin/bash
# SPEC16rev Step 2 — per-rxn ORCA runner (array element).
# 5 SPEs per rxn, each with idempotent skip; rm -f before rerun to prevent
# FSPE-append leaks (spec15 rxn 1479 lesson).
#
#SBATCH --job-name=b3full
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full/logs/%A_%a.out

set -uo pipefail

BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full

mapfile -t DIRS < <(ls -d $BASE/inputs/rxn_*/ | sort)
BASE_OFFSET=${BASE_OFFSET:-0}
IDX=$((BASE_OFFSET + SLURM_ARRAY_TASK_ID))
D="${DIRS[$IDX]}"
D="${D%/}"
[ -z "$D" ] && exit 0
rid=$(basename "$D")

export ORCA_BIN=/home1/yeseo1ee/orca_6_1_1_avx2/orca
export PATH=$(dirname $ORCA_BIN):$PATH
export LD_LIBRARY_PATH=$(dirname $ORCA_BIN):${LD_LIBRARY_PATH:-}

cd "$D"
echo "=== $rid $(date -Is) ==="

for STEM in eda frag1_dist frag2_dist frag1_rel frag2_rel; do
    [ -f "$STEM.inp" ] || continue
    # idempotent skip
    if [ -f "$STEM.out" ] && grep -q "ORCA TERMINATED NORMALLY" "$STEM.out"; then
        echo "  skip $STEM"
        continue
    fi
    # Remove stale output to prevent FSPE-count anomalies from partial reruns
    rm -f "$STEM.out" "$STEM.err"
    $ORCA_BIN "$STEM.inp" > "$STEM.out" 2> "$STEM.err"
    rc=$?
    echo "  $STEM rc=$rc"
done

echo "=== end $(date -Is) ==="
