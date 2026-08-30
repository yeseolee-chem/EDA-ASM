#!/bin/bash
# SPEC15 Step 3 — ORCA submission. Each array element does up to 5 SPEs:
#   eda, frag1_dist, frag2_dist, frag1_rel, frag2_rel  (each with own idempotent skip).
#
# NOTE ON EXECUTION: this is a TEMPLATE. Actual submission strategy (single
# array vs waves) decided at execution time. For 200 rxns × ~5 SPEs each,
# an orchestrator similar to analysis/geometry_validation/orch_geomval.sh
# with continuous-fill would be appropriate.
#
#SBATCH --job-name=b3relabel
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_relabel/logs/%A_%a.out

set -uo pipefail

BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_relabel
GEOM=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation

mapfile -t DIRS < <(ls -d $BASE/inputs/rxn_*/ | sort)
D="${DIRS[$SLURM_ARRAY_TASK_ID]}"
D="${D%/}"
rid=$(basename $D)

export ORCA_BIN=/home1/yeseo1ee/orca_6_1_1_avx2/orca
export PATH=$(dirname $ORCA_BIN):$PATH
export LD_LIBRARY_PATH=$(dirname $ORCA_BIN):${LD_LIBRARY_PATH:-}

cd "$D"
echo "=== $rid $(date -Is) ==="

for STEM in eda frag1_dist frag2_dist frag1_rel frag2_rel; do
    [ -f "$STEM.inp" ] || continue
    # 1) reuse spec14 result if it's the complex EDA and there's a good output
    if [ "$STEM" = "eda" ]; then
        SRC=$GEOM/results_218/$rid/eda.out
        if [ -f "$SRC" ] && grep -q "ORCA TERMINATED NORMALLY" "$SRC"; then
            cp "$SRC" "eda.out"
            echo "  reuse spec14 eda.out for $rid"
            continue
        fi
    fi
    # 2) idempotent skip if already done
    if [ -f "$STEM.out" ] && grep -q "ORCA TERMINATED NORMALLY" "$STEM.out"; then
        echo "  skip $STEM (already done)"
        continue
    fi
    # 3) run
    $ORCA_BIN "$STEM.inp" > "$STEM.out" 2> "$STEM.err"
    rc=$?
    echo "  $STEM rc=$rc"
done

echo "=== end $(date -Is) ==="
