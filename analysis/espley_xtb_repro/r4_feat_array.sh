#!/bin/bash
#SBATCH --job-name=r4_feat
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
# Phase 2: xtb features (xtb_slice.py --geom $GEOM, xtb 6.7.1, 1 core) for the 18 contiguous slices of the 5,260 rxns.
# Array element i = slices i and i+9, one after the other (--array=0-8: 9 queue slots, leaving room under the shared
# MaxSubmit=20). GEOM=g1 (default) -> $ESPLEY_OUT/slices_g1/ (g2 -> slices_g2/). An existing slice parquet is skipped
# (to redo one, delete it and resubmit). A slice with a rxn lacking a .done / .fail_* marker exits 1 and writes
# nothing, so submit only after the geometry step (Phase 1-2) has finished.
# Submit from analysis/espley_xtb_repro/ of the checkout:
#   sbatch --array=0-8 --partition=cpu1,cpu2 --export=ALL,GEOM=g1 --output=$R4_SCRATCH/logs/feat_g1_%a.%A.out r4_feat_array.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
GEOM=${GEOM:-g1}
case "$GEOM" in g1|g2) ;; *) echo "GEOM must be g1|g2, got '$GEOM'"; exit 2 ;; esac
export XTB_THREADS=1 OMP_NUM_THREADS=1
python -c "import pyarrow, networkx, numpy, pandas, dftd3, morfeus; print('py env ok')"
xtb_ver=$("$XTB_BIN" --version 2>&1 || true)
echo "$xtb_ver" | grep -i "xtb version" | head -1
rc=0
for s in ${SLURM_ARRAY_TASK_ID} $((SLURM_ARRAY_TASK_ID + 9)); do
    OUT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}/slices_$GEOM/slice_$(printf '%02d' "$s").parquet
    if [ -e "$OUT" ]; then echo "skip: $OUT exists"; continue; fi
    mkdir -p "$(dirname "$OUT")"
    echo "GEOM=$GEOM labels=$ESPLEY_LABELS G1_ROOT=$G1_ROOT -> $OUT"
    python xtb_slice.py "$s" 18 "$OUT" --geom "$GEOM" || rc=1
done
exit $rc
