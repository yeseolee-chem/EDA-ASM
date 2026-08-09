#!/bin/bash
# ORCA EDA-NOCV SLURM array runner for the tt (three_two_cycloaddition)
# labeling task. Runs one reaction per array task.
#
# EDIT BEFORE SUBMITTING at KISTI (or any new cluster):
#   1. #SBATCH --partition=<KISTI_CPU_PARTITION>
#   2. #SBATCH --account=<YOUR_ALLOCATION_CODE>    # KISTI requires this
#   3. #SBATCH --array=0-3503%<K>    where K = min(N_RUN, N_SUBMIT-1)
#   4. ORCA_BIN=<PATH_TO_ORCA_6.x>
#   5. OpenMPI setup, if using ORCA MPI parallel
#
# The package layout assumed:
#   ./data/rxn_XXXX/eda.inp       (already built, do not modify)
#   ./data/manifest.parquet       (rxn_id ordering)
#   ./data/kisti_final_ids.json   (canonical rxn_id list)

#SBATCH --job-name=tt_eda
#SBATCH --time=48:00:00
#SBATCH --partition=REPLACE_ME_PARTITION
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3503%10
#SBATCH --output=logs/orca.%A_%a.log

set -uo pipefail

# ---- User configuration ----
ORCA_BIN="${ORCA_BIN:-$HOME/orca_6_1_1_avx2/orca}"   # override via env
PKG_ROOT="${PKG_ROOT:-$(pwd)}"                        # location of this package
# ---------------------------

mkdir -p "$PKG_ROOT/logs"

# Resolve rxn_id from the canonical ID list (position -> rxn_id)
RXN_ID=$(python3 -c "
import json, sys
ids = json.load(open('$PKG_ROOT/data/kisti_final_ids.json'))['reaction_ids']
idx = $SLURM_ARRAY_TASK_ID
if idx >= len(ids):
    sys.stderr.write(f'array index {idx} out of range (max {len(ids)-1})\n')
    sys.exit(2)
print(ids[idx])
")
if [ -z "$RXN_ID" ]; then
    echo "FAIL: could not resolve rxn_id for array idx $SLURM_ARRAY_TASK_ID"
    exit 2
fi

RXN_DIR="$PKG_ROOT/data/rxn_$(printf '%04d' $RXN_ID)"
if [ ! -d "$RXN_DIR" ]; then
    echo "SKIP: $RXN_DIR does not exist"
    exit 0
fi

cd "$RXN_DIR"

# Idempotency: skip if already succeeded
if [ -f "eda.out" ] && grep -q "ORCA TERMINATED NORMALLY" eda.out; then
    echo "SKIP: rxn_$RXN_ID already complete"
    exit 0
fi

echo "=== rxn_$RXN_ID  host=$(hostname)  start=$(date -Is) ==="

"$ORCA_BIN" eda.inp > eda.out 2> eda.err
STATUS=$?

echo "=== rxn_$RXN_ID  rc=$STATUS  end=$(date -Is) ==="
if grep -q "ORCA TERMINATED NORMALLY" eda.out; then
    echo "OK"
    # Cleanup: keep eda.inp / eda.out / eda.err / eda.property.txt only
    # These files can be several MB each × 3500 reactions — remove aggressively.
    rm -f eda.cpcm eda.cpcm_corr eda.smd
    rm -f eda_frag{1,2}.cpcm eda_frag{1,2}.cpcm_corr
    rm -f eda_frag{1,2}.densitiesinfo eda_frag{1,2}.inp
    rm -f eda_frag{1,2}.out eda_frag{1,2}.property.txt
    rm -f eda.densities eda.densitiesinfo
    rm -f eda*.gbw eda.nocv.gbw
    rm -f eda*.bas0 eda*.bas1 eda*.bas2 eda*.bas3 eda*.bas4 eda*.bas5
    rm -f eda*.bibtex
    rm -f eda*.tmp eda*.tmpg eda.int.tmp
    exit 0
else
    echo "FAIL — tail of eda.out:"
    tail -30 eda.out 2>&1
    echo "tail of eda.err:"
    tail -10 eda.err 2>&1
    exit $STATUS
fi
