#!/bin/bash
# SPEC17rev2 Step 10 — per-rxn ORCA runner (5 SPEs in parallel).
# Called as an array element; DIR list resolved from channel_impact/inputs/.
# Same OMPI hardening as spec16rev/02_submit.sh (OB1 PML, no hcoll, TCP UCX).
#
#SBATCH --job-name=otfm10
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=40G
#SBATCH --exclude=n045
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train/logs/10_%A_%a.out

set -uo pipefail
BASE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train

mapfile -t DIRS < <(ls -d $BASE/channel_impact/inputs/rxn_*/ | sort)
BASE_OFFSET=${BASE_OFFSET:-0}
IDX=$((BASE_OFFSET + SLURM_ARRAY_TASK_ID))
D="${DIRS[$IDX]}"
D="${D%/}"
[ -z "$D" ] && exit 0
rid=$(basename "$D")

export ORCA_BIN=/home1/yeseo1ee/orca_6_1_1_avx2/orca
export MPI_ROOT=/usr/mpi/gcc/openmpi-4.1.7a1
export PATH=$MPI_ROOT/bin:$(dirname $ORCA_BIN):$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib64:$(dirname $ORCA_BIN):${LD_LIBRARY_PATH:-}
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_pml=ob1
export OMPI_MCA_coll_hcoll_enable=0
export UCX_TLS=tcp,self,sm

cd "$D"
echo "=== $rid $(date -Is) ==="

pids=()
for STEM in eda frag1_dist frag2_dist frag1_rel frag2_rel; do
    [ -f "$STEM.inp" ] || continue
    if [ -f "$STEM.out" ] && grep -q "ORCA TERMINATED NORMALLY" "$STEM.out"; then
        echo "  skip $STEM"
        continue
    fi
    rm -f "$STEM.out" "$STEM.err"
    ( $ORCA_BIN "$STEM.inp" > "$STEM.out" 2> "$STEM.err"; echo "  $STEM rc=$?" ) &
    pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done

# Cleanup: keep .inp/.out/.err/.gbw; drop densities/tmp per b3lyp_full policy.
if grep -l "ORCA TERMINATED NORMALLY" *.out >/dev/null 2>&1; then
    rm -f *.densities *.cpcm *.tmp *.tmp.* *_property.txt 2>/dev/null || true
fi
