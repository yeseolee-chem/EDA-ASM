# r4_env.sh — sourced by every rev 4 and rev 5 sbatch script (conda env, ORCA/MPI, xtb 6.7.1, paths). The r5_*.sh
# scripts export R5_SCRATCH themselves (default /gpfs/tmp_cpu2/yeseo1ee/espley_rev5).
# CODE = the directory the job was submitted from (the checkout), never a hard-coded clone.
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate ${ESPLEY_ENV:-reactot}
export CODE=${ESPLEY_CODE:-$SLURM_SUBMIT_DIR}
export REPO=$(cd "$CODE/../.." && pwd)
export ESPLEY_LABELS=${ESPLEY_LABELS:-$REPO/labels_all.json}
export ORCA_DIR=/home1/yeseo1ee/orca_6_1_1_avx2
export ORCA_BIN=$ORCA_DIR/orca
export MPI_ROOT=/usr/mpi/gcc/openmpi-4.1.7a1
export PATH=$ORCA_DIR:$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$ORCA_DIR:$MPI_ROOT/lib64:$MPI_ROOT/lib:${LD_LIBRARY_PATH:-}
# as label_true/scripts/smd_relabel_worker.sh; SLURM gives N CPUs as 1 task, so mpirun must oversubscribe,
# and several ORCA runs share a node, so no core binding
export OMPI_MCA_rmaps_base_oversubscribe=1 OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_btl=self,vader,tcp OMPI_MCA_pml=ob1 OMPI_MCA_coll_hcoll_enable=0 UCX_TLS=tcp,self,sm
export XTB_BIN=/home1/yeseo1ee/xtb-dist/bin/xtb
export XTBPATH=/home1/yeseo1ee/xtb-dist/share/xtb XTBHOME=/home1/yeseo1ee/xtb-dist
export G1_ROOT=${G1_ROOT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1}
export R4_SCRATCH=${R4_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev4}
export D1_SNAPSHOT=${D1_SNAPSHOT:-$R4_SCRATCH/d1_snapshot_d573111e}
cd "$CODE"
echo "[r4_env] host=$(hostname) code=$CODE repo=$REPO git=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
