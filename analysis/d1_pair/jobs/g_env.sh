# analysis/d1_pair/jobs/g_env.sh — environment of the D1PAIR G / F jobs (sourced after jobs/env.sh).
# Values = d1_g1.g_env() (d1_g1.check_env exits 3 on any difference) + the ORCA / MPI / xtb settings of rev 4 r4_env.sh.
S=$D1PAIR_SCRATCH
export ESPLEY_LABELS=$S/G/labels_d1_g.json ESPLEY_PROF=$S/prof ESPLEY_SMILES=$S/G/smiles_d1.csv ESPLEY_META=$S/G/input_meta_d1.csv
export ESPLEY_PARTITION=label G1_ROOT=$S/g1 G1_ROOT_REV4=$S/g1 G1_ROOT_R6=$S/g1_rec G2_ROOT=$S/g2_unused
export ESPLEY_OUT=$S/xtb R5_SCRATCH=$S/r5 R6_SCRATCH=$S/r6 R4_SCRATCH=$S/r4 D1_SNAPSHOT=$S/d1_snapshot_d573111e
# Si entries for the rev 4-6 element tables (user decision 2026-10-19; site_si/sitecustomize.py)
export D1PAIR_ESP_DIR=$(cd "$CODE/../espley_xtb_repro" && pwd) PYTHONPATH=$CODE/site_si PYTHONDONTWRITEBYTECODE=1
# = r4_env.sh (ORCA 6.1.1, OpenMPI 4.1.7a1, xtb 6.7.1)
export ORCA_DIR=/home1/yeseo1ee/orca_6_1_1_avx2
export ORCA_BIN=$ORCA_DIR/orca
export MPI_ROOT=/usr/mpi/gcc/openmpi-4.1.7a1
export PATH=$ORCA_DIR:$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$ORCA_DIR:$MPI_ROOT/lib64:$MPI_ROOT/lib:${LD_LIBRARY_PATH:-}
export OMPI_MCA_rmaps_base_oversubscribe=1 OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_btl=self,vader,tcp OMPI_MCA_pml=ob1 OMPI_MCA_coll_hcoll_enable=0 UCX_TLS=tcp,self,sm
export XTB_BIN=/home1/yeseo1ee/xtb-dist/bin/xtb XTB_THREADS=1 OMP_STACKSIZE=1G
export XTBPATH=/home1/yeseo1ee/xtb-dist/share/xtb XTBHOME=/home1/yeseo1ee/xtb-dist
mkdir -p "$S/G" "$S/g1" "$S/g1_rec" "$S/xtb" "$S/r5" "$S/r6" "$S/r4"
echo "[g_env] G1_ROOT=$G1_ROOT G1_ROOT_R6=$G1_ROOT_R6 PROF=$ESPLEY_PROF xtb=$XTB_BIN orca=$ORCA_BIN si=$PYTHONPATH"
