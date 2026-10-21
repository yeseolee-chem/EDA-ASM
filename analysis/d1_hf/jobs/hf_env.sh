# D1-HF job environment (docs/specs/D1HF.md 10): conda reactot for python, ORCA 6.1.1 (the label build) run serially
# (nprocs 1, so no MPI settings are needed; jobs/g_env.sh is the D1PAIR G1 environment and is not used here).
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export ORCA_DIR=/home1/yeseo1ee/orca_6_1_1_avx2
export ORCA_BIN=$ORCA_DIR/orca
export PATH=$ORCA_DIR:$PATH
export LD_LIBRARY_PATH=$ORCA_DIR:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_STACKSIZE=4G
ulimit -s unlimited 2>/dev/null || true
export D1HF_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_hf
export PYTHONUNBUFFERED=1
