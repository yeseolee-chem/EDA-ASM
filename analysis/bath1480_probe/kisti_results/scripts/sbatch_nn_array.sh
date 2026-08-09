#!/bin/bash
# Phase C: NN HP tuning per-(target, network_size), parallel array
# 13 targets × 2 sizes = 26 tasks, throttle at 10 concurrent (queue cap)
#SBATCH --job-name=hyp_nn_7ch
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-25%10
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/pipeline_nn.%A_%a.log

set -uo pipefail

# 13 _dft targets (Espley 6 + our 6 + eint bookkeeping) × 2 sizes
TARGETS=(
    "e_barrier_dft"
    "q_barrier_dft"
    "sum_distortion_energies_dft"
    "interaction_energies_dft"
    "distortion_energy_1_dft"
    "distortion_energy_2_dft"
    "elst_dft"
    "pauli_dft"
    "oi_dft"
    "disp_dft"
    "cpcm_dft"
    "cds_dft"
    "eint_dft"
)

# Task index -> (target, size)
IDX=$SLURM_ARRAY_TASK_ID
TIDX=$((IDX / 2))
SIDX=$((IDX % 2))
NSIZE=$([ $SIDX -eq 0 ] && echo 2 || echo 4)
TARGET=${TARGETS[$TIDX]}

BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results
DATASET=$BASE/manual_tt_7ch_xtb.pkl  # updated: with xTB features
SCRIPT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/kisti_results/scripts/nn_one_target.py
OUT_ROOT=$BASE/pipeline_7ch/nn_parallel
OUT_PKL=$OUT_ROOT/nn_hps_${TARGET}_${NSIZE}L.pkl

# Idempotency
if [ -f "$OUT_PKL" ]; then
    echo "SKIP: $OUT_PKL exists"
    exit 0
fi

mkdir -p $OUT_ROOT

# Node-local workdir (keras_tuner trial data avoids GPFS quota)
NODE_WORK=/tmp/nn_7ch_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$NODE_WORK"
cd "$NODE_WORK"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=1
export ONLY_TARGET=$TARGET
export ONLY_NN_SIZE=$NSIZE

echo "=== NN start target=$TARGET size=${NSIZE}L  host=$(hostname)  time=$(date -Is) ==="
python3 -u "$SCRIPT" "$DATASET"
RC=$?
echo "=== end rc=$RC time=$(date -Is) ==="

if [ $RC -eq 0 ] && [ -f "nn_hps_${TARGET}_${NSIZE}L.pkl" ]; then
    cp "nn_hps_${TARGET}_${NSIZE}L.pkl" "$OUT_PKL"
    echo "copied to $OUT_PKL"
fi
cd /tmp
rm -rf "$NODE_WORK"
exit $RC
