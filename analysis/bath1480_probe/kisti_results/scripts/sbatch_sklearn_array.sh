#!/bin/bash
# Phase 5b: per-target sklearn (SVR remaining 5 + RF 13 = 18 tasks)
#SBATCH --job-name=sk_par
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/sk_par.%A_%a.log

set -uo pipefail

# 5 remaining SVR + 13 RF = 18 tasks
COMBOS=(
    "svr:oi_dft"                    # 0
    "svr:disp_dft"                  # 1
    "svr:cpcm_dft"                  # 2
    "svr:cds_dft"                   # 3
    "svr:eint_dft"                  # 4
    "rf:e_barrier_dft"              # 5
    "rf:q_barrier_dft"              # 6
    "rf:sum_distortion_energies_dft" # 7
    "rf:interaction_energies_dft"   # 8
    "rf:distortion_energy_1_dft"    # 9
    "rf:distortion_energy_2_dft"    # 10
    "rf:elst_dft"                   # 11
    "rf:pauli_dft"                  # 12
    "rf:oi_dft"                     # 13
    "rf:disp_dft"                   # 14
    "rf:cpcm_dft"                   # 15
    "rf:cds_dft"                    # 16
    "rf:eint_dft"                   # 17
)

COMBO=${COMBOS[$SLURM_ARRAY_TASK_ID]}
MODEL=${COMBO%:*}
TARGET=${COMBO#*:}

BASE=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results
DATASET=$BASE/manual_tt_7ch_xtb.pkl
SCRIPT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/kisti_results/scripts/sklearn_one_target.py
OUT_ROOT=$BASE/pipeline_7ch/sklearn_parallel
OUT_PKL=$OUT_ROOT/sk_hps_${MODEL}_${TARGET}.pkl

if [ -f "$OUT_PKL" ]; then
    echo "SKIP: $OUT_PKL exists"
    exit 0
fi

mkdir -p $OUT_ROOT
WORK=/tmp/sk_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $WORK && cd $WORK

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate distintml_repro
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export ONLY_MODEL=$MODEL
export ONLY_TARGET=$TARGET

echo "=== sklearn $MODEL:$TARGET  host=$(hostname)  time=$(date -Is) ==="
python3 -u $SCRIPT $DATASET
RC=$?
echo "=== end rc=$RC time=$(date -Is) ==="
if [ $RC -eq 0 ] && [ -f "sk_hps_${MODEL}_${TARGET}.pkl" ]; then
    cp "sk_hps_${MODEL}_${TARGET}.pkl" "$OUT_PKL"
    echo "copied to $OUT_PKL"
fi
cd /tmp && rm -rf $WORK
exit $RC
