#!/bin/bash
# SPEC_08 whole-dataset LC — sbatch wrapper for chain_lc.sh.
#
# This job (a) runs make_lc_splits.py on the compute node so the launcher
# has splits/lc_splits.json before starting, then (b) execs chain_lc.sh
# which polls squeue and submits (size, fold, member) cells as running
# slots free up. Both parts run on cpu2 — no login-node processes.
#
# 48h walltime per CLAUDE.md. If the wall hits before all cells are
# submitted, resubmit this same script — cell trainer is idempotent,
# already-submitted / already-done cells are skipped.

#SBATCH --job-name=s08w_ch
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec08w_chain_%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/spec08w_chain_%j.err

set -uo pipefail
mkdir -p /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

echo "[node] $(hostname)  [start] $(date '+%F %T')"
echo "[jobid] ${SLURM_JOB_ID:-unknown}"

SPLITS="spec/spec08_whole_dataset_learning_curve/splits/lc_splits.json"
if [ ! -f "${SPLITS}" ]; then
    echo "[splits] generating ${SPLITS}"
    python -u spec/spec08_whole_dataset_learning_curve/code/make_lc_splits.py
else
    echo "[splits] reusing existing ${SPLITS}"
fi

echo "[chain] launching chain_lc.sh"
exec bash spec/spec08_whole_dataset_learning_curve/code/chain_lc.sh
