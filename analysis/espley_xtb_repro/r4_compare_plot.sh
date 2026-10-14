#!/bin/bash
#SBATCH --job-name=r4_cmp_plot
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
# Phase 4-1/4-2 tables + figures (compare_espley.py plot): every side re-scored on the labelled ∩ G1-usable rows
# (3,327 of 3,510, asserted) -> results_rev4/espley_compare_{main,role,appendix_best,index_d1d2_appendix,per_seed}.csv,
#    _main.png, _role.png, _checks.json (rewritten atomically). Gates (else exit != 0): the G1 prep.pkl matches its
#    feature parquet as it is now and every ours_preds file carries that parquet's sha256; their protocol re-run
#    reproduced hps.pkl; identical test rows and y on every side.
#   sbatch --dependency=afterok:<TRAIN_JID> --partition=cpu1,cpu2 --output=$R4_SCRATCH/logs/compare_plot.%j.out r4_compare_plot.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export ESPLEY_REPO_DATA=${ESPLEY_REPO_DATA:-/gpfs/tmp_cpu2/yeseo1ee/espley_compare}
unset ESPLEY_GEOM ESPLEY_FEAT                  # plot reads $R4_SCRATCH/compare/{espley,g1}
python compare_espley.py plot
