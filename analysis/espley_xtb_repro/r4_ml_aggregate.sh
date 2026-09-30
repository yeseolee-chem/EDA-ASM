#!/bin/bash
#SBATCH --job-name=r4_ml_agg
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
# Phase 3-4: per geometry aggregate_ml.py + plot_results.py ($ESPLEY_OUT/rev4/<g0|g1>/: ml_report.json,
# ml_table_espley.csv, predictions.parquet, figures/), the PNGs copied to results_rev4/figures/, then rev4_tables.py
# (results_rev4/rev4_headline.{csv,md}, rev4_appendix_{by_arm,by_model,full}.csv, rev4_pre_ml.csv).
# Gates fail loudly (exit != 0: an incomplete target set, mixed geom / rows, G0 vs G1 test rows differ).
#   sbatch --partition=cpu1,cpu2 --dependency=afterany:<ML_JID> --output=$R4_SCRATCH/logs/ml_agg.%j.out r4_ml_aggregate.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
ROOT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
mkdir -p results_rev4/figures
for g in g0 g1; do
    export ESPLEY_ML_OUT=$ROOT/rev4/$g ESPLEY_GEOM=$g
    python aggregate_ml.py || exit $?
    python plot_results.py || exit $?
    cp -f "$ESPLEY_ML_OUT"/figures/*_"$g".png results_rev4/figures/ || exit $?
done
unset ESPLEY_ML_OUT ESPLEY_GEOM
python rev4_tables.py
