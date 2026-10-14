#!/bin/bash
#SBATCH --job-name=r4_ablation_agg
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
# Phase 4-3: merge the 42 JSONs of r4_ablation.sh (one row rule, ABL_ROWS=common default | own) ->
# results_rev4/espley_geometry_ablation[_ownrows].csv (mean MAE of the 5 seeds, SE, sd over seeds, best params, n per
# arm / target / model), _per_seed.csv and _rows.json (the shared rows, dropped rxn ids, rows each arm lost to the
# intersection). Exit 1 = a JSON is missing or was made from other inputs / rows: resubmit r4_ablation.sh first.
#   sbatch --partition=cpu1,cpu2 --dependency=afterany:<ABL_JID> [--export=ALL,ABL_ROWS=own] --output=$R4_SCRATCH/logs/ablation_agg.%j.out r4_ablation_aggregate.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
export PYTHONWARNINGS=ignore
python espley_fairness_ablation.py aggregate
