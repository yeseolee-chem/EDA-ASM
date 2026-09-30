#!/bin/bash
#SBATCH --job-name=r4_ablation_agg
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
# Phase 4-3: merge the 68 JSONs of r4_ablation.sh -> results_rev4/espley_geometry_ablation.csv (mean MAE of the 5 seeds,
# SE, sd over seeds, best params, n per arm / target / model), espley_geometry_ablation_per_seed.csv and
# espley_geometry_ablation_rows.json (the shared rows, dropped rxn ids, rows each arm lost to the intersection).
# Exit 1 = a JSON is missing or was made from other inputs / rows: resubmit r4_ablation.sh first.
#   sbatch --partition=cpu1,cpu2 --dependency=afterany:<ABL_JID> --output=$R4_SCRATCH/logs/ablation_agg.%j.out r4_ablation_aggregate.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export PYTHONWARNINGS=ignore
python espley_fairness_ablation.py aggregate
