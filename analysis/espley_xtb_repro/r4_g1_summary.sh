#!/bin/bash
#SBATCH --job-name=r4_g1_sum
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
# Phase 1-2 report: success rate, failure types, RMSD / Δd_form distributions, STOP flags
# -> results_rev4/g1_geometry_summary.{csv,json}
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
mkdir -p results_rev4
python g1_geom.py summary results_rev4/g1_geometry_summary.csv
