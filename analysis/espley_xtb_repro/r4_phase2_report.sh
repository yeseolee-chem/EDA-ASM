#!/bin/bash
#SBATCH --job-name=r4_phase2_rep
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
# Phase 2-4 report: G0 vs G1 feature shift, ok counts, MAE(b_disp - dft_disp_dft)
# -> results_rev4/phase2_feature_shift.csv, phase2_report.json
#   sbatch --partition=cpu1,cpu2 --output=$R4_SCRATCH/logs/phase2_report.%j.out r4_phase2_report.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
python phase2_report.py
