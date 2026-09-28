#!/bin/bash
# run_report.sh — aggregate after the array (afterany); copies the small result files into the repo.
#SBATCH --job-name=d1p_report
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
set -uo pipefail
source "$PILOT_DIR/pilot_env.sh"
cd "$PILOT_DIR"
python report.py --out "$SCRATCH/results"
mkdir -p "$PILOT_DIR/results"
cp "$SCRATCH/results/"{pilot_report.md,pilot_summary.csv,labels_d1_pilot.json} "$PILOT_DIR/results/"
cp "$SCRATCH/s0/S0_REPORT.md" "$PILOT_DIR/results/" 2>/dev/null || true
