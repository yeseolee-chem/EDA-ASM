#!/bin/bash
#SBATCH --job-name=r4_rows
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
# Phase 3-1: pre-registered ML rows = G1 ok ∩ no NaN (ESPLEY73 + 9 targets) ∩ hygiene
# -> results_rev4/rows_rev4.csv + rows_rev4.json. Needs xtb_features_g1.parquet (r4_feat_aggregate.sh GEOM=g1). Run
# BEFORE the PREREG commit. An existing rows_rev4.csv is never overwritten (exit 3 if the recomputed rows differ);
# exit 1 = gate failed. MODE=check runs `make_rows_rev4.py --check-g1-only` instead (REV5 A-4: the G1-only rows must
# equal rows_rev4.csv; report results_rev5/A_rows_check.json; r5_phaseA.sh step 2 runs the same).
#   sbatch --partition=cpu1,cpu2 [--export=ALL,MODE=check] --output=$R4_SCRATCH/logs/make_rows.%j.out r4_make_rows.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
if [ "${MODE:-}" = check ]; then python make_rows_rev4.py --check-g1-only; else python make_rows_rev4.py; fi
