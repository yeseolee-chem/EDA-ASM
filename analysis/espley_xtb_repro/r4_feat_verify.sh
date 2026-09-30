#!/bin/bash
#SBATCH --job-name=r4_feat_verify
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
# Phase 2-1 gate: new G0 (xtb_slice.py --geom dft) vs the OLD rev 3 parquet, every feature within 1e-8.
# Exit 0 = PASS, 3 = FAIL (= STOP). Report: results_rev4/phase2_verify_geom_dft.json
#   sbatch --partition=cpu1,cpu2 --output=$R4_SCRATCH/logs/feat_verify.%j.out r4_feat_verify.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
ROOT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
python verify_geom_dft.py "$ROOT/xtb_features_g0.parquet" "$ROOT/xtb_features.parquet" \
    "$CODE/results_rev4/phase2_verify_geom_dft.json"
