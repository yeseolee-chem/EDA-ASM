#!/bin/bash
#SBATCH --job-name=r4_cmp_prep
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
# Phase 4-1/4-2 prep (compare_espley.py prep), once per geometry:
#   $R4_SCRATCH/compare/espley/  id mapping, label agreement, split replication (assert), d1/d2 swap vector,
#                                Espley's stored test predictions, their hyp_tuning.py (fetched, for the grid check)
#   $R4_SCRATCH/compare/{g0,g1}/prep.pkl   usable-row mask + pre-ML xTB values of that feature parquet
# Existing outputs are skipped. Needs xtb_features_g0.parquet and xtb_features_g1.parquet.
#   sbatch --partition=cpu1,cpu2 --output=$R4_SCRATCH/logs/compare_prep.%j.out r4_compare_prep.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
ROOT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
export ESPLEY_REPO_DATA=${ESPLEY_REPO_DATA:-/gpfs/tmp_cpu2/yeseo1ee/espley_compare}
rc=0
for g in g0 g1; do
    # derived from g only, so a stale ESPLEY_FEAT in the submit shell cannot mix geometries
    ESPLEY_GEOM=$g ESPLEY_FEAT=$ROOT/xtb_features_$g.parquet python compare_espley.py prep || rc=1
done
ls -la "$R4_SCRATCH/compare/espley" "$R4_SCRATCH/compare/g0" "$R4_SCRATCH/compare/g1"
exit $rc
