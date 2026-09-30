#!/bin/bash
#SBATCH --job-name=r5_downstream
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G
# Phase F downstream (G1, KRR_rbf) of one arm, default EXT_SEL (R5_ARM=<arm> via --export to pick another), on the
# D-2 outputs ($R5_SCRATCH/protoA, r5_protoA.sh) and results_rev5/rows_rev5.csv -> results_rev5/downstream_<arm>/:
#   analyze_extra.py  -> charge_breakdown_g1.csv (D-2 KRR predictions), group_split_g1.csv (D-2 per-seed KRR HP),
#                        analyze_extra_g1.json (inputs)
#   evaluate_pairs.py -> split_metrics_g1.csv (MMP dMAE, dom_agree + per-split null baselines: majority channel and
#                        permutation, exact + 1,000-permutation check), margin_calibration_g1.csv, mmp_pairs_v2_g1.csv,
#                        evaluate_pairs_g1.json (inputs)
# The scripts gate their own inputs (ml_report.json geom / rows / feature / block / prereg_rev5b sha256, the arm's
# recorded columns; for EXT_SEL and other block arms the feature / block / rows_rev5 sha256 recorded in
# prereg_rev5b.json `inputs`, train_ml_single.check_prereg_inputs); existing outputs are skipped. Refuses to run unless
# the pre-registration (PREREG_REV5b) is committed.
#   sbatch --partition=cpu1,cpu2 --dependency=afterok:<PROTOA_JID> --output=$R5_SCRATCH/logs/downstream.%j.out r5_downstream.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
PY=$R5_SCRATCH/venv/bin/python
[ -x "$PY" ] || { echo "rev 5 venv missing: $PY (r5_probe.sh creates it)"; exit 2; }
for f in results_rev5/PREREG_REV5b.md results_rev5/prereg_rev5b.json results_rev5/rows_rev5.csv; do
    if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$f"; then
        echo "PREREG gate: $f is not committed or differs from HEAD — commit PREREG_REV5b (C-3) first"; exit 2
    fi
done
ROOT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
# set here only (as r5_protoA.sh), so a stale ESPLEY_* in the submit shell cannot mix inputs
export ESPLEY_OUT=$ROOT ESPLEY_GEOM=g1 ESPLEY_FEAT=$ROOT/xtb_features_g1.parquet ESPLEY_ML_OUT=$R5_SCRATCH/protoA
export ESPLEY_ROWS=$CODE/results_rev5/rows_rev5.csv ESPLEY_RES5=1
export ESPLEY_DOWNSTREAM_ARM=${R5_ARM:-EXT_SEL}
# serial KRR fits, no GridSearchCV: the BLAS threads are the only parallelism here
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} \
       MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "arm=$ESPLEY_DOWNSTREAM_ARM feat=$ESPLEY_FEAT ml=$ESPLEY_ML_OUT rows=$ESPLEY_ROWS"
rc=0
"$PY" analyze_extra.py || rc=1
"$PY" evaluate_pairs.py || rc=1
ls -la "results_rev5/downstream_$ESPLEY_DOWNSTREAM_ARM"
exit $rc
