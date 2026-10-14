#!/bin/bash
#SBATCH --job-name=r5_lockbox
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
# Phase C-1 (docs/specs/REV5_FEATURES_FIGURES.md): lockbox / dev split of results_rev5/rows_rev5.csv with
# select_blocks.py lockbox -> results_rev5/{lockbox_ids.csv, dev_ids.csv, lockbox.json}: 15 % lockbox drawn with
# np.random.default_rng(20261001) from the ascending rxn_ids (rev5_common constants), dev = the rest. A missing file is
# written atomically; an existing one is never rewritten (exit 3 if the recomputed split differs from it), so a rerun
# only re-verifies. Commit the three files with PREREG_REV5a.md, which must contain the lockbox_ids.csv sha256 printed
# below (lockbox.json lockbox_sha256), before r5_select.sh (which refuses otherwise).
#   sbatch --partition=cpu1,cpu2 --cpus-per-task=1 --mem=4G --output=$R5_SCRATCH/logs/lockbox.%j.out r5_lockbox.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
PY=$R5_SCRATCH/venv/bin/python
[ -x "$PY" ] || { echo "rev 5 venv missing: $PY (r5_probe.sh creates it)"; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f results_rev5/rows_rev5.csv ] || { echo "results_rev5/rows_rev5.csv missing (Phase B merge / gates first)"; exit 2; }
"$PY" select_blocks.py lockbox
rc=$?
if [ "$rc" -eq 0 ]; then
    cat results_rev5/lockbox.json; echo
    ls -la results_rev5/lockbox_ids.csv results_rev5/dev_ids.csv results_rev5/lockbox.json
    sha256sum results_rev5/lockbox_ids.csv results_rev5/dev_ids.csv   # the lockbox one goes into PREREG_REV5a.md
else
    echo "select_blocks.py lockbox exit $rc (3 = an existing lockbox file differs from the recomputed split)"
fi
exit $rc
