#!/bin/bash
#SBATCH --job-name=r5_figures
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
# Phase E (docs/specs/REV5_FEATURES_FIGURES.md): the paper figures fig1..fig8 (figures_rev5.py) from the Phase C / D
# outputs in results_rev5/ (C2_block_cv*, D1_lockbox*, D3a_*, D3b_*, D4_learning_curves*)
#   -> results_rev5/figures/<name>.{pdf,png,csv} + figures_rev5_manifest.json
# A figure whose input is missing or inconsistent is skipped with a message and the job exits 1 at the end (the other
# figures are still drawn); a figure whose inputs, code and font are unchanged is not redrawn, so a rerun (or a
# resubmit after the 48 h wall) is cheap. Open the PNGs afterwards to check label overlap / clipping (spec Phase E);
# the manifest lists the automatic text-overlap check per figure.
#   sbatch --partition=cpu1,cpu2 --cpus-per-task=2 --mem=8G --output=$R5_SCRATCH/logs/figures.%j.out r5_figures.sh
#   (arguments pass through to figures_rev5.py: --only fig1,fig7 / --force)
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export R5_SCRATCH=${R5_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/espley_rev5}
PY=$R5_SCRATCH/venv/bin/python
[ -x "$PY" ] || { echo "rev 5 venv missing: $PY (r5_probe.sh creates it)"; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
# matplotlib config / font cache of this job on scratch: the compute node's fonts decide Arial vs DejaVu Sans
export MPLCONFIGDIR=$R5_SCRATCH/mplconfig
mkdir -p "$R5_SCRATCH/logs" "$MPLCONFIGDIR" results_rev5/figures || exit 2
"$PY" -c "import matplotlib, pandas, numpy; print('py env ok: matplotlib', matplotlib.__version__, 'pandas', \
pandas.__version__, 'numpy', numpy.__version__)" || exit 2
"$PY" figures_rev5.py "$@"
rc=$?
ls -la results_rev5/figures/
[ "$rc" -eq 0 ] || echo "figures_rev5.py exit $rc (1 = a figure was skipped or failed: see above and the manifest)"
exit $rc
