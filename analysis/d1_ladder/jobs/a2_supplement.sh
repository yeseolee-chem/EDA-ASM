#!/bin/bash
#SBATCH --job-name=lad_A2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/logs/%x_%j.out
# D1LADDER Phase A stage 2: prereg.py supplement (results/PREREG_D1PAIR_ADD3_G2.md + the ADD3 json extension, ADD3 10),
# then commit both files at once so the window in which the json differs from HEAD (any trainer starting then exits 3
# at gate_add3) is seconds, then prereg.py check --stage 2 / 1. Push is done by hand from the login node.
# Submit from analysis/d1_ladder: sbatch -p cpu1,cpu2 [--dependency=afterok:<stage-2 C merge>] jobs/a2_supplement.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh; conda activate reactot
export D1LADDER_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder PYTHONUNBUFFERED=1 D1LADDER_JOB_T0=$(date +%s)
R=$(git rev-parse --show-toplevel)
MD=analysis/d1_ladder/results/PREREG_D1PAIR_ADD3_G2.md; JS=analysis/d1_ladder/results/prereg_d1pair_add3.json
echo "[a2] host=$(hostname) git=$(git rev-parse --short HEAD) $(date '+%F %T')"
python prereg.py supplement; rc=$?
[ $rc -ne 0 ] && { echo "=== supplement exit $rc"; exit $rc; }
if ! git -C "$R" diff --quiet HEAD -- "$JS" || [ -n "$(git -C "$R" status --porcelain -- "$MD")" ]; then
  git -C "$R" add "$MD" "$JS" && git -C "$R" -c user.name=yeseo1ee -c user.email=yeseolele@gmail.com commit -q \
    -m "D1LADDER Phase A stage 2: PREREG ADD3 supplement (g2; prereg.py supplement, md sha256 $(sha256sum "$R/$MD" | cut -c1-16))" \
    -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" -- "$MD" "$JS" || { echo "STOP: commit failed"; exit 3; }
  echo "[a2] committed $(git -C "$R" rev-parse --short HEAD)"
fi
python prereg.py check --stage 2 || exit $?
python prereg.py check --stage 1 || exit $?
echo "=== exit 0 $(date '+%F %T')"
