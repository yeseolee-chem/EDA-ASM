#!/bin/bash
#SBATCH --job-name=espb_s0
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# analysis/espb/jobs/s0.slurm — ESPB 1 (docs/specs/ESPB.md): git fetch of origin/gnn3r2 (1-1), the free disk (1-9), then
# analysis/espb/s0_start.py in env_espley. $EB must be absent or empty (1-9), so give --output outside $EB.
# Submit from the repository root:
#   sbatch -p cpu1|cpu2 --time=48:00:00 --output=<dir outside $EB>/%x_%j.out analysis/espb/jobs/s0.slurm
# Free disk: mmlsquota when the node has it, else 300 GB user quota (/home1 + /gpfs/tmp_cpu2, memory gpfs_quota_shared)
# minus du of the two user trees (= analysis/gnn3r2/jobs/s0_gr.slurm).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot                                   # git only; python = env_espley
PY=/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair
echo "[espb_s0] host=$(hostname) job=${SLURM_JOB_ID:-?} git=$(git rev-parse --short HEAD) $(date '+%F %T')"
git fetch -q origin gnn3r2:refs/remotes/origin/gnn3r2 && echo "origin/gnn3r2 $(git rev-parse origin/gnn3r2)"
free=""; method=""
if [ -x /usr/lpp/mmfs/bin/mmlsquota ]; then
  q=$(/usr/lpp/mmfs/bin/mmlsquota --block-size G 2>&1); echo "$q"
  free=$(echo "$q" | awk '$2=="USR" && $4+0>0 {f=$4-$3; if (m=="" || f<m) m=f} END {print m}')
  [ -n "$free" ] && method="mmlsquota (quota - used, GB)"
fi
if [ -z "$free" ]; then
  used=$(timeout 7200 du -s -BG "$(realpath /home1/yeseo1ee)" /gpfs/tmp_cpu2/yeseo1ee 2>/dev/null | awk '{s+=$1} END {print s}')
  echo "du used GB: $used"
  free=$((300 - ${used:-300})); method="300 GB quota - du -s -BG of /home1/yeseo1ee + /gpfs/tmp_cpu2/yeseo1ee"
fi
echo "free GB: $free ($method)"
"$PY" analysis/espb/s0_start.py "$free" "$method"; rc=$?
echo "=== exit $rc $(date '+%F %T')"
exit $rc
