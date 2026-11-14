#!/bin/bash
#SBATCH --job-name=gnn3r2_s0
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# analysis/gnn3r2/jobs/s0_gr.slurm — GNN3R2 1 (docs/specs/GNN3R2.md): git fetch of origin/gnn3r (1-1), the free disk
# (1-5), then analysis/gnn3r2/s0_start.py in reactot. $GR2 must be absent or empty (1-5), so give --output outside $GR2.
# (Copied from analysis/gnn3r/jobs/s0_gr.slurm; fetch target and script path changed: GNN3R2 DECISIONS.md DEC-2.)
# Submit from the repository root:
#   sbatch -p cpu1|cpu2 --time=48:00:00 --output=<dir outside $GR2>/%x_%j.out analysis/gnn3r2/jobs/s0_gr.slurm
# Free disk: mmlsquota when the node has it, else 300 GB user quota (/home1 + /gpfs/tmp_cpu2, memory gpfs_quota_shared)
# minus du of the two user trees (= analysis/g2r/jobs/s0_g2r.slurm).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair
echo "[gnn3r2_s0] host=$(hostname) job=${SLURM_JOB_ID:-?} git=$(git rev-parse --short HEAD) $(date '+%F %T')"
git fetch -q origin gnn3r:refs/remotes/origin/gnn3r && echo "origin/gnn3r $(git rev-parse origin/gnn3r)"
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
python analysis/gnn3r2/s0_start.py "$free" "$method"; rc=$?
echo "=== exit $rc $(date '+%F %T')"
exit $rc
