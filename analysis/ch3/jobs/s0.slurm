#!/bin/bash
#SBATCH --job-name=ch3_s0
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/ch3/logs/%x_%j.out
# analysis/ch3/jobs/s0.slurm — CH3-MODEL 1 + 2 (docs/specs/CH3_MODEL.md): scratch folders, the spec file copy (the
# attached text, extracted to $CH3/spec_extract by job ch3_spec), the 1-8 free disk, then s0_start.py.
# Submit from analysis/ch3/:  sbatch -p cpu1,cpu2 jobs/s0.slurm
# Free disk: mmlsquota when the node has it, else 300 GB user quota (/home1 + /gpfs/tmp_cpu2, memory gpfs_quota_shared)
# minus du of the two user trees.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
CH3=/gpfs/tmp_cpu2/yeseo1ee/ch3
WT=$(git rev-parse --show-toplevel)
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
echo "[ch3_s0] host=$(hostname) job=${SLURM_JOB_ID:-?} git=$(git rev-parse --short HEAD) $(date '+%F %T')"
mkdir -p $CH3/{logs,lf,units,model,smoke,c3r} $CH3/lf/{g2lf_final,F,fu2_lf,units,model,smoke,prof_empty}
ls -A $CH3/lf/prof_empty | wc -l | xargs echo "prof_empty entries:"
SPEC=$CH3/spec_extract/line035107_user_7.txt
if [ ! -f "$WT/docs/specs/CH3_MODEL.md" ]; then
  head -1 "$SPEC" | grep -q '^# CH3-MODEL' && cp "$SPEC" "$WT/docs/specs/CH3_MODEL.md" && echo "spec copied"
fi
sha256sum "$WT/docs/specs/CH3_MODEL.md" "$SPEC"
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
python s0_start.py "$free" "$method"; rc=$?
echo "=== exit $rc $(date '+%F %T')"
exit $rc
