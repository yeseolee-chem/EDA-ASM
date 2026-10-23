#!/bin/bash
#SBATCH --job-name=lad_P3
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/logs/%x_%j.out
# D1LADDER Phase P for stage 2 (spec 12). inputs.py changed after the stage-1 builds (P20 --g2-cons-own,
# --accept-g2f-stop), so the dft / g1 build fingerprints are stale and 'pairs' needs current builds. The dft / g1 tables
# are gated by ADD3 (sha256) and fingerprinted into the running stage-1 units: they are rebuilt only if byte-identical.
#   1. back up $LAD/inputs/{dft,g1}.parquet   2. build dft / g1 --force   3. cmp with the backups: a difference restores
#   the backup and STOPs (exit 3) before anything else is written   4. build g2 (--accept-g2f-stop --g2-cons-own)
#   5. pairs (P_report.json with P_g2 / P_012).  Submit from analysis/d1_ladder: sbatch -p cpu1,cpu2 jobs/p_stage2.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh; conda activate reactot
export D1LADDER_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder PYTHONUNBUFFERED=1 D1LADDER_JOB_T0=$(date +%s)
I=$D1LADDER_SCRATCH/inputs; B=$I/backup_stage1_$(date +%Y%m%d_%H%M%S)
echo "[p_stage2] host=$(hostname) git=$(git rev-parse --short HEAD) $(date '+%F %T') backup $B"
mkdir -p "$B" && cp -p "$I/dft.parquet" "$I/g1.parquet" "$B/" || exit 3
python inputs.py build dft --force --accept-f0-stop || { echo "STOP: build dft failed"; exit 3; }
python inputs.py build g1 --force || { echo "STOP: build g1 failed"; exit 3; }
for r in dft g1; do
  if ! cmp -s "$B/$r.parquet" "$I/$r.parquet"; then
    cp -p "$B/$r.parquet" "$I/$r.parquet.restore.tmp" && mv "$I/$r.parquet.restore.tmp" "$I/$r.parquet"
    echo "STOP: rebuilt $r.parquet differs from the stage-1 table: backup restored ($B); P_build_$r.json now stale"
    exit 3
  fi
done
echo "[p_stage2] dft / g1 rebuilt byte-identical"
python inputs.py build g2 --accept-g2f-stop --g2-cons-own || { echo "STOP: build g2 failed"; exit 3; }
python inputs.py pairs || { echo "STOP: pairs failed"; exit 3; }
echo "=== exit 0 $(date '+%F %T')"
