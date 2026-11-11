#!/bin/bash
#SBATCH --job-name=abs3_s0
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/abs3/logs/%x_%j.out
# analysis/abs3/jobs/s0_abs.slurm — ABS3 1 (docs/specs/ABS3_MODELS.md): git fetch origin ch3m (1-1), env_espley versions
# (1-6), free disk (1-7), then s0_abs.py in env_d1fu2 and commit S0. Submit from analysis/abs3/:
#   sbatch -p cpu1,cpu2 jobs/s0_abs.slurm
# Free disk (compute nodes cannot run mmlsquota): conservative = 300 GB - (187 GB used at the ch3 S0 du of 11-10 11:55
# + du of $CH3, $CH3M and $ABS3 now, all largely created after that du) (= CH3M S0 method + $ABS3).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
CH3=/gpfs/tmp_cpu2/yeseo1ee/ch3; CH3M=/gpfs/tmp_cpu2/yeseo1ee/ch3m; ABS3=/gpfs/tmp_cpu2/yeseo1ee/abs3
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export DEP_SCRATCH=$CH3/lf DEP_RESULTS=$(git rev-parse --show-toplevel)/analysis/ch3/results/lf_inputs PYTHONDONTWRITEBYTECODE=1
export D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair D1LADDER_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder
echo "[abs3_s0] host=$(hostname) job=${SLURM_JOB_ID:-?} git=$(git rev-parse --short HEAD) $(date '+%F %T')"
git fetch -q origin ch3m && echo "origin/ch3m $(git rev-parse origin/ch3m)"
VJ=$ABS3/s0_espley_versions.json
/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley/bin/python -c "import json,sys,xgboost,sklearn,numpy,pandas; json.dump(dict(xgboost=xgboost.__version__, sklearn=sklearn.__version__, numpy=numpy.__version__, pandas=pandas.__version__, python=sys.version.split()[0]), open('$VJ','w'))"
cat $VJ; echo
u=$(du -s -BG $CH3 $CH3M $ABS3 2>/dev/null | awk '{s+=$1} END {print s}')
free=$((300 - 187 - ${u:-300})); method="300 GB - (187 GB used at ch3 S0 11:55 + du -s -BG of \$CH3, \$CH3M, \$ABS3 = ${u} GB)"
echo "free GB (conservative): $free ($method)"
/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2/bin/python s0_abs.py "$VJ" "$free" "$method"; rc=$?
echo "=== exit $rc $(date '+%F %T')"
[ $rc -eq 0 ] || exit $rc
/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2/bin/python commit_abs.py S0; rc=$?
echo "=== commit exit $rc $(date '+%F %T')"
exit $rc
