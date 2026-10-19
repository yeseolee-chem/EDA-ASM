#!/bin/bash
# jobs/submit_M.sh — one-shot submission of Phase M (training) -> collect -> Phase E (evaluation + figures) as a SLURM
# dependency chain (no login-node poller). Run on the login node from analysis/d1_pair/:  bash jobs/submit_M.sh [SVR_DEP]
#   SVR_DEP  optional job id the SVR array waits for (afterany; e.g. the SVR benchmark that is running the first SVR unit)
# Tasks (MaxSubmit 20 counts array elements): NN 6 + grid 2 + PB 1 + SVR 4 + collect 1 + E 1 = 15.
# Every training task is idempotent (finished units are cached and skipped), so a task clipped by the 48 h wall is
# resubmitted with the same command; collect then exits 2 while units are missing and the chain stops there.
set -euo pipefail
cd "$(dirname "$0")/.."
P=cpu1,cpu2
E="ALL,PYENV=espley"
nn=$(sbatch --parsable --array=0-5 -p $P -J d1m_nn --cpus-per-task=48 --mem=180G \
     --output=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/logs/%x_%A_%a.out \
     --export=$E,D1PAIR_MODELS=NN2+NN4,D1PAIR_WORKERS=24,D1PAIR_NJOBS=1 jobs/py.slurm \
     d1_train.py run T1 @I 6 ::: d1_train.py run LODO @I 6 ::: d1_train.py run LOSO @I 6 ::: d1_train.py run T2 @I 6)
grid=$(sbatch --parsable --array=0-1 -p $P -J d1m_grid --cpus-per-task=48 --mem=160G \
     --output=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/logs/%x_%A_%a.out \
     --export=$E,D1PAIR_MODELS=KRR+XGB+Ridge,D1PAIR_WORKERS=1,D1PAIR_NJOBS=48 jobs/py.slurm \
     d1_train.py run T1 @I 2 ::: d1_train.py run LODO @I 2 ::: d1_train.py run LOSO @I 2 ::: d1_train.py run T2 @I 2)
pb=$(sbatch --parsable -p $P -J d1m_pb --cpus-per-task=16 --mem=200G \
     --export=$E,D1PAIR_MODELS=KRR,D1PAIR_WORKERS=1,D1PAIR_NJOBS=8 jobs/py.slurm d1_train.py run PB 0 1)
dep=${1:+--dependency=afterany:$1}
svr=$(sbatch --parsable --array=0-3 $dep -p $P -J d1m_svr --cpus-per-task=48 --mem=400G \
     --output=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/logs/%x_%A_%a.out \
     --export=$E,D1PAIR_MODELS=SVR,D1PAIR_WORKERS=1,D1PAIR_NJOBS=48 jobs/py.slurm \
     d1_train.py run T1 @I 4 ::: d1_train.py run LODO @I 4 ::: d1_train.py run LOSO @I 4 ::: d1_train.py run T2 @I 4)
col=$(sbatch --parsable --dependency=afterok:$nn:$grid:$pb:$svr --kill-on-invalid-dep=yes -p $P -J d1m_collect \
     --cpus-per-task=4 --mem=64G --export=$E jobs/py.slurm d1_train.py collect)
ev=$(sbatch --parsable --dependency=afterok:$col --kill-on-invalid-dep=yes -p $P -J d1e_all --cpus-per-task=8 \
     --mem=64G --export=$E jobs/py.slurm d1_eval.py all ::: d1_figures.py)
echo "nn=$nn grid=$grid pb=$pb svr=$svr collect=$col eval=$ev"
