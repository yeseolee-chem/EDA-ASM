#!/bin/bash
# jobs/submit_train.sh — ESPB (docs/specs/ESPB.md 12): one-shot submission of the A / B training arrays -> C collect ->
# E compare -> F figures + R report as a SLURM dependency chain (no login-node poller). Run once on the login node from
# analysis/espb/:  bash jobs/submit_train.sh        (check `squeue -u $USER -h | wc -l` + 19 <= 20 first)
# Tasks (MaxSubmit 20 counts array elements): EA grid 2 + EA NN 4 + EB grid 2 + EB SVR 4 + EB NN 4 + collect 1 +
# compare 1 + figures/report 1 = 19. Resources = analysis/d1_pair/jobs/submit_M.sh. Every training task is idempotent
# (finished units are cached and skipped), so a task clipped by the 48 h wall is resubmitted with the same command;
# collect then exits 2 while units are missing and the chain stops there.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${ESPB_PARTITION:-cpu1,cpu2}
L=/gpfs/tmp_cpu2/yeseo1ee/espb/logs
E="ALL,PYENV=espley"
T="--time=48:00:00"
eag=$(sbatch --parsable --array=0-1 -p $P $T -J espb_ea_grid --cpus-per-task=48 --mem=160G --output=$L/%x_%A_%a.out \
      --export=$E,ESPB_MODELS=Ridge+KRR+SVR,ESPB_WORKERS=1,ESPB_NJOBS=48 jobs/py.slurm ea_train.py run @I 2)
ean=$(sbatch --parsable --array=0-3 -p $P $T -J espb_ea_nn --cpus-per-task=48 --mem=180G --output=$L/%x_%A_%a.out \
      --export=$E,ESPB_MODELS=NN2+NN4,ESPB_WORKERS=24,ESPB_NJOBS=1 jobs/py.slurm ea_train.py run @I 4)
ebg=$(sbatch --parsable --array=0-1 -p $P $T -J espb_eb_grid --cpus-per-task=48 --mem=160G --output=$L/%x_%A_%a.out \
      --export=$E,ESPB_MODELS=Ridge+KRR,ESPB_WORKERS=1,ESPB_NJOBS=48 jobs/py.slurm eb_train.py run @I 2)
ebs=$(sbatch --parsable --array=0-3 -p $P $T -J espb_eb_svr --cpus-per-task=48 --mem=400G --output=$L/%x_%A_%a.out \
      --export=$E,ESPB_MODELS=SVR,ESPB_WORKERS=1,ESPB_NJOBS=48 jobs/py.slurm eb_train.py run @I 4)
ebn=$(sbatch --parsable --array=0-3 -p $P $T -J espb_eb_nn --cpus-per-task=48 --mem=180G --output=$L/%x_%A_%a.out \
      --export=$E,ESPB_MODELS=NN2+NN4,ESPB_WORKERS=24,ESPB_NJOBS=1 jobs/py.slurm eb_train.py run @I 4)
col=$(sbatch --parsable --dependency=afterok:$eag:$ean:$ebg:$ebs:$ebn --kill-on-invalid-dep=yes -p $P $T -J espb_collect \
      --cpus-per-task=4 --mem=64G --output=$L/%x_%j.out --export=$E jobs/py.slurm collect.py)
cmp=$(cd ../.. && sbatch --parsable --dependency=afterok:$col --kill-on-invalid-dep=yes -p $P $T -J espb_compare \
      --cpus-per-task=4 --mem=64G --output=$L/%x_%j.out gnn3/jobs/run.slurm analysis/espb/compare.py)
rep=$(sbatch --parsable --dependency=afterok:$cmp --kill-on-invalid-dep=yes -p $P $T -J espb_report \
      --cpus-per-task=2 --mem=16G --output=$L/%x_%j.out --export=ALL,PYENV=reactot jobs/py.slurm figures.py ::: report.py)
echo "ea_grid=$eag ea_nn=$ean eb_grid=$ebg eb_svr=$ebs eb_nn=$ebn collect=$col compare=$cmp report=$rep"
