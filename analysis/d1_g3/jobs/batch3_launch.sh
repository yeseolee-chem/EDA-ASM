#!/bin/bash
#SBATCH --job-name=g3_b3launch
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/d1_g3/logs/%x_%j.out
# D1-G3 batch 3 launcher (CLAUDE.md option 2: one-shot sbatch calls from a compute node, no login-node process).
# usage (from analysis/d1_g3): sbatch -p cpu1,cpu2 --dependency=afterany:<pilot array> jobs/batch3_launch.sh <JA>
#   JA = the opt.py assemble job; batch 3 = Phase F (G, H, F, P/report), E (E3) and R (g3_eval run ::: summary),
#   the same lines as the session plan. Submits all 8 tasks or nothing (MaxSubmit 20 counts array elements).
set -euo pipefail
JA=${1:?assemble job id}
D=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1g3/analysis/d1_g3
L=/gpfs/tmp_cpu2/yeseo1ee/d1_g3/logs
n=$(squeue -u "$USER" -h -r | wc -l)            # includes this launcher, which leaves right after
if [ $((n - 1 + 8)) -gt 20 ]; then
  echo "NOT SUBMITTED: $n tasks queued (incl. this job), batch 3 needs 8 (MaxSubmit 20); resubmit this launcher"
  exit 2
fi
cd "$D"
JG=$(sbatch --parsable -p cpu1,cpu2 -J g3_G --time=48:00:00 --cpus-per-task=16 --mem=32G --output=$L/%x_%j.out \
     --export=ALL,G3ENV=d1 --dependency=afterok:$JA --kill-on-invalid-dep=yes jobs/g3.slurm feat.py g all)
JH=$(cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J g3_H --time=48:00:00 --cpus-per-task=48 --mem=180G --array=0-1 \
     --output=$L/%x_%A_%a.out --dependency=afterok:$JA --kill-on-invalid-dep=yes jobs/hf.slurm ../d1_g3/feat.py h A)
JF=$(cd ../d1_pair && sbatch --parsable -p cpu1,cpu2 -J g3_F --time=48:00:00 --cpus-per-task=32 --mem=64G --array=0-1 \
     --output=$L/%x_%A_%a.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F --dependency=afterok:$JG --kill-on-invalid-dep=yes \
     jobs/g.slurm ../d1_g3/feat.py f @I)
JPF=$(sbatch --parsable -p cpu1,cpu2 -J g3_P --time=48:00:00 --cpus-per-task=2 --mem=16G --output=$L/%x_%j.out \
      --export=ALL,G3ENV=reactot --dependency=afterok:$JF:$JH --kill-on-invalid-dep=yes \
      jobs/g3.slurm feat.py p g3h ';;;' feat.py p g3r ';;;' feat.py report)
JE3=$(sbatch --parsable -p cpu1,cpu2 -J g3_E3 --time=48:00:00 --cpus-per-task=4 --mem=24G --output=$L/%x_%j.out \
      --dependency=afterok:$JPF --kill-on-invalid-dep=yes ../d1_ladder/jobs/gpu.slurm apply.py e3)
JEV=$(sbatch --parsable -p cpu1,cpu2 -J g3_E --time=48:00:00 --cpus-per-task=8 --mem=32G --output=$L/%x_%j.out \
      --export=ALL,G3ENV=reactot --dependency=afterok:$JE3 --kill-on-invalid-dep=yes \
      jobs/g3.slurm g3_eval.py run ::: g3_eval.py summary)
echo "batch 3 submitted: JG=$JG JH=$JH JF=$JF JPF=$JPF JE3=$JE3 JEV=$JEV (after JA=$JA)"
