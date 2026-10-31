#!/bin/bash
# analysis/geo2/m3_launch.sh — one-shot submitter of the GEO2b LADDER-training steps (design steps 9-12; RUNBOOK_lad.md).
# Runs on the login node and only calls `squeue` (queue room) and `sbatch` (dependency chains, CLAUDE.md option 3): no
# python, no loop, no sleep, nothing stays resident. Every job: --time=48:00:00, logs under /gpfs/tmp_cpu2/yeseo1ee/
# geo2b/logs, partitions cpu1,cpu2 / gpu1,gpu2,gpu3,gpu5,gpu6 (SLURM takes whichever has room). All python steps are
# atomic and idempotent (resubmitting a step skips finished units / rewrites the same files).
#
#   bash analysis/geo2/m3_launch.sh splits [--dry-run]    1 task : d0d1_splits.py build ::: check (after the FINAL GEO1b
#                                                          D0_pairs.csv is in $GEO1B/d0 and its report in this worktree)
#   bash analysis/geo2/m3_launch.sh ctests [--dry-run]    3 tasks: run_tests.py geo gpu (GPU, incl. the atom smoke),
#                                                          geo cpu (32 CPUs, incl. XGB smoke, diag, R-3), geo merge
#   bash analysis/geo2/m3_launch.sh prereg [--dry-run]    1 task : prereg_geo2b.py write (after C_tests_geo.json
#                                                          all_pass); then review + commit + push by hand (login node)
#   bash analysis/geo2/m3_launch.sh m3 [--workers W] [--dry-run]
#                                                          W + 5 tasks (default W = 9): prereg check -> XGB run (1 x 32
#                                                          CPUs) -> XGB check; atom array 0..W-1 (1 GPU each); collect
#                                                          --with-xgb after both; prereg check again (LADDER files)
# Exit 0 = submitted (or printed with --dry-run), 2 = not enough queue room (MaxSubmit 20 counts array tasks), a job
# name of the step already in the queue (GEO1B.md 6: never the same job twice) / usage.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)       # analysis/geo2 = the submit dir of jobs/m3_*.slurm
cd "$HERE"
GL=/gpfs/tmp_cpu2/yeseo1ee/geo2b/logs
CPU=cpu1,cpu2
GPU=gpu1,gpu2,gpu3,gpu5,gpu6
MAXSUB=20
mode=${1:-}; a2=${2:-}; a3=${3:-}; a4=${4:-}          # at most: <mode> [--workers W] [--dry-run] (no arg loop)
W=9; DRY=0
if [ "$a2" = "--dry-run" ] || [ "$a3" = "--dry-run" ] || [ "$a4" = "--dry-run" ]; then DRY=1; fi
if [ "$a2" = "--workers" ]; then W=$a3; elif [ "$a3" = "--workers" ]; then W=$a4; fi
case "$W" in ''|*[!0-9]*) echo "--workers needs a positive number (got '$W')"; exit 2 ;; esac
[ "$W" -ge 1 ] && [ "$W" -le 10 ] || { echo "--workers must be 1..10 (MaxJobs 10)"; exit 2; }
case "$mode" in      # NAMES = the job names this step submits (GEO1B.md 6: a name already in the queue -> not again)
  splits) NEED=1; NAMES='geo2b_splits' ;; ctests) NEED=3; NAMES='geo2b_ctest_[gcm]' ;;
  prereg) NEED=1; NAMES='geo2b_prereg' ;; m3) NEED=$((W + 5)); NAMES='geo2b_(Rc|x3|x3c|m3|m3col|Rc2)' ;;
  *) sed -n '2,19p' "$0"; exit 2 ;;
esac
N=$(squeue -u "$USER" -h -r | wc -l)
echo "[m3_launch] mode $mode, queue now $N task(s), this step adds $NEED (MaxSubmit $MAXSUB)"
if [ $((N + NEED)) -gt $MAXSUB ]; then
  echo "NOT SUBMITTED: $N + $NEED > $MAXSUB queued tasks; submit again when the queue has room"
  [ $DRY -eq 1 ] || exit 2
fi
DUP=$(squeue -u "$USER" -h -o '%j' | { grep -Ex "$NAMES" || true; } | sort -u | tr '\n' ' ')
if [ -n "$DUP" ]; then
  echo "NOT SUBMITTED: job(s) of this step already running / pending: $DUP(GEO1B.md 6: never submit the same job" \
       "twice; resubmit a dead task alone with the RUNBOOK_lad lines)"
  [ $DRY -eq 1 ] || exit 2
fi
sub() {   # sub <name> <sbatch args...> : prints the line; submits unless --dry-run; echoes the job id
  local name=$1; shift
  if [ $DRY -eq 1 ]; then echo "DRY (cd $HERE) sbatch --time=48:00:00 $*" >&2; echo "DRY_$name"; return 0; fi
  sbatch --parsable --time=48:00:00 "$@"
}
case "$mode" in
  splits)
    J=$(sub splits -p $CPU -J geo2b_splits --cpus-per-task=2 --mem=8G --output=$GL/%x_%j.out \
        --export=ALL,M3ENV=reactot jobs/m3_cpu.slurm ../geo2/d0d1_splits.py build ::: ../geo2/d0d1_splits.py check)
    echo "splits: $J  (done = log '=== exit 0' twice; analysis/geo2/results/S_splits_d0d1.json preliminary false)" ;;
  ctests)
    JG=$(sub ctg -p $GPU --gres=gpu:1 -J geo2b_ctest_g --cpus-per-task=4 --mem=48G --output=$GL/%x_%j.out \
         jobs/m3_gpu.slurm tests/run_tests.py geo gpu)
    JC=$(sub ctc -p $CPU -J geo2b_ctest_c --cpus-per-task=32 --mem=64G --output=$GL/%x_%j.out \
         --export=ALL,M3ENV=espley jobs/m3_cpu.slurm tests/run_tests.py geo cpu)
    JM=$(sub ctm -p $CPU -J geo2b_ctest_m --cpus-per-task=1 --mem=4G --output=$GL/%x_%j.out \
         --dependency=afterany:$JG:$JC --export=ALL,M3ENV=reactot jobs/m3_cpu.slurm tests/run_tests.py geo merge)
    echo "ctests: gpu $JG, cpu $JC, merge $JM  (done = analysis/geo2/results/C_tests_geo.json all_pass true)" ;;
  prereg)
    J=$(sub prereg -p $CPU -J geo2b_prereg --cpus-per-task=1 --mem=8G --output=$GL/%x_%j.out \
        --export=ALL,M3ENV=reactot jobs/m3_cpu.slurm ../geo2/prereg_geo2b.py write)
    echo "prereg: $J  (then read results/PREREG_GEO2B.md; commit + push it, prereg_geo2b.json and the final code)" ;;
  m3)
    JRC=$(sub rc -p $CPU -J geo2b_Rc --cpus-per-task=1 --mem=4G --output=$GL/%x_%j.out \
          --export=ALL,M3ENV=reactot jobs/m3_cpu.slurm ../geo2/prereg_geo2b.py check)
    JX=$(sub x3 -p $CPU -J geo2b_x3 --cpus-per-task=32 --mem=64G --output=$GL/%x_%j.out \
         --dependency=afterok:$JRC --kill-on-invalid-dep=yes --export=ALL,M3ENV=espley \
         jobs/m3_cpu.slurm train_xgb.py run 0 1 --stage 3)
    JXC=$(sub x3c -p $CPU -J geo2b_x3c --cpus-per-task=2 --mem=16G --output=$GL/%x_%j.out \
          --dependency=afterok:$JX --kill-on-invalid-dep=yes --export=ALL,M3ENV=espley \
          jobs/m3_cpu.slurm train_xgb.py check --stage 3)
    JM=$(sub m3 -p $GPU --gres=gpu:1 -J geo2b_m3 --cpus-per-task=4 --mem=32G --array=0-$((W - 1)) \
         --output=$GL/%x_%A_%a.out --dependency=afterok:$JRC --kill-on-invalid-dep=yes \
         jobs/m3_gpu.slurm train_atoms.py run @I $W --stage 3)
    JCO=$(sub col -p $CPU -J geo2b_m3col --cpus-per-task=2 --mem=48G --output=$GL/%x_%j.out \
          --dependency=afterok:$JM:$JXC --kill-on-invalid-dep=yes --export=ALL,M3ENV=d1fu2 \
          jobs/m3_cpu.slurm train_atoms.py collect --stage 3 --with-xgb)
    JPC=$(sub rc2 -p $CPU -J geo2b_Rc2 --cpus-per-task=1 --mem=4G --output=$GL/%x_%j.out \
          --dependency=afterok:$JCO --kill-on-invalid-dep=yes --export=ALL,M3ENV=reactot \
          jobs/m3_cpu.slurm ../geo2/prereg_geo2b.py check)
    echo "m3: prereg check $JRC, XGB $JX, XGB check $JXC, atoms $JM (array 0-$((W - 1))), collect $JCO, final check $JPC"
    echo "  resubmit a dead atom task k (finished units are skipped; same GPU type as its finished units, log line" \
         "'[geo2b_gpu] ... gpu=<name>'):"
    echo "    cd $HERE && sbatch --time=48:00:00 -p <partition of that GPU type> --gres=gpu:1 -J geo2b_m3 --cpus-per-task=4" \
         "--mem=32G --array=<k> --output=$GL/%x_%A_%a.out jobs/m3_gpu.slurm train_atoms.py run @I $W --stage 3"
    echo "  then the collect line again with --dependency=afterok:<new id>:$JXC and the geo2b_Rc2 prereg check line with" \
         "--dependency=afterok:<new collect id> (both were cancelled by --kill-on-invalid-dep)" ;;
esac
