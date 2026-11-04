#!/bin/bash
# analysis/deploy/jobs/i3_chain.sh — one-shot submission of the DEPLOY-MODEL I-3 chain smoke (docs/specs/DEPLOY_MODEL.md
# 14-3; analysis/deploy/i3_chain.py). Login node: six sbatch calls with dependencies, then it returns (no loop, no
# polling, no resident process). Uses 6 of the 20 submit slots.
#   bash analysis/deploy/jobs/i3_chain.sh [partition(s), default cpu1,cpu2] [root, default $DEP/i3]
# Before: sinfo -p cpu1,cpu2 -o "%.9P %.6t %C" and pass the partition with idle CPUs (cpu2 QOS: <= 64 CPUs per job).
# prep -> g2 (autodE, 10 x 4 cores) -> lm -> ft -> predict: afterok + kill-on-invalid-dep; report: afterany (always
# writes R/results/I3_chain.json + analysis/deploy/results/I3_chain.json, also after a failed / cancelled step).
# Every step is idempotent: after a failure or the 48 h wall resubmit that step's line by hand (see the log).
set -euo pipefail
DEPDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)            # analysis/deploy
P=${1:-cpu1,cpu2}
R=${2:-/gpfs/tmp_cpu2/yeseo1ee/deploy/i3}
L=$R/logs
n=$(squeue -u "$USER" -h -r | wc -l)
[ "$n" -le 14 ] || { echo "STOP: $n tasks queued; the chain needs 6 of the 20 submit slots"; exit 2; }
[ -d "$L" ] || { echo "STOP: $L missing (sbatch --time=48:00:00 --wrap 'mkdir -p $L' first)"; exit 2; }
O=(--time=48:00:00 --parsable -p "$P")
K=(--kill-on-invalid-dep=yes)
cd "$DEPDIR"
J0=$(sbatch "${O[@]}" -J i3_prep --cpus-per-task=1 --mem=4G --output="$L/%x_%j.out" \
       jobs/cpu_d1.slurm i3_chain.py prep --root "$R"); J0=${J0%%;*}
J1=$(sbatch "${O[@]}" -J i3_g2 --cpus-per-task=40 --mem=64G --dependency=afterok:"$J0" "${K[@]}" --output="$L/%x_%j.out" \
       jobs/cpu_d1.slurm i3_chain.py g2 --root "$R" --n-par 10 --cores 4); J1=${J1%%;*}
J2=$(sbatch "${O[@]}" -J i3_lm --cpus-per-task=4 --mem=16G --dependency=afterok:"$J1" "${K[@]}" --output="$L/%x_%j.out" \
       jobs/cpu_d1.slurm i3_chain.py lm --root "$R"); J2=${J2%%;*}
J3=$(cd ../d1_pair && sbatch "${O[@]}" -J i3_ft --cpus-per-task=10 --mem=32G --dependency=afterok:"$J2" "${K[@]}" \
       --output="$L/%x_%j.out" --export=ALL,PYENV=venv5,D1PAIR_PHASE=F \
       jobs/g.slurm ../deploy/i3_chain.py ft --root "$R"); J3=${J3%%;*}
J4=$(sbatch "${O[@]}" -J i3_pred --cpus-per-task=4 --mem=32G --dependency=afterok:"$J3" "${K[@]}" --output="$L/%x_%j.out" \
       --export=ALL,DEPENV=d1fu2 jobs/cpu_py.slurm i3_chain.py predict --root "$R"); J4=${J4%%;*}
J5=$(sbatch "${O[@]}" -J i3_rep --cpus-per-task=2 --mem=16G --dependency=afterany:"$J4" --output="$L/%x_%j.out" \
       --export=ALL,DEPENV=d1fu2 jobs/cpu_py.slurm i3_chain.py report --root "$R"); J5=${J5%%;*}
echo "I-3 chain ($P, root $R): prep $J0 -> g2 $J1 -> lm $J2 -> ft $J3 -> predict $J4 -> report $J5; logs $L/i3_*"
