#!/bin/bash
# gnn3/jobs/prepare_inputs.sh — one-shot submitter (sbatch calls only) of the input chain for a list of D1 reactions:
#   job 1  D1 run env (analysis/deploy/jobs/cpu_d1.slurm):      prepare_inputs.py finalize metadata
#   job 2  venv5 env  (analysis/d1_pair/jobs/g.slurm), afterok: prepare_inputs.py features tokens
#
#   bash gnn3/jobs/prepare_inputs.sh ROOT LIST.csv [REUSE_FINALIZED_ROOT]        (run from the repository root)
#
# Every job: --time=48:00:00, 8 CPUs, partitions cpu1,cpu2, logs in ROOT/logs. Prints the job ids.
set -euo pipefail
ROOT=$(realpath -m "${1:?ROOT}"); LIST=$(realpath "${2:?LIST.csv}"); REUSE=${3:-}
REPO=$(git rev-parse --show-toplevel)
PREP=$REPO/gnn3/scripts/prepare_inputs.py
mkdir -p "$ROOT/logs"
extra=(); [ -n "$REUSE" ] && extra=(--reuse-finalized "$(realpath "$REUSE")")
j1=$(cd "$REPO/analysis/deploy" && sbatch --parsable -p cpu1,cpu2 --cpus-per-task=8 --mem=16G --time=48:00:00 \
     -J gnn3_inL -o "$ROOT/logs/%x_%j.out" jobs/cpu_d1.slurm \
     "$PREP" finalize metadata --root "$ROOT" --jobs "$LIST" "${extra[@]}")
j2=$(cd "$REPO/analysis/d1_pair" && sbatch --parsable -p cpu1,cpu2 --cpus-per-task=8 --mem=16G --time=48:00:00 \
     -J gnn3_inF -o "$ROOT/logs/%x_%j.out" --dependency=afterok:$j1 --kill-on-invalid-dep=yes \
     --export=ALL,PYENV=venv5,D1PAIR_PHASE=F jobs/g.slurm "$PREP" features tokens --root "$ROOT" --jobs "$LIST")
echo "$j1 $j2"
