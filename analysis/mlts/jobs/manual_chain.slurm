#!/bin/bash
#SBATCH --job-name=mc
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/mlts/logs/%x_%A_%a.out
# analysis/mlts/jobs/manual_chain.slurm — hand-submitted steps 2-5 outside the frozen launcher (user 11-05 20:5x).
# A thin wrapper of jobs/mlt.slurm (same segment syntax ' ::: ' / ' ;;; ', same envs via MLTENV) that
#   * runs from analysis/mlts whatever directory sbatch was called from (one-shot sbatch from the login node), and
#   * with MC_ROOT=full points every mlt_common path at the deploy root: MLT_SCRATCH=$MLT0/full,
#     MLT_RESULTS=analysis/mlts/results/full (the launcher's $MLT0 / results/ are not touched).
# usage: sbatch [-J mc_<name>] -p cpu1,cpu2|gpu1,...,gpu6 [--gres=gpu:1] -c N --mem=MG [--array=0-K] \
#          --export=ALL,MLTENV=<reactot|mlts|d1>[,MC_ROOT=full] \
#          /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-mlts/analysis/mlts/jobs/manual_chain.slurm <script.py> <args> [::: ...]
# Commands of every step: python manual_chain.py cmds <step> (also in $MLT0/full/cmds/).
set -uo pipefail
HERE=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-mlts/analysis/mlts
MLT0=/gpfs/tmp_cpu2/yeseo1ee/mlts
case "${MC_ROOT:-}" in
  "") ;;
  full) export MLT_SCRATCH=$MLT0/full MLT_RESULTS=$HERE/results/full
        mkdir -p "$MLT_SCRATCH" "$MLT_RESULTS" ;;
  *) echo "[mc] unknown MC_ROOT=${MC_ROOT}"; exit 2 ;;
esac
echo "[mc] MC_ROOT=${MC_ROOT:-none} MLT_SCRATCH=${MLT_SCRATCH:-default} MLT_RESULTS=${MLT_RESULTS:-default}"
export SLURM_SUBMIT_DIR=$HERE
cd "$HERE" || exit 2
exec bash "$HERE/jobs/mlt.slurm" "$@"
