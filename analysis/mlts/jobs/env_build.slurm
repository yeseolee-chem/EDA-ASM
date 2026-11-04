#!/bin/bash
#SBATCH --job-name=mlt_env
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/mlts/logs/%x_%j.out
# analysis/mlts/jobs/env_build.slurm — DEPLOY-V2 start condition 0-7: build the conda env `mlts`
# (conda create --clone reactot, then pip installs; reactot itself is never modified).
# torch stays 2.2.1 (constraint file) so the cloned MACE / e3nn / torch_geometric stack keeps working.
# Candidates (spec 4): MACE-OMOL-0 + MACE-OFF24 (mace-torch upgrade), AIMNet2 (aimnet), sella, orca-mlips.
# UMA-S is not installed: no HuggingFace token on this account (spec 4: "없으면 뺀다").
# Then analysis/mlts/env_check.py loads every candidate on the GPU and writes results/0_env.json.
# Idempotent: the build is skipped when the stamp exists; the check always reruns.
# Submit from analysis/mlts:  sbatch -p gpu3,gpu4,gpu5 jobs/env_build.slurm
set -uo pipefail
ENVN=mlts
CONDA=/home1/yeseo1ee/miniconda3
ENV=$CONDA/envs/$ENVN
STAMP=$ENV/.mlts_env_ok
LOG=/gpfs/tmp_cpu2/yeseo1ee/mlts/logs/env_build_${SLURM_JOB_ID}
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=4
source $CONDA/etc/profile.d/conda.sh
echo "[mlt_env] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -Is)"
if [[ ! -f $STAMP ]]; then
  if [[ ! -d $ENV ]]; then
    conda create -y -n $ENVN --clone reactot || { echo "clone failed"; exit 3; }
  fi
  PY=$ENV/bin/python
  CONS=$ENV/.mlts_constraints.txt
  echo "torch==2.2.1" > $CONS
  $PY -m pip freeze | grep -iE '^(torch-geometric|torch_geometric|e3nn)==' >> $CONS || true
  status=0
  inst() {  # inst <name> <pip args...> : record success / failure, never abort the build
    local n=$1; shift
    echo "=== pip $n: $*"
    if $PY -m pip install -c $CONS "$@" > ${LOG}_$n.txt 2>&1; then echo "pip_$n=ok"; else echo "pip_$n=FAIL (see ${LOG}_$n.txt)"; tail -5 ${LOG}_$n.txt; fi
  }
  inst mace -U "mace-torch"
  inst sella "sella"
  inst aimnet "aimnet"
  inst orcamlips "orca-mlips"
  $PY -m pip check > ${LOG}_pipcheck.txt 2>&1; echo "pip check rc=$?"; tail -20 ${LOG}_pipcheck.txt
  $PY -c "import torch; assert torch.__version__.startswith('2.2.1'), torch.__version__; print('torch', torch.__version__, torch.cuda.is_available())" || exit 3
  touch $STAMP
fi
cd ${SLURM_SUBMIT_DIR:-.}
$ENV/bin/python env_check.py
rc=$?
echo "[mlt_env] env_check rc=$rc done $(date -Is)"
exit $rc
