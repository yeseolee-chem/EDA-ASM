#!/bin/bash
#SBATCH --job-name=mlt_env
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/mlts/logs/%x_%j.out
# analysis/mlts/jobs/env_build.slurm — DEPLOY-V2 start condition 0-7: build the conda env `mlts`
# (conda create --clone reactot, then pip installs; reactot itself is never modified).
# Stage 1 (stamp .mlts_env_ok, job 1061590): clone + mace-torch 0.3.16 + sella + orca-mlips; sella's jax pulled numpy 2.2.6,
#   which broke matscipy / tblite (built for numpy 1.x), and aimnet needs torch >= 2.5 (orca-mlips extras >= 2.8).
# Stage 2 (stamp .mlts_env_v2, decision 0-7 in results/DECISIONS.md): in the mlts env only, torch 2.8.0 (cu128 wheels) and
#   numpy pinned to reactot's 1.26.4 (aimnet needs numpy < 2), then aimnet, sella, orca-mlips[mace,aimnet2] with those pins.
# Then analysis/mlts/env_check.py loads every candidate on the GPU and writes results/0_env.json.
# UMA-S is not installed: no HuggingFace token on this account (spec 4: "없으면 뺀다").
# Idempotent: each stage is skipped when its stamp exists; the check always reruns.
# Submit from analysis/mlts:  sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 jobs/env_build.slurm
set -uo pipefail
ENVN=mlts
CONDA=/home1/yeseo1ee/miniconda3
ENV=$CONDA/envs/$ENVN
PY=$ENV/bin/python
LOG=/gpfs/tmp_cpu2/yeseo1ee/mlts/logs/env_build_${SLURM_JOB_ID}
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=4
source $CONDA/etc/profile.d/conda.sh
echo "[mlt_env] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) start=$(date -Is)"
inst() {  # inst <name> <pip args...> : record success / failure, never abort the build
  local n=$1; shift
  echo "=== pip $n: $*"
  if $PY -m pip install "$@" > ${LOG}_$n.txt 2>&1; then echo "pip_$n=ok"; else echo "pip_$n=FAIL (see ${LOG}_$n.txt)"; tail -8 ${LOG}_$n.txt; fi
}
if [[ ! -f $ENV/.mlts_env_ok ]]; then
  [[ -d $ENV ]] || conda create -y -n $ENVN --clone reactot || { echo "clone failed"; exit 3; }
  CONS=$ENV/.mlts_constraints.txt
  echo "torch==2.2.1" > $CONS
  inst mace -c $CONS -U "mace-torch"
  inst sella -c $CONS "sella"
  inst orcamlips -c $CONS "orca-mlips"
  touch $ENV/.mlts_env_ok
fi
if [[ ! -f $ENV/.mlts_env_v2 ]]; then
  CONS2=$ENV/.mlts_constraints_v2.txt
  printf 'torch==2.8.0\nnumpy==1.26.4\ne3nn==0.4.4\n' > $CONS2
  inst torch28 torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  inst numpy -c $CONS2 "numpy==1.26.4"
  inst mace2 -c $CONS2 "mace-torch==0.3.16"
  inst sella2 -c $CONS2 "sella"
  inst aimnet -c $CONS2 "aimnet"
  inst orcamlips2 -c $CONS2 "orca-mlips[mace,aimnet2]"
  $PY -m pip check > ${LOG}_pipcheck.txt 2>&1; echo "pip check rc=$?"; tail -30 ${LOG}_pipcheck.txt
  $PY -c "import torch, numpy; assert torch.__version__.startswith('2.8.0'), torch.__version__; assert numpy.__version__ == '1.26.4', numpy.__version__; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), 'numpy', numpy.__version__)" || exit 3
  touch $ENV/.mlts_env_v2
fi
# Stage 3 (stamp .mlts_env_v3; done by hand on 11-04 inside the GPU sandbox with `jobs/sbx.sh mlts -m pip ...`, recorded here
#   so a rebuild repeats it): the cloned torchvision 0.17.1 / torchaudio 2.2.1 broke `import mace` under torch 2.8.
if [[ ! -f $ENV/.mlts_env_v3 ]]; then
  CONS2=$ENV/.mlts_constraints_v2.txt
  inst tv torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  inst cueq -c $CONS2 cuequivariance-torch cuequivariance-ops-torch-cu12 "nvidia-cuda-nvcc-cu12==12.8.*" "nvidia-cuda-runtime-cu12==12.8.*"
  touch $ENV/.mlts_env_v3
fi
cd ${SLURM_SUBMIT_DIR:-.}
$PY env_check.py
rc=$?
echo "[mlt_env] env_check rc=$rc done $(date -Is)"
exit $rc
