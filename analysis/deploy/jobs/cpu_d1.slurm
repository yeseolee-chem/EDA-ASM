#!/bin/bash
#SBATCH --job-name=dep_d1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/deploy/logs/%x_%j.out
# DEPLOY-MODEL phases L, M (and I-3 g2_deploy.py) in the D1 run env (docs/specs/DEPLOY_MODEL.md 17) = G2ENV=d1 of
# analysis/d1_g2/jobs/g2.slurm: d1_run/nurion/env.sh -> venv python 3.9.5 + autodE 1.4.5 + rdkit, xtb 6.7.1, ORCA 6.1.1
# and the self-built OpenMPI 4.1.8f (no XPMEM / UCX; the UBAI system 4.1.7a1 XPMEM build SIGBUSes ORCA *_mpi at
# MPI_Finalize), with OMPI_MCA_pml=ob1, OMPI_MCA_btl=self,vader,tcp, OMPI_MCA_coll_hcoll_enable=0, UCX_TLS=tcp,self,sm,
# oversubscribe, default (core) binding - exactly the env the D1-G2 Freq step ran in.
# submit from analysis/deploy/:
#   sbatch [-p cpu1|cpu2 --cpus-per-task=N --mem=M -J name --array=... --dependency=...] jobs/cpu_d1.slurm <script.py> <args...>
# "@I" -> $SLURM_ARRAY_TASK_ID (left as "@I" outside an array: dep_common.array_task then exits 2).
# D1G2_SCRATCH (read by g2_common at import) = the D1-G2 root (read only for L / M); DEP_D1G2_SCRATCH overrides it
# (I-3 g2_deploy.py sets its own root).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
source /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh
PY="${D1_PY:?}"
export D1G2_SCRATCH="${DEP_D1G2_SCRATCH:-/gpfs/tmp_cpu2/yeseo1ee/d1_g2}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
       DEP_JOB_T0=$(date +%s)
echo "[dep_d1] host=$(hostname) job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}" \
     "git=$(git rev-parse --short HEAD 2>/dev/null) D1G2_SCRATCH=$D1G2_SCRATCH mpirun=$(command -v mpirun)" \
     "$($PY -V 2>&1) $(date '+%F %T')"
args=(); for a in "$@"; do args+=("${a//@I/${SLURM_ARRAY_TASK_ID:-@I}}"); done
echo "=== ${args[*]}"
"$PY" "${args[@]}"; rc=$?
echo "=== exit $rc $(date '+%F %T')"
exit $rc
