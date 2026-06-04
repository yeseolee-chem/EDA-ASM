#!/bin/bash
#SBATCH -J ams_mpi_smoke
#SBATCH -p cpu1
#SBATCH -n 4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/logs/cpu1_mpi_smoke.out
#SBATCH -e /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/logs/cpu1_mpi_smoke.err

# Test whether AMS runs with MPI parallelism (NSCM=$SLURM_CPUS_PER_TASK)
# on a cpu1 compute node. If MPI_Init works here (unlike on gate1),
# production array can use NSCM=$SLURM_CPUS_PER_TASK for speedup.

set -e

source $HOME/.bashrc   # loads mpi/2021.9.0 + amsbashrc.sh

echo "===== host info ====="
hostname
echo "AMSHOME=$AMSHOME"
which ams
which mpirun
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo ""

export OMP_NUM_THREADS=1
export NSCM=$SLURM_NTASKS
echo "Using NSCM=$NSCM (MPI tasks)"

cd /tmp
mkdir -p mpi_smoke_$SLURM_JOB_ID
cd mpi_smoke_$SLURM_JOB_ID

echo "===== Running H2 SP with NSCM=$NSCM ====="
$AMSBIN/ams << 'END_INPUT'
Task SinglePoint
System
    Atoms
        H 0.0 0.0 0.0
        H 0.0 0.0 0.74
    End
End
Engine ADF
    Basis
        Type DZ
    End
    NumericalQuality Basic
    XC
        GGA PBE
    End
EndEngine
END_INPUT
EXIT=$?
echo ""
echo "===== Exit code: $EXIT ====="
echo "===== ams.log tail ====="
tail -20 ams.log 2>/dev/null || true
echo ""
echo "===== ams.out (energy line) ====="
grep -E "Energy.*hartree|Energy \(hartree\)" ams.out 2>/dev/null | head -3 || true
echo ""
echo "===== Cleanup ====="
cd /tmp
rm -rf mpi_smoke_$SLURM_JOB_ID
echo "Done at $(date)"
