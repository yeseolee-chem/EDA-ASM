#!/bin/bash
#SBATCH --job-name=ams_lic_check
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=ADF_800/logs/lic_check_%j.out
#SBATCH --error=ADF_800/logs/lic_check_%j.err

set -e  # no pipefail — `cmd | head` would otherwise kill the script on SIGPIPE

echo "=== AMS license probe on $(hostname) at $(date -Iseconds) ==="
echo "Slurm job: ${SLURM_JOB_ID:-NOJOB}"
echo ""
echo "=== local NIC MACs ==="
for f in /sys/class/net/*/address; do
    iface=$(basename $(dirname $f))
    printf "  %-15s %s\n" "$iface" "$(cat $f)"
done
echo ""
echo "=== source AMS env ==="
source /home1/yeseo1ee/ams2026.103/amsbashrc.sh
set -u
echo "AMSHOME=$AMSHOME"
echo "SCMLICENSE=$SCMLICENSE"
echo ""

cd /tmp
mkdir -p ams_lic_${SLURM_JOB_ID:-test}
cd ams_lic_${SLURM_JOB_ID:-test}

export NSCM=1
export SCM_TMPDIR=$PWD
ulimit -s unlimited 2>/dev/null || true

echo "=== ams --version (full) ==="
"$AMSBIN/ams" --version 2>&1 > ams_version.out || true
head -5 ams_version.out
echo ""

echo "=== attempt a tiny H-atom SinglePoint ==="
"$AMSBIN/ams" > ams_test.out 2>&1 << 'EOF'
System
  Atoms
    H 0.0 0.0 0.0
  End
End
Task SinglePoint
Engine ADF
  Basis
    Type SZ
  End
  XC
    LDA SCF VWN
  End
  Unrestricted Yes
  SpinPolarization 1
EndEngine
EOF
rc=$?
echo "ams returned $rc"
echo ""
echo "=== first 50 lines of ams_test.out ==="
head -50 ams_test.out 2>/dev/null
echo ""
echo "=== relevant license/error grep ==="
grep -iE "license|error|fail|invalid|kid0|sigsegv" ams_test.out 2>/dev/null | head -20 || echo "(none)"
echo ""
echo "=== Done ==="
