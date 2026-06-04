#!/bin/bash
#SBATCH --job-name=ams_autolic
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=ADF_800/logs/autolic_%j.out
#SBATCH --error=ADF_800/logs/autolic_%j.err

# Register THIS compute node's hostid with SCM via the autolicense web API.
# Requires the following environment variables to be set by the submitter:
#   SCM_USER     — SCM username (e.g. u30731)
#   SCM_PASS     — SCM password
#   SCM_EMAIL    — contact email registered with SCM
#
# Usage:
#   SCM_USER=u30731 SCM_PASS='...' SCM_EMAIL=yeseolele@gmail.com \
#       sbatch --export=ALL scripts/slurm/ams_autolicense_register.sbatch
#
# The license is fetched & installed into $AMSHOME/license.txt (or merged in
# if appropriate). After running this once per new compute node hostid, normal
# ams jobs should work on that node.

set -e

if [[ -z "${SCM_USER:-}" || -z "${SCM_PASS:-}" || -z "${SCM_EMAIL:-}" ]]; then
    echo "ERROR: must set SCM_USER, SCM_PASS, SCM_EMAIL before submitting" >&2
    echo "  e.g. SCM_USER=u30731 SCM_PASS=... SCM_EMAIL=...@... sbatch ..." >&2
    exit 2
fi

echo "=== autolicense register on $(hostname) at $(date -Iseconds) ==="
echo "Slurm job: ${SLURM_JOB_ID:-NOJOB}"

source /home1/yeseo1ee/ams2026.103/amsbashrc.sh

echo ""
echo "=== invoking autolicense nogui (online registration via license.scm.com) ==="
# This contacts https://license.scm.com/autolicense, sends the hostid of the
# current node, and writes back an updated license entry.
"$AMSBIN/autolicense" nogui -u "$SCM_USER" -p "$SCM_PASS" -m "$SCM_EMAIL"
rc=$?
echo ""
echo "autolicense returned $rc"
echo ""

echo "=== verify with a tiny SinglePoint ==="
cd /tmp
mkdir -p ams_verify_${SLURM_JOB_ID:-test}
cd ams_verify_${SLURM_JOB_ID:-test}
export NSCM=1
export SCM_TMPDIR=$PWD
ulimit -s unlimited 2>/dev/null || true

"$AMSBIN/ams" > verify.out 2>&1 << 'EOF'
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

echo ""
echo "=== verify.out — search for license result ==="
grep -iE 'license|error|valid|invalid' verify.out 2>/dev/null | head -10 || true
echo ""
echo "=== verify.out — final 30 lines (success indicators or errors) ==="
tail -30 verify.out 2>/dev/null

echo ""
echo "=== Done at $(date -Iseconds) ==="
