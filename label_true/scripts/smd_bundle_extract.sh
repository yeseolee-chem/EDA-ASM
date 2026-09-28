#!/bin/bash
# Extract a returned SMD bundle archive into inputs/ on a compute node.
# Usage: sbatch smd_bundle_extract.sh /gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs/<bundle>.tar.gz
# Skips the bundled ORCA install and wavefunction / scratch files (only
# .inp/.out/.err are kept at merge time anyway). Does not delete the archive:
# that happens after the extracted rxns are verified and merged.
#SBATCH --job-name=smd_extract
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/logs/extract.%j.out
set -uo pipefail
TAR=${1:?archive path}
cd "$(dirname "$TAR")" || exit 1
echo "=== extract $TAR ($(stat -c %s "$TAR") bytes) start=$(date -Is) node=$(hostname -s)"
t0=$(date +%s)
tar -xzf "$TAR" \
    --exclude='*/orca_6_1_1_avx2' --exclude='*/orca_6_1_1_avx2/*' \
    --exclude='*/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2*' \
    --exclude='*.gbw' --exclude='*.tmp' --exclude='*.tmp.*' --exclude='*.densities' \
    --exclude='*.densitiesinfo' --exclude='*.bas[0-9]*'
rc=$?
echo "tar rc=$rc  elapsed=$(( $(date +%s) - t0 ))s"
B=$(dirname "$TAR")/$(basename "$TAR" .tar.gz)
[ -d "$B" ] || B=$(find "$(dirname "$TAR")" -maxdepth 6 -type d -name "$(basename "$TAR" .tar.gz)" | head -1)
echo "bundle dir: $B"
ls "$B"
echo "rxn dirs: $(ls "$B/inputs" 2>/dev/null | grep -cE '^rxn_[0-9]+$')  bundle state/done: $(ls "$B/state/done" 2>/dev/null | wc -l)  claims: $(ls "$B/state/claim" 2>/dev/null | wc -l)"
du -sh "$B" 2>/dev/null
echo "=== end $(date -Is)"
exit $rc
