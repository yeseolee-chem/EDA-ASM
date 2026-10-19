#!/bin/bash
#SBATCH --job-name=d1p_clean
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/d1_pair/logs/%x_%j.out
# cleanup_orca_aux.slurm — delete ORCA by-products that no D1PAIR step reads, in the D1PAIR G1 roots only
# (g1, g1_rec, g1_infra, G0 under $D1PAIR_SCRATCH): *.property.txt, *_trj.xyz, per-step ts.NNN.hess, *.carthess
# (the categories the user approved for the rev 4-6 roots on 2026-10-12). Kept: *.inp *.out *.err ts.xyz ts.hess
# result.json attempts.json markers (read by g1_geom / r6_recover / xtb_slice / ext_features). A manifest (path, bytes)
# is written first to $D1PAIR_SCRATCH/cleanup_<date>/manifest.tsv. Never touches anything outside $D1PAIR_SCRATCH.
set -uo pipefail
S=/gpfs/tmp_cpu2/yeseo1ee/d1_pair
M=$S/cleanup_$(date +%Y%m%d_%H%M%S); mkdir -p "$M"
for d in g1 g1_rec g1_infra G0; do
  [ -d "$S/$d" ] || continue
  find "$S/$d" -type f \( -name '*.property.txt' -o -name '*_trj.xyz' -o -name 'ts.[0-9][0-9][0-9].hess' -o -name '*.carthess' \) \
       -printf '%p\t%s\n' >> "$M/manifest.tsv"
done
awk -F'\t' '{s+=$2; n++} END{printf "deleting %d files, %.2f GB\n", n, s/1e9}' "$M/manifest.tsv"
cut -f1 "$M/manifest.tsv" | while read -r f; do case "$f" in "$S"/*) rm -f -- "$f" ;; esac; done
echo "done $(date '+%F %T')"
