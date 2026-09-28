#!/bin/bash
# Uniformity scan of the SMD relabel scratch: is every file the SMD version?
# Per rxn dir: file set, eda.inp method/FRAG/%pal/%maxcore lines, eda_frag*.inp
# method lines, and for each .out the echoed "!" line, ORCA version, solvent,
# epsilon, SMD module banner and SMD CDS term. Also checks the eda.inp files
# inside the outgoing bundle tarballs. Read-only; writes only to audit/.
#SBATCH --job-name=smd_uniform
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/logs/uniform.%j.out
set -uo pipefail
ROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel
OLD=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/inputs
OUT=$ROOT/audit/uniformity.tsv
SUM=$ROOT/audit/uniformity_summary.txt
TMP=$OUT.tmp

first() { grep -m1 -E "$1" "$2" 2>/dev/null | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//'; }
bang() { grep -m1 -E '^[[:space:]]*!' "$1" 2>/dev/null | sed -E 's/^[[:space:]]*![[:space:]]*//; s/[[:space:]]+/ /g; s/ $//'; }
echo_bang() { grep -m1 -E '^\|[[:space:]]*[0-9]+>[[:space:]]*!' "$1" 2>/dev/null | sed -E 's/^\|[[:space:]]*[0-9]+>[[:space:]]*![[:space:]]*//; s/[[:space:]]+/ /g; s/ $//'; }
outinfo() {  # $1 = .out -> echo|version|solvent|epsilon|smd_banner|cds|terminated
    local f=$1
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s' \
        "$(echo_bang "$f")" \
        "$(first 'Program Version' "$f")" \
        "$(first '^Solvent:' "$f" | awk '{print $NF}')" \
        "$(first '^[[:space:]]*Epsilon[[:space:]]+\.\.\.' "$f" | awk '{print $NF}')" \
        "$(grep -c 'utilizes the SMD solvation module' "$f" 2>/dev/null)" \
        "$(grep -c 'SMD CDS free energy correction energy' "$f" 2>/dev/null)" \
        "$(grep -c 'ORCA TERMINATED NORMALLY' "$f" 2>/dev/null)"
}

{
printf 'rxn\tfiles\tinp_bang\tfrag1_str\tfrag2_str\tpal\tmaxcore\tinp_has_cpcm\tef1_inp_bang\tef2_inp_bang'
for k in eda ef1 ef2 sd1 sd2 rel1 rel2; do printf '\t%s_echo\t%s_version\t%s_solvent\t%s_eps\t%s_smdmod\t%s_cds\t%s_term' $k $k $k $k $k $k $k; done
printf '\n'
for d in $(ls $ROOT/inputs | grep -E '^rxn_[0-9]+$'); do
    R=$ROOT/inputs/$d
    files=$(ls -1 $R | sort | paste -sd,)
    fs1=$(grep -m1 -oE 'FRAG1[[:space:]]+"[^"]+"' $R/eda.inp | sed -E 's/FRAG1[[:space:]]+//; s/"//g; s/[[:space:]]+/ /g')
    fs2=$(grep -m1 -oE 'FRAG2[[:space:]]+"[^"]+"' $R/eda.inp | sed -E 's/FRAG2[[:space:]]+//; s/"//g; s/[[:space:]]+/ /g')
    cp=$(grep -ciE 'CPCM\(water\)|^%cpcm' $R/eda.inp)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "$d" "$files" "$(bang $R/eda.inp)" "$fs1" "$fs2" \
        "$(first '^%pal' $R/eda.inp)" "$(first '^%maxcore' $R/eda.inp)" "$cp" \
        "$(bang $R/eda_frag1.inp)" "$(bang $R/eda_frag2.inp)"
    for f in eda.out eda_frag1.out eda_frag2.out; do printf "\t"; outinfo $R/$f; done
    O=$OLD/$d
    for f in frag1_dist.out frag2_dist.out frag1_rel.out frag2_rel.out; do printf "\t"; outinfo $O/$f; done
    printf '\n'
done
} > $TMP && mv $TMP $OUT

{
echo "SMD relabel scratch uniformity — $(date -Is)"
echo "rxn dirs scanned: $(($(wc -l < $OUT) - 1))"
echo "non-rxn entries in inputs/: [$(ls $ROOT/inputs | grep -vE '^rxn_[0-9]+$' | paste -sd' ')]"
head -1 $OUT | tr '\t' '\n' | nl -ba | tail -n +2 | while read i col; do
    echo ""
    echo "== $col (distinct values: $(tail -n +2 $OUT | cut -f$i | sort -u | wc -l))"
    tail -n +2 $OUT | cut -f$i | sort | uniq -c | sort -rn | head -6
done
echo ""
echo "== outgoing bundle tarballs (bundles_tar/): eda.inp method lines"
for t in $ROOT/bundles_tar/*.tar.gz; do
    [ -f "$t" ] || continue
    echo "-- $(basename $t)"
    tar -xzOf "$t" --wildcards '*/inputs/rxn_*/eda.inp' 2>/dev/null \
        | grep -E '^[[:space:]]*!|FRAG[12][[:space:]]+"|^%cpcm' | sed -E 's/[[:space:]]+/ /g; s/^ //' | sort | uniq -c
done
} > $SUM
cat $SUM
