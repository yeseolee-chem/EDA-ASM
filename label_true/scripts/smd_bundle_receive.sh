#!/bin/bash
#SBATCH --job-name=smd_recv
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/logs/receive.%j.out
set -euo pipefail

ROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel
INPUTS=$ROOT/inputs

for K in 3 4; do
    TAR=$INPUTS/bundle_$K.tar.gz
    if [ ! -f $TAR ]; then
        echo "=== bundle_$K.tar.gz 없음 skip ==="; continue
    fi
    echo ""
    echo "=== bundle_$K.tar.gz 압축해제 시작 (원본: $(du -h $TAR | awk '{print $1}')) ==="
    t0=$(date +%s)
    # Overwrite existing bundle_$K (which has eda.inp only) with returned bundle
    tar xzf $TAR -C $INPUTS
    echo "  extract: $(( $(date +%s) - t0 ))s"

    # ORCA 재설치 파일들 (17 GB) 삭제 — 여기 이미 있으므로 불필요
    for X in orca_6_1_1_avx2 orca_6_1_1_linux_x86-64_shared_openmpi418_avx2.tar.xz; do
        P=$INPUTS/bundle_$K/$X
        if [ -e $P ]; then
            SZ=$(du -sh $P 2>/dev/null | awk '{print $1}')
            echo "  삭제: bundle_$K/$X ($SZ)"
            rm -rf $P
        fi
    done

    # 원본 tar.gz 삭제
    echo "  삭제: bundle_$K.tar.gz"
    rm -f $TAR
done

echo ""
echo "=== 최종: 각 번들 폴더 상태 ==="
for K in 3 4; do
    B=$INPUTS/bundle_$K
    if [ ! -d $B ]; then continue; fi
    N_RXN=$(ls $B/inputs 2>/dev/null | grep -c '^rxn_' || echo 0)
    N_DONE=$(find $B/inputs -maxdepth 2 -name eda.out 2>/dev/null | xargs -I{} grep -l "ORCA TERMINATED NORMALLY" {} 2>/dev/null | wc -l)
    N_FAIL=$((N_RXN - N_DONE))
    SZ=$(du -sh $B | awk '{print $1}')
    printf "bundle_%d: %d rxns  완료 %d  실패/미완료 %d  크기 %s\n" $K $N_RXN $N_DONE $N_FAIL $SZ
done
