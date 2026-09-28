#!/bin/bash
#SBATCH --job-name=smd_pack
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/logs/compress.%j.out
set -euo pipefail

SRC=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs
OUT=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/bundles_tar
mkdir -p $OUT
cd $SRC

echo "=== 압축 시작 (5개 번들, 각 ~450 MB) ==="
for k in 1 2 3 4 5; do
    echo ""
    echo "[$(date -Is)] bundle_$k → bundle_$k.tar.gz"
    t0=$(date +%s)
    tar czf $OUT/bundle_$k.tar.gz bundle_$k/
    echo "  완료: $(( $(date +%s) - t0 ))s, size $(du -h $OUT/bundle_$k.tar.gz | awk '{print $1}')"
done

echo ""
echo "=== 최종 결과 ==="
ls -lh $OUT/
echo ""
echo "총 크기: $(du -sh $OUT | awk '{print $1}')"
echo ""
echo "홈 접근용 심볼릭 링크:"
ln -sfn $OUT /home1/yeseo1ee/projects/eda-asm-prediction/bundles_tar
ls -la /home1/yeseo1ee/projects/eda-asm-prediction/bundles_tar
