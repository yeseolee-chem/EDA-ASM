#!/bin/bash
# Prepare a mini test bundle to verify: tar.xz extraction + sbatch + ORCA run
# for one small reaction. Delete on success.
set -euo pipefail

TEST_ROOT=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/test_bundle
BUNDLE_SRC=/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs/bundle_1
TARXZ=/home1/yeseo1ee/projects/eda-asm-prediction/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2.tar.xz

echo "=== 1) 테스트 번들 골격 생성 ==="
rm -rf $TEST_ROOT
mkdir -p $TEST_ROOT/inputs

echo "  test rxn 2개 복사 (bundle_1 에서 가장 작은 것 선정)"
# bundle_1 의 rxn 중 작은 시스템 (원자 수) 을 찾아 2개 복사
for R in rxn_5973 rxn_5972; do
    cp -a $BUNDLE_SRC/inputs/$R $TEST_ROOT/inputs/
done
ls $TEST_ROOT/inputs/

echo ""
echo "  tar.xz 복사 (448 MB)"
cp $TARXZ $TEST_ROOT/
ls -lh $TEST_ROOT/*.tar.xz

echo ""
echo "  launch.sh / worker.sh (bundle_1 것 그대로 복사)"
cp $BUNDLE_SRC/launch.sh $BUNDLE_SRC/worker.sh $BUNDLE_SRC/README.md $TEST_ROOT/

echo ""
echo "=== 2) worker.sh 를 테스트용으로 축소 (array 2-element) ==="
sed -i 's/--array=0-9%10/--array=0-1%2/' $TEST_ROOT/worker.sh
grep "^#SBATCH --array" $TEST_ROOT/worker.sh

echo ""
echo "=== 3) launch.sh 를 tar.xz 자동해제 버전으로 교체 (미리 작성한 것 사용) ==="
