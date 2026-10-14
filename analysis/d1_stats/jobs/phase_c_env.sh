# analysis/d1_stats/jobs/phase_c_env.sh — Phase C (SPEC_D1_ALL): rev 4 / rev 5 환경 그대로 + D1 경로만 바꾼다.
#   analysis/espley_xtb_repro/r4_env.sh 를 그대로 source 한다 (reactot, ORCA 6.1.1, xtb 6.7.1, 같은 MPI 설정, D1_SNAPSHOT).
#   그 전에 rev 5 스크립트가 읽는 환경 변수를 D1 뿌리(rev5_common.D1_*)로 돌려 둔다 — D0 파일은 읽지도 쓰지도 않는다.
#   ext_features.py 는 rev 5 venv (tblite 0.7.0) 의 python 으로 돌린다 ($PY).
#   한 가지만 더한다: OMPI_MCA_btl_vader_single_copy_mechanism=none (복사 방식만 바꾸는 MPI 전송 설정, 수치 영향 없음).
#   UBAI 시스템 OpenMPI 4.1.7a1 의 XPMEM 단일 복사가 MPI_Finalize 에서 SIGBUS 를 내는 문제(2026-10-13 측정, 6/6 무사)를 피한다.
B_FEAT=${D1_FEAT_ROOT:-/gpfs/tmp_cpu2/yeseo1ee/d1_features}
export D1_FEAT_ROOT=$B_FEAT
export ESPLEY_LABELS=$B_FEAT/labels_g1in.json
export ESPLEY_PROF=$B_FEAT/prof
export ESPLEY_META=$B_FEAT/input_meta.csv
export ESPLEY_SMILES=$B_FEAT/smiles.csv
export G1_ROOT=$B_FEAT/g1
export ESPLEY_OUT=$B_FEAT/xtb
export R5_SCRATCH=$B_FEAT/r5
export ESPLEY_CODE=${SLURM_SUBMIT_DIR:?저장소 루트에서 제출}/analysis/espley_xtb_repro
export OMPI_MCA_btl_vader_single_copy_mechanism=none
source "$ESPLEY_CODE/r4_env.sh" || exit 2
PY=/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/venv/bin/python
ST=$REPO/analysis/d1_stats
mkdir -p "$B_FEAT/logs" "$G1_ROOT" "$ESPLEY_OUT" "$R5_SCRATCH"
for v in ESPLEY_LABELS ESPLEY_PROF ESPLEY_META ESPLEY_SMILES G1_ROOT ESPLEY_OUT R5_SCRATCH D1_SNAPSHOT; do echo "[phase_c_env] $v=${!v}"; done
