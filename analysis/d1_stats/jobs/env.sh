# analysis/d1_stats/jobs/env.sh — SPEC_D1_ALL 잡들이 공통으로 읽는 환경 (sbatch 안에서만 source 한다; 로그인 노드에서 python 을 돌리지 않는다)
#   잡은 저장소 루트에서 제출한다: 각 .slurm 이 "$SLURM_SUBMIT_DIR/analysis/d1_stats/jobs/env.sh" 를 source 한다.
#   python: reactot (pandas 2.3.3 · numpy 1.26.4 · scipy 1.15.3 · scikit-learn 1.7.2 · statsmodels 0.14.6 · rdkit · matplotlib 3.10.8)
#   D1_build/d1_run = 실행 트리 /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run 으로 가는 링크 (git 에는 없음). 링크가 없는 checkout 이면 실행 트리를 직접 쓴다.
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ST=$REPO/analysis/d1_stats
B=/gpfs/tmp_cpu2/yeseo1ee/d1
D0_LABELS=$REPO/labels_all.json
# Phase R: 중간 파일 (sha256 a2df136b…, 9,459행 — 2026-10-14 07:21 snapshot, D1 이다)
INTERIM=$REPO/../eda-asm-prediction/D1_build/results_d1/snapshot_20261014_0721/labels_all.json
[ -s "$INTERIM" ] || INTERIM=$B/d1_results/snapshot_20261014_0721/labels_d1.json
INTERIM_SHA=a2df136b3d5964b288ed1500ef1393e5ead4d3f8df280191e8d2ad59d11240a7
# Phase A 이후: 확정 파일 (D1 본계산이 끝난 뒤 aggregate_run.py --final 이 쓰는 곳)
D1R=$REPO/D1_build/d1_run; [ -d "$D1R" ] || D1R=$B/d1_run
FINAL_LABELS=$D1R/results/labels_d1.json
FINAL_STATUS=$D1R/results/d1_status.csv
FINAL_MANIFEST=$D1R/d1_manifest.csv
FINAL_GEOM=$D1R/results/geometries
STATUS_COL=class                       # d1_status.csv 의 분류 열 (aggregate_run.write_status_csv)
SIGMA=$ST/sigma_hansch1991.csv          # B3 · B6 (사용자에게 받는다; 없으면 B3 · B6 보류)
OUT_FINAL=$ST/results_final
# 아카이브 (B2 결측 원인 · B5 GEDT · Phase C A_idx 대조): 누리온 묶음 tar 와 UBAI 작업 폴더
NURION_TARS=$REPO/../eda-asm-prediction/D1_build/nurion
[ -d "$NURION_TARS" ] || NURION_TARS=$REPO/D1_build/nurion
D1_JOBS=$B/d1_scratch/jobs
mkdir -p "$ST/logs"
cd "$REPO" || exit 2
