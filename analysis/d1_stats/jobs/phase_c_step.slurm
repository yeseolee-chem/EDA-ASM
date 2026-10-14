#!/bin/bash
# Phase C (SPEC_D1_ALL) 단일 단계. 저장소 루트에서 제출한다. 순서와 제출 묶음은 analysis/d1_stats/RUNBOOK_D1_ALL.md.
#   prep     d1c_prep.py      확정 라벨 → D1 입력 (labels_g1in.json · prof/ · input_meta.csv · smiles.csv)   RELAY_DONE 필요
#   agg      d1c_aggregate.py xtb_slice 조각 18개 → $ESPLEY_OUT/xtb_features_g1.parquet (ext_features 입력)
#   probe    ext_features.py rxn — S 행 · Si 행 하나씩 먼저 (agg 뒤, ext 배열 전; D0 에 없던 원소)
#   merge    d1c_merge.py     SPEC 출력 parquet 2개 (job_id 키) + 게이트 C → results_final/GATE_C.md (실패면 rc 1 = STOP)
#   control  d1c_control.py   게이트 C (iii): D0 대조 2건을 D0 환경에서 같은 스크립트로 (+ G1 재최적화는 참고용)
#   sbatch -p cpu1,cpu2 --cpus-per-task=4 --mem=16G -o analysis/d1_stats/logs/c_%x_%j.out -J d1sC_prep analysis/d1_stats/jobs/phase_c_step.slurm prep
#SBATCH --job-name=d1sC
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
set -uo pipefail
STEP=${1:?prep | agg | probe | merge | control}
SUB=${SLURM_SUBMIT_DIR:?저장소 루트에서 제출}
if [ "$STEP" = control ]; then
    # D0 환경 그대로 (r4_env.sh 기본값): D1 경로 변수를 지우고 시작한다
    unset ESPLEY_LABELS ESPLEY_PROF ESPLEY_META ESPLEY_SMILES G1_ROOT ESPLEY_OUT R5_SCRATCH D1_FEAT_ROOT
    export ESPLEY_CODE=$SUB/analysis/espley_xtb_repro OMPI_MCA_btl_vader_single_copy_mechanism=none
    source "$ESPLEY_CODE/r4_env.sh" || exit 2
    export PY=/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/venv/bin/python
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XTB_THREADS=1 OMP_STACKSIZE=1G PYTHONWARNINGS=ignore
    OUT=$REPO/analysis/d1_stats/results_final; mkdir -p "$OUT"
    "$PY" "$REPO/analysis/d1_stats/d1c_control.py" "$OUT" --g1-rerun; rc=$?
    echo "control rc=$rc (0 = 게이트 C iii 통과)"; exit $rc
fi
source "$SUB/analysis/d1_stats/jobs/phase_c_env.sh" || exit 2
OUT=$ST/results_final; mkdir -p "$OUT"
B=/gpfs/tmp_cpu2/yeseo1ee/d1
D1R=$REPO/D1_build/d1_run; [ -d "$D1R" ] || D1R=$B/d1_run
case "$STEP" in
  prep)
    [ -f "$B/d1_scratch/queue/RELAY_DONE" ] || { echo "RELAY_DONE 없음 — D1 본계산이 끝나지 않았다"; exit 1; }
    SCAN=(); [ -s "$OUT/archive_scan.csv" ] && SCAN=(--scan "$OUT/archive_scan.csv")
    python "$ST/d1c_prep.py" "$D1R/results/labels_d1.json" "$D1R/results/geometries" "$D1R/d1_manifest.csv" "${SCAN[@]}" ;;
  agg)
    python "$ST/d1c_aggregate.py" ;;
  probe)
    # ext 배열 전에: S 가 있는 행 하나와 Si 가 있는 행 하나를 ext_features.py rxn 으로 먼저 (D0 에 없던 원소 — rev 5 B-0 smoke 는 D0 원소만 봤다).
    # 캐시는 D1 의 $R5_SCRATCH/ext/<rid>/ 에 남아 slice 가 그대로 다시 쓴다. ext_status 가 ok 가 아니면 rc 1 → ext 배열을 내기 전에 본다.
    [ -s "$ESPLEY_OUT/xtb_features_g1.parquet" ] || { echo "STOP: agg 를 먼저"; exit 2; }
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XTB_THREADS=1 OMP_STACKSIZE=1G PYTHONWARNINGS=ignore
    ids=$("$PY" - <<'EOF'
import json, os
from pathlib import Path
import pandas as pd
lab = json.load(open(os.environ["ESPLEY_LABELS"])); g1 = Path(os.environ["G1_ROOT"])
f = pd.read_parquet(Path(os.environ["ESPLEY_OUT"]) / "xtb_features_g1.parquet", columns=["rxn_id", "xtb_status"])
ok = set(f.rxn_id[f.xtb_status == "ok"].astype(int))
pick = {}
for d in sorted(lab, key=lambda d: int(d["rxn_id"])):
    r = int(d["rxn_id"])
    if r not in ok or not (g1 / str(r) / ".done").is_file():
        continue
    els = {l.split()[0] for l in (g1 / str(r) / "ts.xyz").read_text().splitlines()[2:] if l.strip()}
    for e in ("S", "Si"):
        if e in els and e not in pick:
            pick[e] = r
    if len(pick) == 2:
        break
print(" ".join(f"{e}:{r}" for e, r in pick.items()))
EOF
)
    echo "probe rxns: $ids"; rcp=0
    for er in $ids; do
        echo "=== ${er%%:*} rxn ${er#*:}"; "$PY" ext_features.py rxn "${er#*:}" > "$D1_FEAT_ROOT/logs/probe_${er#*:}.json" 2>&1 || rcp=1
        grep -E '"(ext_status|status_B[1-6])"' "$D1_FEAT_ROOT/logs/probe_${er#*:}.json"
    done
    [ -n "$ids" ] || { echo "S / Si 행을 찾지 못함"; rcp=1; }
    ( exit $rcp ) ;;
  merge)
    export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
    "$PY" "$ST/d1c_merge.py" "$OUT" ;;
  *) echo "STEP 은 prep | agg | probe | merge | control"; exit 2 ;;
esac
rc=$?; echo "step=$STEP rc=$rc $(date -Is)"; exit $rc
