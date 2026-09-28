#!/bin/bash
# (copy of /gpfs/tmp_cpu2/yeseo1ee/espley_xtb/publish_rev3.sh, the job that produced the rev 3 results/)
# rev 3 (SMD relabel, option-① labels, c_ghost target): after s04/s05, copy the ML
# outputs into the repo results/, rerun the extra analyses (s06/s07) on them and
# write a rev3-vs-rev2 comparison table. Read-only on the ML outputs; overwrites
# only files under analysis/espley_xtb_repro/results/. Safe to re-run.
#SBATCH --job-name=xtb_publish
#SBATCH --time=48:00:00
#SBATCH --partition=cpu1,cpu2
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/logs/publish_rev3.%j.out
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
R=/gpfs/tmp_cpu2/yeseo1ee/espley_xtb
E=/home1/yeseo1ee/projects/eda-asm-prediction/analysis/espley_xtb_repro
RES=$E/results
fail() { echo "### STOPPED: $1"; exit 1; }

echo "### [1/5] gate on ML outputs  $(date -Is)"
nj=$(ls $R/ml_targets/target_*.json 2>/dev/null | wc -l); np_=$(ls $R/ml_targets/preds_*.parquet 2>/dev/null | wc -l)
[ "$nj" -eq 12 ] && [ "$np_" -eq 12 ] || fail "expected 12 target json + 12 preds, got $nj / $np_"
ls $R/ml_targets | grep -q barrier_eda && fail "stale barrier_eda file in ml_targets"
python - <<PY || fail "ml_report gate"
import json
r = json.load(open("$R/ml_report.json"))["per_target"]
assert len(r) == 12 and "dft_c_ghost_kcal" in r and "dft_barrier_eda" not in r, sorted(r)
n = {t: v["n_rows_ml"] for t, v in r.items()}
assert len(set(n.values())) == 1, n
print("ml_report ok:", len(r), "targets, n_rows_ml =", set(n.values()))
PY

echo "### [2/5] copy outputs into repo results/  $(date -Is)"
cp -f $R/xtb_features.parquet $R/ml_report.json $R/ml_table_espley.csv $R/predictions.parquet $RES/
rm -f $RES/ml_targets/target_11_dft_barrier_eda.json
cp -f $R/ml_targets/target_*.json $RES/ml_targets/
mkdir -p $RES/figures && cp -f $R/figures/*.png $RES/figures/
ls -la $RES $RES/ml_targets $RES/figures

echo "### [3/5] s06 analyze_extra  $(date -Is)"
cd $E && python analyze_extra.py || fail "analyze_extra"

echo "### [4/5] s07 evaluate_pairs  $(date -Is)"
cd $E && python evaluate_pairs.py results /gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv || fail "evaluate_pairs"
mv -f $E/split_metrics.csv $E/margin_calibration.csv $E/mmp_pairs_v2.csv $RES/

echo "### [5/5] rev3 vs rev2 table  $(date -Is)"
python - <<PY || fail "comparison table"
import pandas as pd
new = pd.read_csv("$RES/ml_table_espley.csv"); old = pd.read_csv("$R/rev2_backup_20260915/ml_table_espley.csv")
k = ["protocol", "feature_set", "target", "model"]
m = new.merge(old[k + ["test_mae", "test_r2"]], on=k, how="left", suffixes=("", "_rev2"))
m.round(4).to_csv("$RES/rev3_vs_rev2.csv", index=False)
a = m[(m.protocol == "A") & (m.feature_set == "ESPLEY73")]
piv = a.pivot(index="target", columns="model", values="test_mae")
piv2 = a.pivot(index="target", columns="model", values="test_mae_rev2")
print("Protocol A, ESPLEY73, test MAE (kcal/mol)  rev3 [rev2]")
for t in piv.index:
    print(f"  {t:20s} " + "  ".join(f"{c} {piv.loc[t, c]:.2f} [{piv2.loc[t, c]:.2f}]" for c in piv.columns))
PY
echo "### DONE  $(date -Is)"
