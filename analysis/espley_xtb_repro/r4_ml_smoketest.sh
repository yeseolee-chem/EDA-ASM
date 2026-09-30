#!/bin/bash
#SBATCH --job-name=r4_ml_smoke
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
# Phase 3 pre-flight, no training (replaces s03_smoketest.sh): the rev 4 ML scripts compile and import, grid keys
# resolve on _make_pipe, the train_ml_single source slice exec'd by analyze_extra.py / evaluate_pairs.py still yields
# FEATURE_SETS, and train_ml_single.load_ml accepts rows_rev4.csv on both parquets (the exact rows the array trains on).
# Run after r4_make_rows.sh. Last line SMOKE_OK = pass.
#   sbatch --partition=cpu1,cpu2 --output=$R4_SCRATCH/logs/ml_smoke.%j.out r4_ml_smoketest.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export ESPLEY_OUT=${ESPLEY_OUT:-/gpfs/tmp_cpu2/yeseo1ee/espley_xtb}
export ESPLEY_ROWS=$CODE/results_rev4/rows_rev4.csv
python -m py_compile train_ml_single.py aggregate_ml.py plot_results.py make_rows_rev4.py rev4_tables.py || exit $?
python - <<'EOF'
import json, os
from pathlib import Path
import sklearn, xgboost
import train_ml_single as t
import aggregate_ml, plot_results, make_rows_rev4, rev4_tables  # noqa: F401  (module-level code only)

print("sklearn", sklearn.__version__, "xgboost", xgboost.__version__)
tg = t.TARGET_SETS["rev4"]
assert len(tg) == 9 and set(tg) <= set(t.TARGETS), tg
print("rev4 targets:", tg)
print("feature counts:", {k: len(v) for k, v in t.FEATURE_SETS.items()})
assert [len(v) for v in t.FEATURE_SETS.values()] == [46, 54, 73]
for name, (est, grid) in t.GRIDS.items():
    params = t._make_pipe(est).get_params(deep=True)
    missing = [k for k in grid if k not in params]
    assert not missing, f"{name}: unknown grid keys {missing}"
    print(f"{name}: grid keys ok {sorted(grid)}")
src = Path("train_ml_single.py").read_text()
ns = {}
exec(src[src.index("DIST11"):src.index("TARGETS =")], ns)
assert ns["FEATURE_SETS"] == t.FEATURE_SETS, "exec slice DIST11..TARGETS = no longer reproduces FEATURE_SETS"
print("exec slice (analyze_extra / evaluate_pairs) ok")

root, rows = Path(os.environ["ESPLEY_OUT"]), Path(os.environ["ESPLEY_ROWS"])
rep = json.loads(rows.with_suffix(".json").read_text())
ml = {}
for g in ("g0", "g1"):
    ml[g], info = t.load_ml(root / f"xtb_features_{g}.parquet", rows, tg, g)
    assert info["rows_sha256"] == rep["rows_sha256"], (g, info["rows_sha256"], rep["rows_sha256"])
    tr, te = t.split_80_10_10(len(ml[g]), t.SEEDS[0])
    print(f"{g}: {len(ml[g])} rows (rows_rev4.json n_final {rep['n_final']}), seed {t.SEEDS[0]} train {len(tr)} test {len(te)}")
assert len(ml["g0"]) == rep["n_final"]
assert (ml["g0"].rxn_id.to_numpy() == ml["g1"].rxn_id.to_numpy()).all(), "row order differs between G0 and G1"
d = (ml["g0"][tg] - ml["g1"][tg]).abs().to_numpy().max()
assert d <= 1e-9, f"targets differ between G0 and G1: max |diff| {d}"
print(f"G0 / G1 same rxn order, targets max |diff| {d:.3g}")
bd = (ml["g0"].b_disp - ml["g0"].dft_disp_dft).abs().mean(), (ml["g1"].b_disp - ml["g1"].dft_disp_dft).abs().mean()
print(f"MAE(b_disp - dft_disp_dft) on the ML rows: G0 {bd[0]:.3g} (identity expected), G1 {bd[1]:.3g}")
print("SMOKE_OK")
EOF
