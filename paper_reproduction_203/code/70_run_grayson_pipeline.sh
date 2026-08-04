#!/bin/bash
#SBATCH --job-name=paper203_ml
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/paper_reproduction_203/logs/pipeline_%j.log
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/paper_reproduction_203/logs/pipeline_%j.err
#
# Run the full Grayson-style ML pipeline on the 203-rxn snapshot:
#   1. f_extract.py × 3 (ts, gs, dist_gs)          → tt_ts.pkl, tt_gs.pkl, tt_dist_gs.pkl
#   2. f_extract.py -c (collate + join barriers)   → tt_features.pkl
#   3. 45_pre_ml_filter.py                         → manual_dipolar.pkl
#   4. hyp_tuning.py <manual_dipolar.pkl>          → checkpoint.pkl / hps.pkl
#   5. ml_analysis.py                              → ml_results.pkl
#   6. 80_visualize.py                             → parity plots + REPORT.md
#
# Grayson NN branch (Keras 3) is skipped — sklearn only (Ridge/KRR/SVR/RF).

set -euo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

ROOT=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/paper_reproduction_203
WORK="$ROOT/pipeline_work"
GRAY="$ROOT/grayson_code"
CODE="$ROOT/code"

echo "[$(date +%H:%M:%S)] === setup ==="
mkdir -p "$WORK" "$ROOT/logs"
cd "$WORK"

# Symlink logs (Grayson chdirs to setup['path'] and iterates *.log/*.out)
ln -sfn "$ROOT/gaussian_inputs/am1_logs/ts"      ts
ln -sfn "$ROOT/gaussian_inputs/am1_logs/gs"      gs
ln -sfn "$ROOT/gaussian_inputs/am1_logs/dist_gs" dist_gs

# Symlink common_atoms/mapping/mol_types (f_extract loads by relative filename)
ln -sfn "$ROOT/grayson_pkls/common_atoms.pkl" common_atoms.pkl
ln -sfn "$ROOT/grayson_pkls/mapping.pkl"      mapping.pkl
ln -sfn "$ROOT/grayson_pkls/mol_types.pkl"    mol_types.pkl

echo "[$(date +%H:%M:%S)] === Step 1: f_extract × 3 ==="
for TYPE in ts gs dist_gs; do
    echo "[$(date +%H:%M:%S)]   ---- extracting $TYPE ----"
    cat > config.yaml <<EOF
dataset: tt
type: $TYPE
path: $WORK/$TYPE
spe: False
ca_from_file: True
ca_file: $WORK/common_atoms.pkl
mapping_file: $WORK/mapping.pkl
mol_types: $WORK/mol_types.pkl
verbose: 1
EOF
    python "$GRAY/feature_extraction/f_extract.py"
done

echo "[$(date +%H:%M:%S)] === Step 2: f_extract collate ==="
cat > config.yaml <<EOF
gs_path: $WORK/tt_gs.pkl
ts_path: $WORK/tt_ts.pkl
dist_gs_path: $WORK/tt_dist_gs.pkl
am1_barriers_path: $ROOT/barriers/am1_barriers.pkl
dft_barriers_path: $ROOT/barriers/dft_barriers.pkl
filename: tt_features.pkl
EOF
python "$GRAY/feature_extraction/f_extract.py" -c

echo "[$(date +%H:%M:%S)] === Step 3: feature filter ==="
python "$CODE/45_pre_ml_filter.py" \
    --input "$WORK/tt_features.pkl" \
    --output "$WORK/manual_dipolar.pkl"

echo "[$(date +%H:%M:%S)] === Step 4: hyperparameter tuning (sklearn only, NN skipped) ==="
# hyp_tuning.py takes pkl path as CLI arg. We disable NN by patching its main
# to skip the tf branch (Keras 3 broken).
cd "$WORK"
# The script writes checkpoint.pkl to cwd; run there.
python "$GRAY/hyperparameter_tuning/hyp_tuning.py" "$WORK/manual_dipolar.pkl" 2>&1 | tee "$ROOT/logs/hyp_tuning.log" || true

# Ensure hps.pkl exists — lift from checkpoint.pkl if only checkpoint written
if [ -f "$WORK/checkpoint.pkl" ] && [ ! -f "$WORK/hps.pkl" ]; then
    cp "$WORK/checkpoint.pkl" "$WORK/hps.pkl"
fi

echo "[$(date +%H:%M:%S)] === Step 5: ML analysis ==="
python "$GRAY/machine_learning/ml_analysis.py" 2>&1 | tee "$ROOT/logs/ml_analysis.log" || true

echo "[$(date +%H:%M:%S)] === Step 6: visualize + REPORT ==="
python "$CODE/80_visualize.py" || true

echo "[$(date +%H:%M:%S)] === DONE ==="
ls -la "$WORK/"
