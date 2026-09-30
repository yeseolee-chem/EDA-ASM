#!/bin/bash
#SBATCH --job-name=r4_g1_smoke
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
# Phase 1-1: G1 engine smoke test on rxn 20, 105 + 3 random ok rxns (seed 20260930).
# Submit from analysis/espley_xtb_repro/ of the rev 4 checkout:
#   sbatch --partition=<cpu1|cpu2> --output=$R4_SCRATCH/logs/g1_smoke.%j.out r4_g1_smoke.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
python -c "import yaml, rdkit, networkx; print('py env ok')"
echo "### otool_xtb:"; "$ORCA_DIR/otool_xtb" --version 2>&1 | grep -i version | head -3
echo "### xtb:"; "$XTB_BIN" --version 2>&1 | grep -i version | head -2
echo "### internet from compute node (Phase 4-4 prerequisite):"
curl -sS -m 20 -o /dev/null -w "researchdata.bath.ac.uk HTTP %{http_code}\n" https://researchdata.bath.ac.uk/ || echo "no internet"
python g1_geom.py smoke
