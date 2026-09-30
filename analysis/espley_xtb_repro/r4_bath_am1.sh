#!/bin/bash
#SBATCH --job-name=r4_bath_am1
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
# Phase 4-4: start structure of Espley's ds3 AM1 TS optimisations (bath_am1_start.py). Downloads happen only here,
# by HTTP Range from researchdata.bath.ac.uk: zip tail + central directory (~12 MB) + 20 AM1 logs (~5 MB compressed);
# the 4.3 GB archive is never fetched whole. Cache $R4_SCRATCH/bath/ is reused on resubmit; an existing
# results_rev4/espley_am1_start_rmsd.{csv,json} pair is skipped. No dependency on the G1 / feature / ML jobs.
# exit 2 = no readable archive holds ds3; exit 3 = ts_<n> -> Coley rxn mapping not established; exit 4 = no archive
# could be read (network / HTTP: resubmit). Diagnostics in the log and $R4_SCRATCH/bath/mapping_diagnostics.json.
#   sbatch --partition=cpu1,cpu2 --output=$R4_SCRATCH/logs/bath_am1.%j.out r4_bath_am1.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/r4_env.sh"
export ESPLEY_REPO_DATA=${ESPLEY_REPO_DATA:-/gpfs/tmp_cpu2/yeseo1ee/espley_compare}
python -c "import networkx, numpy, pandas; print('py env ok')"
curl -sS -m 20 -o /dev/null -w "researchdata.bath.ac.uk/1480 HTTP %{http_code}\n" \
    https://researchdata.bath.ac.uk/1480/ || echo "no internet"
echo "labels $ESPLEY_LABELS repo data $ESPLEY_REPO_DATA d1 $D1_SNAPSHOT cache $R4_SCRATCH/bath" \
     "-> results_rev4/espley_am1_start_rmsd.{csv,json}"
python bath_am1_start.py run
