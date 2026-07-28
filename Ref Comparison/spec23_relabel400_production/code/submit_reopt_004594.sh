#!/bin/bash
#SBATCH --job-name=s23_reopt_004594
#SBATCH --partition=cpu2
#SBATCH --time=48:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/s23_reopt_%j.out
#SBATCH --error=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/Ref_Comparison_slurm/s23_reopt_%j.err

# Re-optimise dipolar_004594 fragA with TightOpt starting from the current
# opt.xyz. If it stabilises at the same energy → previous opt was already
# at the minimum and strain_A < 0 is a BSSE-in-strain artifact. If it
# lowers energy → previous opt was suboptimal.

set -uo pipefail
WORKDIR=/gpfs/tmp_cpu2/yeseo1ee/spec23_wb97x3c_workdir/dipolar_004594/fragA_opt

OPENMPI=/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.6
HWLOC=/opt/ohpc/pub/libs/hwloc
ORCA_ROOT=/home1/yeseo1ee/orca_6_1_1_avx2
export PATH="$OPENMPI/bin:$ORCA_ROOT:$PATH"
export LD_LIBRARY_PATH="$OPENMPI/lib:$HWLOC/lib:$ORCA_ROOT:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_mca_base_env_list="LD_LIBRARY_PATH;PATH"

cd "$WORKDIR"

# Backup old outputs
mkdir -p _backup_first_opt
mv fragA_opt.out fragA_opt.xyz fragA_opt.gbw fragA_opt.property.txt fragA_opt.densitiesinfo _backup_first_opt/ 2>/dev/null || true
rm -f -- *.tmp *.densities *.bas[0-9]* *.SHARKINP.tmp 2>/dev/null || true

# Read the previous optimised geom (in _backup) and build a TightOpt input
python3 - <<'PY_BUILD'
import json
from pathlib import Path
wd = Path("/gpfs/tmp_cpu2/yeseo1ee/spec23_wb97x3c_workdir/dipolar_004594/fragA_opt")
prev_xyz = wd / "_backup_first_opt" / "fragA_opt.xyz"
lines = prev_xyz.read_text().splitlines()
n = int(lines[0].strip())
atoms = lines[2:2+n]
# READ charge/mult from meta.json — dipolar_004594 fragA is charge=-1 anion.
meta = json.loads((wd / "meta.json").read_text())
ch, mult = int(meta["charge"]), int(meta["multiplicity"])
print(f"charge={ch} multiplicity={mult}")
route = "! wB97X-3c TightOpt TightSCF NoSym"
out = [
    route,
    "%maxcore 3500",
    "%pal nprocs 4 end",
    "%scf",
    "  MaxIter 500",
    "end",
    "%geom",
    "  MaxIter 300",
    "  TolE  5e-7",
    "  TolRMSG 5e-5",
    "  TolMaxG 1e-4",
    "end",
    f"* xyz {ch} {mult}",
]
for a in atoms:
    out.append("  " + a)
out.append("*")
(wd / "fragA_opt.inp").write_text("\n".join(out) + "\n")
print("wrote TightOpt input from previous opt.xyz")
PY_BUILD

echo "=== [$(date -Is)] running TightOpt ==="
"$ORCA_ROOT/orca" fragA_opt.inp > fragA_opt.out 2>&1

echo "=== [$(date -Is)] result ==="
grep -E "OPTIMIZATION RUN DONE|FINAL SINGLE POINT|TERMINATED" fragA_opt.out | tail -5
