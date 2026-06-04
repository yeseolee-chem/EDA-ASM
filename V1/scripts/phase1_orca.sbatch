#!/bin/bash
# Phase 1 — ORCA wB97X-3c geometry pipeline for ONE substrate.
# Runs sequentially: reactant Opt+Freq -> relaxed scan -> TS OptTS+Freq.
# Substrate id passed via --export=ID=<id>
#
# Submit:
#   sbatch --job-name=v1cl_<id>_p1 --export=ID=<id> phase1_orca.sbatch
#SBATCH --nodes=1
#SBATCH --partition=cpu1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH -o ./_out/%x.%j.out
#SBATCH -e ./_err/%x.%j.err

set -uo pipefail

: "${ID:?ID must be set via --export=ID=<substrate_id>}"

source "$HOME/orca6/orca-env.sh"
export OMP_NUM_THREADS=1

PROJECT="/gpfs/home1/yeseo1ee/projects/v1-claisen-asr"
RUN_DIR="$PROJECT/runs/$ID/orca"
cd "$RUN_DIR"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

log "=== Phase 1 start, ID=$ID ==="
log "host=$(hostname)  nproc=${SLURM_CPUS_PER_TASK:-?}  pwd=$(pwd)"
log "orca=$(command -v orca)"

# ---------- 1a. reactant Opt + Freq ----------
if [ -f .done_reactant ]; then
    log "1a) reactant: already done, skip"
else
    log "1a) reactant Opt+Freq starting"
    "$ORCA_DIR/orca" reactant.inp > reactant.out 2> reactant.err
    rc=$?
    log "1a) reactant orca exit=$rc"
    if ! grep -q 'ORCA TERMINATED NORMALLY' reactant.out; then
        log "1a) reactant FAILED — see reactant.out tail:"
        tail -30 reactant.out
        exit 11
    fi
    touch .done_reactant
fi

# Sanity: reactant.xyz (optimized geom) must exist for the scan
if [ ! -f reactant.xyz ]; then
    log "ERROR: reactant.xyz not produced by Opt. searching alternatives..."
    ls -1 *.xyz | head -20
    exit 12
fi

# Reactant freq sanity: 0 imaginary modes (real minimum)
NIMAG_R=$(grep -c '^\s*[0-9]\+:\s*-[0-9]' reactant.out || true)
log "1a) reactant n_imag_modes = $NIMAG_R (expect 0)"

# ---------- 1b. relaxed scan C1-C6 ----------
# Note: ORCA may exit non-zero if the last scan point fails inner-loop opt
# (commonly the most compressed r=1.9 Å point). We salvage converged points
# regardless of ORCA's exit code by parsing the energies embedded in each
# `ts_scan.NNN.xyz` title line.
if [ -f .done_scan ]; then
    log "1b) scan: already done, skip"
else
    log "1b) relaxed scan starting"
    "$ORCA_DIR/orca" ts_scan.inp > ts_scan.out 2> ts_scan.err
    rc=$?
    log "1b) scan orca exit=$rc"
    log "1b) (continuing regardless; will salvage from per-step xyz files)"
fi

# Extract ts_guess.xyz from the highest-energy converged scan point.
log "1b) selecting TS guess from scan"
python3 - <<'PYEOF' || exit 22
import re, shutil, sys
from pathlib import Path

# ORCA writes ts_scan.NNN.xyz for each converged scan step. The xyz title
# line of each contains "E <energy_in_Eh>" — robust to relaxscanact.dat
# being absent (ORCA may fail to write it when a later step crashes).
xyz_files = sorted(Path('.').glob('ts_scan.[0-9][0-9][0-9].xyz'))
if not xyz_files:
    print('ERROR: no ts_scan.NNN.xyz produced — scan never started', file=sys.stderr)
    sys.exit(1)
print(f'found {len(xyz_files)} converged scan step xyz files')

pts = []
for fp in xyz_files:
    head = fp.read_text().splitlines()[:2]
    if len(head) < 2:
        continue
    m = re.search(r'E\s+(-?[0-9.]+)', head[1])
    if not m:
        continue
    pts.append((fp, float(m.group(1))))

if not pts:
    print('ERROR: could not parse any step energies', file=sys.stderr)
    sys.exit(1)

max_fp, e_max = max(pts, key=lambda x: x[1])
print(f'scan max-E among {len(pts)} converged steps: {max_fp.name} E={e_max:.6f} Eh')
shutil.copy(max_fp, 'ts_guess.xyz')
print(f'copied {max_fp.name} -> ts_guess.xyz')

# Sanity: max should not be the first or last step (suggests scan was cut short)
idx_max = xyz_files.index(max_fp) + 1
if idx_max == 1:
    print(f'WARN: max-E at first step — energy still rising near reactant; '
          f'TS may need a wider scan range', file=sys.stderr)
elif idx_max == len(xyz_files):
    print(f'WARN: max-E at last step ({idx_max}/{len(xyz_files)}) — '
          f'TS may lie beyond scan range', file=sys.stderr)
PYEOF

if [ ! -f ts_guess.xyz ]; then
    log "ERROR: ts_guess.xyz not produced"
    exit 22
fi

touch .done_scan

# ---------- 1c. TS OptTS + Freq ----------
if [ -f .done_ts ]; then
    log "1c) TS: already done, skip"
else
    log "1c) TS OptTS+Freq starting"
    "$ORCA_DIR/orca" ts.inp > ts.out 2> ts.err
    rc=$?
    log "1c) TS orca exit=$rc"
    if ! grep -q 'ORCA TERMINATED NORMALLY' ts.out; then
        log "1c) TS FAILED — see ts.out tail:"
        tail -40 ts.out
        exit 31
    fi
    touch .done_ts
fi

# Verify TS: exactly 1 imaginary mode
NIMAG_T=$(grep -c '^\s*[0-9]\+:\s*-[0-9]' ts.out || true)
log "1c) TS n_imag_modes = $NIMAG_T (expect exactly 1)"

# Energies
E_R=$(grep 'FINAL SINGLE POINT ENERGY' reactant.out | tail -1 | awk '{print $NF}')
E_TS=$(grep 'FINAL SINGLE POINT ENERGY' ts.out | tail -1 | awk '{print $NF}')
log "E_reactant_wb97x3c = $E_R Eh"
log "E_TS_wb97x3c       = $E_TS Eh"

touch .done_orca
log "=== Phase 1 DONE for ID=$ID ==="
