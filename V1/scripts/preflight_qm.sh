#!/bin/bash
# Preflight environment discovery per V1 Claisen ASR/EDA Spec §2.2.
# Do NOT assume ORCA/ADF availability — this script reports what is found
# and what is missing so the user can resolve before any QM computation.
set -uo pipefail

echo "## preflight $(date -Is)"
echo "host: $(hostname)"
echo "user: $(whoami)"
echo "pwd:  $(pwd)"
echo

echo "### 1) module avail (QM-related)"
module avail 2>&1 | grep -Ei 'orca|adf|ams|scm' || echo "NO QM MODULE matched (orca/adf/ams/scm)"
echo

echo "### 2) ORCA discovery"
# Source local ORCA install if present (selective DFT-only install at $HOME/orca6)
if [ -f "$HOME/orca6/orca-env.sh" ]; then
  # shellcheck disable=SC1090
  source "$HOME/orca6/orca-env.sh"
fi
ORCA_BIN="$(command -v orca || true)"
if [ -n "$ORCA_BIN" ] && [ "$ORCA_BIN" != "/usr/bin/orca" ]; then
  echo "orca: $ORCA_BIN"
  echo "ORCA_DIR=${ORCA_DIR:-<unset>}"
  # ORCA's banner prints on stdout when run with no args; --version may exit non-zero
  "$ORCA_BIN" --version 2>/dev/null | grep -E 'Program Version|ORCA' | head -3 || echo "(version banner not parsed cleanly; orca itself is on PATH)"
else
  echo "orca QM: NOT FOUND (or only GNOME screen reader at /usr/bin/orca)"
fi
echo

echo "### 3) ADF / AMS discovery"
echo "AMSBIN=${AMSBIN:-<unset>}"
if [ -n "${AMSBIN:-}" ] && [ -x "$AMSBIN/ams" ]; then
  echo "ams: $AMSBIN/ams (executable)"
  "$AMSBIN/ams" --version 2>/dev/null | head -3 || true
else
  echo "ams: NOT FOUND (set AMSBIN / module load required)"
fi
echo "SCM_PATH=${SCM_PATH:-<unset>}"
echo "SCMLICENSE=${SCMLICENSE:-<unset>}"
echo

echo "### 4) Python stack"
python -c "import sys; print('python', sys.version.split()[0], sys.executable)" 2>/dev/null || echo "python: NOT FOUND"
python -c "import rdkit; print('rdkit', rdkit.__version__)" 2>/dev/null || echo "rdkit: MISSING"
python -c "import scm.plams as p; print('plams', getattr(p, '__version__', '?'))" 2>/dev/null || echo "plams: MISSING"
python -c "import numpy, pandas, pyarrow; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'pyarrow', pyarrow.__version__)" 2>/dev/null || echo "numpy/pandas/pyarrow: MISSING"
python -c "import d2af" 2>/dev/null && echo "d2af: importable" || echo "d2af: not importable (manual fragmentation fallback)"
echo

echo "### 5) SLURM partitions"
sinfo -o "%P %a %D %T %c %m" 2>/dev/null | head -40 || echo "sinfo: NOT FOUND"
echo

echo "### 6) Conda environments"
if command -v conda >/dev/null 2>&1; then
  conda env list 2>/dev/null | head -20 || true
else
  echo "conda: NOT FOUND in PATH"
fi
echo

echo "## preflight complete $(date -Is)"
