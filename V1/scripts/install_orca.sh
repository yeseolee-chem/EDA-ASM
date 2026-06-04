#!/bin/bash
# Install ORCA 6.1.1 (Linux x86-64 AVX2, shared, OpenMPI 4.1.8 ABI) under
# $HOME/orca6/orca_6_1_1/. Idempotent: re-running on an existing install
# only re-runs the smoke tests.
#
# Usage:
#   bash install_orca.sh                  # auto-detects tarball in $HOME
#   bash install_orca.sh /path/to/tar.xz  # explicit path
#
# After install, activate ORCA in any new shell via:
#   source $HOME/orca6/orca-env.sh

set -euo pipefail

ORCA_ROOT="$HOME/orca6"
ORCA_DIR="$ORCA_ROOT/orca_6_1_1"
ENV_FILE="$ORCA_ROOT/orca-env.sh"
EXPECTED_BIN="$ORCA_DIR/orca"

# Default source candidates — accept either the .tar.xz archive
# or a pre-extracted directory (e.g. Windows auto-extracted before scp).
DEFAULT_CANDIDATES=(
  "$HOME/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2.tar.xz"
  "$HOME/Downloads/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2.tar.xz"
  "$HOME/downloads/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2.tar.xz"
  "$HOME/orca_6_1_1_linux_x86-64_shared_openmpi418_avx2"
)

log() { printf '[install_orca] %s\n' "$*"; }
die() { printf '[install_orca][ERROR] %s\n' "$*" >&2; exit 1; }

# ---- 1) locate source (tarball OR extracted directory) ---------------------
SRC=""
if [ $# -ge 1 ]; then
  [ -e "$1" ] || die "source not found: $1"
  SRC="$1"
else
  for c in "${DEFAULT_CANDIDATES[@]}"; do
    if [ -e "$c" ]; then SRC="$c"; break; fi
  done
fi

if [ -z "$SRC" ] && [ ! -x "$EXPECTED_BIN" ]; then
  die "no ORCA tarball OR extracted directory found in standard locations; \
pass path explicitly. checked: ${DEFAULT_CANDIDATES[*]}"
fi

# ---- 2) place into $ORCA_DIR ------------------------------------------------
if [ -x "$EXPECTED_BIN" ]; then
  log "ORCA already present at $ORCA_DIR — skipping placement"
elif [ -d "$SRC" ]; then
  # Pre-extracted directory: move/copy in place
  if [ -x "$SRC/orca" ]; then
    log "source is pre-extracted directory: $SRC"
    mkdir -p "$ORCA_ROOT"
    log "moving $SRC -> $ORCA_DIR"
    mv "$SRC" "$ORCA_DIR"
  else
    die "directory $SRC has no executable 'orca' binary at its top level — \
check contents (Windows extraction may have stripped permissions or lost files)"
  fi
elif [ -f "$SRC" ]; then
  log "extracting $SRC -> $ORCA_ROOT/"
  mkdir -p "$ORCA_ROOT"
  tar xJf "$SRC" -C "$ORCA_ROOT/"
  # After extraction, find the top-level dir (avoid SIGPIPE on `tar tJf | head`)
  TOP="$(ls "$ORCA_ROOT" | grep -m1 '^orca')"
  log "extracted top-level dir: $TOP"
  if [ -n "$TOP" ] && [ "$TOP" != "orca_6_1_1" ] && [ -d "$ORCA_ROOT/$TOP" ]; then
    log "renaming $TOP -> orca_6_1_1"
    mv "$ORCA_ROOT/$TOP" "$ORCA_DIR"
  fi
else
  die "source path $SRC is neither a file nor a directory"
fi

[ -x "$EXPECTED_BIN" ] || {
  # Windows may have stripped +x — try to restore on binaries/libs
  log "ORCA binary lacks +x; attempting chmod fix (likely Windows-stripped perms)"
  find "$ORCA_DIR" -maxdepth 1 -type f \( -name 'orca*' -o -name '*.so*' \) -exec chmod +x {} \; 2>/dev/null || true
  [ -x "$EXPECTED_BIN" ] || die "still no executable at $EXPECTED_BIN after chmod fix"
}

# ---- 3) write env activation script -----------------------------------------
log "writing env activation script -> $ENV_FILE"
cat > "$ENV_FILE" <<'EOF'
# ORCA 6.1.1 (Linux x86-64 AVX2) env activation
# Source from any shell:  source $HOME/orca6/orca-env.sh
export ORCA_DIR="$HOME/orca6/orca_6_1_1"
export PATH="$ORCA_DIR:$PATH"
export LD_LIBRARY_PATH="$ORCA_DIR:${LD_LIBRARY_PATH:-}"
# Use the cluster's OpenMPI 4.1.x; ORCA was built against 4.1.8 ABI.
# 4.1.5 module is normally ABI-compatible (libmpi.so.40 SONAME stable across 4.1.x).
if command -v module >/dev/null 2>&1; then
  module load openmpi/4.1.5 2>/dev/null || true
fi
EOF

# ---- 4) smoke 1: orca --version ---------------------------------------------
log "sourcing env and probing 'orca --version'"
# shellcheck disable=SC1090
source "$ENV_FILE"
ORCA_BIN="$(command -v orca || true)"
[ -n "$ORCA_BIN" ] || die "orca not on PATH after env source"

# Sanity: must NOT be /usr/bin/orca (GNOME screen reader)
if [ "$ORCA_BIN" = "/usr/bin/orca" ]; then
  die "PATH is still picking up /usr/bin/orca (GNOME screen reader). \
check ENV_FILE ordering."
fi
log "orca binary -> $ORCA_BIN"

# ldd check: confirm libmpi.so.40 resolves
log "ldd $ORCA_BIN | head"
ldd "$ORCA_BIN" 2>&1 | head -20 || true

# orca --version may try to MPI-init; some builds print version on stderr.
"$ORCA_BIN" --version 2>&1 | head -5 || log "(orca --version exited non-zero; OK if it's the 'no input file' mode)"

# ---- 5) smoke 2: H2 wB97X-3c single point -----------------------------------
SMOKE_DIR="$ORCA_ROOT/smoke_h2"
mkdir -p "$SMOKE_DIR"
cat > "$SMOKE_DIR/h2.inp" <<'EOF'
! wB97X-3c TightSCF
%pal nprocs 1 end
* xyz 0 1
H  0.0  0.0  0.0
H  0.0  0.0  0.74
*
EOF

cd "$SMOKE_DIR"
log "running H2 / wB97X-3c smoke (nprocs=1)..."
if "$ORCA_BIN" h2.inp > h2.out 2> h2.err; then
  if grep -q 'FINAL SINGLE POINT ENERGY' h2.out; then
    E=$(grep 'FINAL SINGLE POINT ENERGY' h2.out | tail -1 | awk '{print $NF}')
    log "smoke OK — final SP energy = $E Eh"
  else
    log "smoke RAN but no 'FINAL SINGLE POINT ENERGY' found — investigate h2.out"
    tail -20 h2.out
    exit 2
  fi
else
  log "smoke FAILED — see $SMOKE_DIR/h2.err and h2.out tails:"
  echo "--- h2.err tail ---"; tail -20 h2.err || true
  echo "--- h2.out tail ---"; tail -20 h2.out || true
  exit 3
fi

log "ORCA install + smoke complete."
log "Activate in future shells with:  source $ENV_FILE"
